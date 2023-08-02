# coding=utf-8

import os
import numpy as np
import math
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping

import evaluate

import torch
from torch.utils.data import RandomSampler, DataLoader
sys.path.append("..")

from utils.util import *
from utils.dataset import *

from model.chatGLM2.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMForSequenceClassification
from model.chatGLM2.tokenization_chatglm import ChatGLMTokenizer
from model.chatGLM2.configuration_chatglm import ChatGLMConfig

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    is_torch_tpu_available,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": ("The model checkpoint for weights initialization.")
        }
    )
    
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples : Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "max train samples"
            )
        },
    )
    max_eval_samples : Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "max eval samples"
            )
        },
    )
    max_predict_samples : Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "max predict samples"
            )
        },
    )
    max_seq_len: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "max len"
            )
        },
    )
    max_src_len: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "max_src_len"
            )
        },
    )
    prompt_text: Optional[str] = field(default="", metadata={"help": ""})


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    predict_with_generate : Optional[bool] = field(default=False)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
         # If we pass only one argument to the script and it's the path to a json file,
         # let's parse it to get our arguments.
         model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
         model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config_kwargs = {
    }
    config = ChatGLMConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    model = ChatGLMForSequenceClassification.from_pretrained(model_args.model_name_or_path)
    print(model)
    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.model_name_or_path)
    lora_config = LoraConfig(r=training_args.lora_rank,
                             lora_alpha=training_args.lora_alpha,
                             target_modules=["query_key_value"],
                             lora_dropout=training_args.lora_dropout,
                             bias="none",
                             task_type="CAUSAL_LM",
                             inference_mode=False,
                            )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    if training_args.do_train:
        train_dataset = ChatGLM2DataSet(data_args.dataset_dir, tokenizer, data_args.max_seq_len, data_args.max_src_len, data_args.prompt_text)
    if training_args.do_eval:
        eval_dataset = ChatGLM2DataSet(data_args.dataset_dir, tokenizer, data_args.max_seq_len, data_args.max_src_len, data_args.prompt_text)
    if training_args.do_predict:
        predict_dataset = ChatGLM2DataSet(data_args.dataset_dir, tokenizer, data_args.max_seq_len, data_args.max_src_len, data_args.prompt_text)


    # Data collator
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # Metric
    metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset if training_args.do_train else None,
                      eval_dataset=eval_dataset if training_args.do_eval else None,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics if training_args.predict_with_generate else None
                      )
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
   
    results = {}
    max_seq_length = data_args.max_seq_len
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                        writer.write(f"{res}\n")
    return results
        
        

if __name__ == "__main__":
   main() 
