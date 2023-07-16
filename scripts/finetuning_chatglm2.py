# coding=utf-8

import os
import numpy as np
import math
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping

import torch
from torch.utils.data import RandomSampler, DataLoader
sys.path.append("..")

from utils.util import *
from utils.dataset import *

from model.chatGLM2.modeling_chatglm import ChatGLMForConditionalGeneration
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
    max_len: Optional[int] = field(
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
    model = ChatGLMForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.model_name_or_path)
    lora_config = LoraConfig(r=training_args.lora_rank,
                             lora_alpha=training_args.lora_alpha,
                             lora_dropout=training_args.lora_dropout,
                             bias="none",
                             task_type="CAUSAL_LM",
                             inference_mode=False,
                            )
    model = get_peft_model(model, lora_config)
    print(model)
    print_trainable_parameters(model)
    print(tokenizer)

    train_dataset = ChatGLM2DataSet(data_args.dataset_dir, tokenizer, data_args.max_len, data_args.max_src_len, data_args.prompt_text)

    # Data collator
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

if __name__ == "__main__":
   main() 
