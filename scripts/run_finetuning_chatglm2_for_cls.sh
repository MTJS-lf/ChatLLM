lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=/mnt/data1/AIModel/chatglm2-6b/
chinese_tokenizer_path=/mnt/data1/AIModel/chatglm2-6b/
dataset_dir=/mnt/data1/Data/SFT/glm_sft/d2q_0.json
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
output_dir=../output_dir
peft_model=../models/peft/model/dir
validation_file=/mnt/data1/Data/SFT/glm_sft/d2q_1.json

deepspeed_config_file=deepspeed.json

#python finetuning_chatglm2.py \
torchrun --nnodes 1 --nproc_per_node 1 finetuning_chatglm2_for_cls.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --peft_path ${peft_model} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
