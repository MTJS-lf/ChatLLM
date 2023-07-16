# -*- coding:utf-8 -*-

import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ChatGLM2DataSet(Dataset):
    """数据处理函数"""
    def __init__(self, data_path, tokenizer, max_len, max_src_len, prefix):
        max_tgt_len = max_len - max_src_len - 3
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt = tokenizer.build_prompt(src_tokens)
                prompt = prefix + prompt
                prompt_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=max_src_len)

                tgt_tokens = tokenizer.tokenize(sample["answer"])
                answer_ids = tokenizer.encode(text=prompt, add_special_tokens=False, truncation=True, max_length=max_tgt_len)

                context_length = len(prompt_ids)
                input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
                labels = [tokenizer.pad_token_id] * context_length + answer_ids + [tokenizer.eos_token_id]

                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len

                self.all_data.append(
                    {"text": sample["text"], "answer": sample["answer"], "input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


def coll_fn(batch):
    input_ids_list, labels_list = [], []
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=20003),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=20003)}
