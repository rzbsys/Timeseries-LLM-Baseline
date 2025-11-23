from typing import Callable, List, Dict, Optional

import torch
from transformers.models.llama import LlamaTokenizer

message_form = List[List[dict]]


def create_data_format(messages: List[str], roles: List[str]) -> message_form:
    assert len(messages) == len(roles), "Messages and roles must have the same length."
    formatted = []
    for msg, role in zip(messages, roles):
        formatted.append({"role": role, "content": msg})
    return formatted


def tokenize(tokenizer: LlamaTokenizer, messages: message_form, add_generation_prompt=False, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    results = tokenizer(inputs, padding="longest", truncation=False, return_tensors="pt")  # truncation=True, max_length =768,
    if device is not None:
        results = {k: v.to(device) for k, v in results.items()}
    return results
