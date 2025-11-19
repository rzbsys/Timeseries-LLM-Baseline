import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def freeze_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model

def unfreeze_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model

def to_device(batch, device):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch