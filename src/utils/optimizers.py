from config.optimizer import OptimizerConfig
import torch_optimizer
from torch import optim
from torch import nn


def get_optimizer_from_config(config: OptimizerConfig,
                              model: nn.Module) -> optim.Optimizer:
    if config.name == "lamb":
        return torch_optimizer.Lamb(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.name == "adamw":
        return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.name == "adam":
        return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.name == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr)
    else:
        raise ValueError(f"Unknown optimizer type {config.name}")