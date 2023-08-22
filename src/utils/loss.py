from config.cdr_config import CDRConfig
import torch
from torch import Tensor
from torch import nn
from torch.nn import CrossEntropyLoss


# TODO: Change if use ner


class CIDGCNLossFn:
    def __init__(self, pos_weight, device):
        self.device = device
        weight = torch.tensor([1, pos_weight], dtype=torch.float32)
        if device == "cuda":
            weight = weight.cuda()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def __call__(self, logits, labels, label_mask):
        logits = torch.permute(logits, (0, 2, 1))
        losses = self.cross_entropy(logits, labels) * label_mask
        loss = torch.sum(losses, dim=-1)
        loss = torch.mean(loss)
        return loss


def get_loss_from_config(config: CDRConfig, device: str = "cpu"):
    pos_weight = 1 if 'pos_weight' not in config.train.loss.arguments else config.train.loss.arguments['pos_weight']
    return CIDGCNLossFn(pos_weight, device)
