import torch
from typing import Any
from torchmetrics.classification import MulticlassAccuracy
from torch import Tensor

class Seq2SeqAccuracy:

    def __init__(self, **kwargs) -> None:
        self.acc = MulticlassAccuracy(**kwargs)

    def __call__(self, input : Tensor, target : Tensor) -> Any:

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)

        return self.acc(input, target)
    
    def to(self, device : torch.device):
        self.acc = self.acc.to(device)
        return self