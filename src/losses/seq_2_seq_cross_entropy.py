from torch import Tensor
from torch.nn import CrossEntropyLoss


class Seq2SeqCrossentropy(CrossEntropyLoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)

        return super().forward(input, target)