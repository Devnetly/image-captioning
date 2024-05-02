import torch

class TensorTuple:

    def __init__(self, t : tuple[torch.Tensor]) -> None:
        self.t = t

    def __getitem__(self, index : int):
        return self.t[index]
    
    def to(self, device : torch.device):
        
        r = []

        for e in self.t:
            e.to(device)
            r.append(e.to(device))

        return tuple(r)