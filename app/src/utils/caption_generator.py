import torch
from torch.nn import Module
from typing import Optional,Callable
from .tokenizer import Tokenizer
from PIL import Image

class ImageCaptionGenerator:

    def __init__(self,
        model : Module,
        tokenizer : Tokenizer,
        img : Image.Image,
        max_len : int,
        device : torch.device,
        preprocessor : Optional[Callable] = None,
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.img = img
        self.max_len = max_len
        self.preprocessor = preprocessor
        self.device = device

        self.x, self.y = self._prepare()

    def _prepare(self) -> tuple[torch.Tensor,torch.Tensor]:

        sos = torch.tensor([self.tokenizer.vocab.sos_idx])
        padding = torch.zeros(self.max_len - 2).long().fill_(self.tokenizer.vocab.pad_idx)
        y = torch.cat([sos,padding], dim=0).unsqueeze(0).to(self.device)

        x = torch.unsqueeze(self.preprocessor(self.img), 0).to(self.device)

        return x,y
    
    def __getitem__(self, index : int) -> str:

        x, y = self.x, self.y
        i = index

        y_hat = self.model((x, y))
        y_hat = torch.squeeze(y_hat)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        words_ids = torch.argmax(y_hat, dim=1)  
        word_id = words_ids[i].item()

        self.y[:,i+1] = word_id

        if word_id == self.tokenizer.vocab.eos_idx or  i == self.max_len - 2:
            raise StopIteration()
                
        return self.tokenizer.vocab.idx_2_str[word_id]