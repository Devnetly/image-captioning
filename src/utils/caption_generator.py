import torch
from torch.nn import Module
from typing import Optional,Callable
from .tokenizer import Tokenizer
from PIL import Image

class ImageCaptionGenerator:

    def __init__(self,
        model : Module,
        tokenizer : Tokenizer,
        max_len : int,
        device : torch.device,
        preprocessor : Optional[Callable] = None,
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocessor = preprocessor
        self.device = device

    def get_caption(self, img : Image.Image) -> str:

        sos = torch.tensor([self.tokenizer.vocab.sos_idx])
        padding = torch.zeros(self.max_len - 2).long().fill_(self.tokenizer.vocab.pad_idx)
        y = torch.cat([sos,padding], dim=0).unsqueeze(0).to(self.device)

        x = torch.unsqueeze(self.preprocessor(img), 0).to(self.device)

        i = 0
        stop = False
        caption = []

        while not stop:

            y_hat = self.model((x, y))
            y_hat = torch.squeeze(y_hat)
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)

            words_ids = torch.argmax(y_hat, dim=1)  
            word_id = words_ids[i].item()

            y[:,i+1] = word_id

            if word_id != self.tokenizer.vocab.eos_idx:
                caption.append(self.tokenizer.vocab.idx_2_str[word_id])
                i += 1
            
            stop = word_id == self.tokenizer.vocab.eos_idx or  i == self.max_len - 2   

        caption[0] = caption[0].title()
        caption = ' '.join(caption) + '.'

        return caption
