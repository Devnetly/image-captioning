import torch
from torch.nn import Module
from typing import Optional,Callable
from .tokenizer import Tokenizer
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    
    def batch_caption(self, loader : DataLoader) -> dict[str,str]:

        result = {
            "image" : [],
            "caption" : []
        }

        with torch.inference_mode():

            for x,names in tqdm(loader):

                x = x.to(self.device)

                i = 0
                flags = torch.zeros((x.shape[0],)).fill_(0).bool()
                stop = False
                captions = [[] for _ in range(x.shape[0])]

                sos = torch.zeros((x.shape[0],1)).long().fill_(self.tokenizer.vocab.sos_idx)
                padding = torch.zeros((x.shape[0], self.max_len - 2)).long().fill_(self.tokenizer.vocab.pad_idx)
                y = torch.cat([sos,padding], dim=1).to(self.device)

                while not stop:

                    y_hat = self.model((x, y))
                    y_hat = torch.nn.functional.softmax(y_hat, dim=2)

                    ids = torch.argmax(y_hat, dim=2)  
                    word_ids = ids[:,i]

                    y[:,i+1] = word_ids

                    for j,word_id in enumerate(word_ids):

                        word_id = word_id.item()
                        flags[j] = flags[j] or word_id == self.tokenizer.vocab.eos_idx

                        if not flags[j]:
                            captions[j].append(self.tokenizer.vocab.idx_2_str[word_id])
                    
                    stop = torch.all(flags) or i == self.max_len - 3

                    i += 1 

                captions = map(lambda caption : [caption[0].title()] + caption[1:],captions)
                captions = map(' '.join,captions)
                captions = map(lambda caption : caption + '.', captions)
                captions = list(captions)

                for j,name in enumerate(names):
                    result['caption'].append(captions[j])
                    result['image'].append(name)

        return result
