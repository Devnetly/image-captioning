import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional,Callable,Any
from PIL import Image

class CaptionsDataset(Dataset):

    def __init__(self,
        root : str,
        captions_df : str,
        img_transform : Optional[Callable] = None,
        caption_transform : Optional[Callable] = None,
    ):
        
        super().__init__()
        
        self.root = root
        self.img_transform = img_transform
        self.caption_transform = caption_transform
        self.captions_df = captions_df

        self.df = pd.read_csv(self.captions_df)
        self.images = self.df['image'].tolist()
        self.captions = self.df['caption'].tolist()

    def __getitem__(self, index) -> Any:
        
        path = os.path.join(self.root, self.images[index])
        caption = self.captions[index]
        img = Image.open(path).convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.caption_transform is not None:
            caption = self.caption_transform(caption)

        return img,caption
    
    def __len__(self) -> int:
        return len(self.captions)