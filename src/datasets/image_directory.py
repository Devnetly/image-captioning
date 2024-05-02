import os
from torch.utils.data import Dataset
from typing import Optional,Callable,Any
from PIL import Image

class ImageDirectory(Dataset):

    DEFAULT_EXTS = ['jpg','jpeg','png']

    def __init__(self, 
        root : str, 
        exts : Optional[list[str]] = None,
        transform : Optional[Callable] = None
    ) -> None:
        
        super().__init__()

        self.root = root
        self.exts = exts if exts is not None else ImageDirectory.DEFAULT_EXTS
        self.images = self._find_images()
        self.transform = transform

    def _find_images(self) -> list[str]:
        
        files = os.listdir(self.root)
        images = filter(lambda f : os.path.splitext(f)[1] in self.exts,files)
        images = map(lambda f : os.path.join(self.root, f), images)
        images = list(images)

        return images

    def __getitem__(self, index : str) -> Any:
        
        path = self.images[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def __len__(self) -> int:
        return len(self.images)