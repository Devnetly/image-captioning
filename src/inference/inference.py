import argparse
import os
import torch
import logging
import logging.config
import pandas as pd
import sys
sys.path.append('../..')
from dataclasses import dataclass
from src.utils import DatasetDescriptor,Tokenizer,read_json,ImageCaptionGenerator
from src.models import Transformer
from definitions import *
from src.datasets import ImageDirectory
from torchvision.transforms import Compose,Resize,ToTensor
from tqdm import tqdm
logging.basicConfig(level = logging.INFO, format=' %(name)s :: %(levelname)-8s :: %(message)s')

class GLOBAL:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 384

@dataclass
class Args:
    dataset : str
    model : str
    checkpoint : str
    source : str
    destination : str


def main(args : Args):

    logger = logging
    
    descriptor : DatasetDescriptor = DatasetDescriptor.get_by_name(args.dataset)
    tokenizer : Tokenizer = Tokenizer.load(descriptor.vocab_path)
    model_config = read_json(
        os.path.join(MODELS_DIR, args.model, "config.json")
    )

    model = Transformer(
        **model_config,
        vocab_size=len(tokenizer.vocab),
        device=GLOBAL.DEVICE,
        pad_idx=tokenizer.vocab.pad_idx
    ).to(GLOBAL.DEVICE)

    weights_path = os.path.join(MODELS_DIR, args.model,args.checkpoint)
    state_dict = torch.load(weights_path)
    msg = model.load_state_dict(state_dict)

    model = model.eval()

    logger.info(msg)

    preporcessor = Compose([
        Resize((GLOBAL.IMG_SIZE,GLOBAL.IMG_SIZE)),
        ToTensor()
    ])

    captions_generator = ImageCaptionGenerator(
        model, 
        tokenizer, 
        model_config['max_len'], 
        GLOBAL.DEVICE,
        preporcessor
    )

    dataset = ImageDirectory(root=args.source, return_name=True)

    df = {
        "name" : [],
        "caption" : []
    }

    with torch.inference_mode():
        
        for img, name in tqdm(dataset):

            caption = captions_generator.get_caption(img)

            df["name"].append(name)
            df["caption"].append(caption)

    df = pd.DataFrame(df)

    df.to_csv(args.destination, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',type=str,choices=DatasetDescriptor.LOOKUP.keys(), default="flickr30k")
    parser.add_argument('--model',type=str,choices=["transformer"], default="transformer")
    parser.add_argument('--checkpoint',type=str,required=True)
    parser.add_argument('--source',type=str)
    parser.add_argument('--destination',type=str,required=True)

    args = parser.parse_args()

    main(args)