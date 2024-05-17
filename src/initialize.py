import sys
import argparse
import pandas as pd
import logging
import logging.config
sys.path.append('..')
from src.utils import Tokenizer,DatasetDescriptor,TextPreprocessor
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
logging.basicConfig(level = logging.INFO, format=' %(name)s :: %(levelname)-8s :: %(message)s')

@dataclass
class Args:
    dataset : str
    min_freq : int
    on_already_exists : str

def main(args : Args):

    logger = logging

    logger.info(f"Processing dataset : {args.dataset}")

    descriptor : DatasetDescriptor = DatasetDescriptor.get_by_name(args.dataset)

    logger.info(f"Loading dataframe from path : {descriptor.captions}")

    df = pd.read_csv(descriptor.captions)

    logger.info(f"Found : {len(df)} rows.")

    logger.info(f"Removing Any possible empty captions.")

    df.dropna(inplace=True)

    logger.info(f"Remained : {len(df)} rows.")

    logger.info("Data splitting")

    images = df['image'].value_counts().index.to_list()

    train_images,test_images = train_test_split(images, test_size=0.2)

    logger.info(f"Train data contains : {len(train_images)} unique images and test data contains : {len(test_images)} unique images.")

    train_df = df[df['image'].isin(train_images)]
    test_df = df[df['image'].isin(test_images)]

    logger.info(f"Train dataframe contains : {len(train_df)} rows and test dataframe contains : {len(test_df)} rows.")

    train_df.to_csv(descriptor.train_captions,index=False)
    test_df.to_csv(descriptor.test_captions,index=False)

    logger.info(f"Train dataframe was saved to : {descriptor.train_captions}")
    logger.info(f"Test dataframe was saved to : {descriptor.test_captions}")

    logger.info(f"Building vocabulary.")

    tokenizer = Tokenizer(preprocessor=TextPreprocessor())
    tokenizer.fit(train_df['caption'].tolist())

    logger.info(f"Vocab size : {len(tokenizer.vocab)}")

    tokenizer.save(descriptor.vocab_path)

    logger.info(f"Tokenizer object was saved to {descriptor.vocab_path}.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=DatasetDescriptor.LOOKUP.keys(), default="flickr30k")
    parser.add_argument("--min-freq", type=int, default=8)

    args = parser.parse_args()

    main(args)