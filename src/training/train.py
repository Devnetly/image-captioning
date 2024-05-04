import sys
sys.path.append('../..')
import argparse
import torch
import time
import os
import logging
import logging.config
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.models import Transformer
from src.datasets import CaptionsDataset
from src.utils import Tokenizer,TensorTuple,read_json,seed_everything
from src.losses import Seq2SeqCrossentropy
from src.metrics import Seq2SeqAccuracy
from torchmetrics.text import Perplexity
from src.utils import DatasetDescriptor
from src.trainer import Trainer
from dataclasses import dataclass
from definitions import *
from transformers import get_linear_schedule_with_warmup
from torchvision.transforms import Compose,RandomHorizontalFlip,RandomResizedCrop,ToTensor
logging.basicConfig(level = logging.INFO, format=' %(name)s :: %(levelname)-8s :: %(message)s')

@dataclass
class Args:

    dataset : str
    features : str

    batch_size : int
    weight_decay : float
    learning_rate : float
    epochs : int

    num_workers : int
    prefetch_factor : int

    weights_folder : str
    histories_folder : str

class Collate:

    def __init__(self, maxlen : int, pad_idx : int):
        self.maxlen = maxlen
        self.pad_idx = pad_idx

    def __call__(self, batch : list[tuple[torch.Tensor,torch.Tensor]]) -> tuple[tuple[Tensor,Tensor],Tensor]:

        images = torch.cat([torch.unsqueeze(x[0], dim=0) for x in batch], dim=0)

        captions = [x[1] for x in batch]
        captions = torch.nn.utils.rnn.pad_sequence(sequences=captions, batch_first=True)

        captions_ = torch.zeros(size=(captions.size(0),self.maxlen)).type(torch.long).fill_(self.pad_idx)

        captions_[:,:captions.size(1)] = captions

        y_input = captions_[:,:-1]
        y_expected = captions_[:,1:]

        return TensorTuple((images,y_input)), y_expected

def create_datasets(
    descriptor : DatasetDescriptor, 
    tokenizer : Tokenizer,
) -> tuple[CaptionsDataset, CaptionsDataset]:
    
    train_transfroms = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop(size=(384,384)),
        ToTensor()
    ])
    
    val_transfroms = Compose([
        RandomResizedCrop(size=(384,384)),
        ToTensor()
    ])
    
    train_data = CaptionsDataset(
        root=descriptor.images_dir,
        captions_df=descriptor.train_captions,
        caption_transform=tokenizer,
        img_transform=train_transfroms
    )

    test_data = CaptionsDataset(
        root=descriptor.images_dir,
        captions_df=descriptor.test_captions,
        caption_transform=tokenizer,
        img_transform=val_transfroms
    )

    return train_data, test_data

def read_weights_folder(path : str):
    
    folders = os.listdir(path)
    folders = filter(lambda f : f.endswith('.pt'),folders)
    folders = map(lambda f : os.path.join(path, f), folders)
    folders = sorted(folders)
    num_epochs = len(folders)
    
    last_weights = None
    
    if num_epochs != 0:
        print(f"Loading weights : {folders[-1]}")
        last_weights = torch.load(folders[-1])
        
    return last_weights,num_epochs

def main(args : Args):


    ### Preparing paths
    histories_folder = os.path.join(HISTORIES_DIR,args.histories_folder)
    weights_folder = os.path.join(MODELS_DIR,args.weights_folder)
    best_weights_folder = os.path.join(MODELS_DIR,args.weights_folder,'best_weights')

    ### Check for directories existance
    if not os.path.exists(histories_folder):
        raise Exception(f"{histories_folder} not found")
    
    if not os.path.exists(weights_folder):
        raise Exception(f"{weights_folder} not found")
    
    if not os.path.exists(best_weights_folder):
        raise Exception(f"{best_weights_folder} not found")
    

    ### Setup code to be device agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Reproducibility
    seed_everything()


    ### creating a logger
    logger = logging

    ### Loading model config
    config = read_json(os.path.join(weights_folder, "config.json"))
    logger.info(f'config = {str(config)}')

    ### Creating training & test datasets
    logger.info('creating training & testing datasets')

    descriptor : DatasetDescriptor = DatasetDescriptor.get_by_name(args.dataset)

    tokenizer : Tokenizer = Tokenizer.load(descriptor.vocab_path)

    train_data, test_data = create_datasets(descriptor=descriptor,tokenizer=tokenizer)

    ### Creating train & test data loaders
    logger.info('creating training & testing dataloaders')

    vocab_size = len(tokenizer.vocab)
    pad_idx = tokenizer.vocab.pad_idx

    train_loader = DataLoader(dataset=train_data,batch_size=args.batch_size,shuffle=True,collate_fn=Collate(config["max_len"], pad_idx))
    test_loader = DataLoader(dataset=test_data,batch_size=args.batch_size,shuffle=True,collate_fn=Collate(config["max_len"], pad_idx))

    logger.info(f'vocab_size = {vocab_size}')

    ### Creating the model
    logger.info('createing model,optimizer,loss & trainer instances')

    model = Transformer(
        **config,
        vocab_size=vocab_size,
        device=device,
        pad_idx=pad_idx
    ).to(device)    

    last_weights,num_epochs = read_weights_folder(weights_folder)

    if last_weights is not None:
        model.load_state_dict(last_weights)

    ### Optimizer
    optimizer = AdamW([{'params':model.parameters(),'initial_lr':args.learning_rate}],
        lr=args.learning_rate, weight_decay=args.weight_decay
    )

    ### Loss & other metrics
    loss = Seq2SeqCrossentropy(ignore_index=pad_idx).to(device=device)
    accuracy = Seq2SeqAccuracy(num_classes=vocab_size,ignore_index=pad_idx).to(device=device)
    perplexity = Perplexity(ignore_index=pad_idx).to(device=device)

    ### Learning rate scheduling
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(0.05 * num_training_steps)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=num_epochs-1,
    )

    ### Trainer class
    trainer = Trainer() \
        .set_model(model) \
        .set_criteron(loss) \
        .set_scheduler(lr_scheduler) \
        .set_save_weights_every(1) \
        .set_weights_folder(weights_folder) \
        .add_metric("accuracy", accuracy) \
        .add_metric("perplexity", perplexity) \
        .set_device(device) \
        .set_optimizer(optimizer) \
        .set_score_metric("accuracy") \
        .set_weights_folder(weights_folder)
    
    ### Start training
    logger.info('training starts now')
    
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        epochs=args.epochs
    )

    logger.info('end of training')
    logger.info('saving results to the disk')

    ### Save the history
    t = time.time()

    trainer.history.to_df().to_csv(
        os.path.join(histories_folder,f"{t}.csv"), 
        index=False
    )

    #### Save the model
    torch.save(model.state_dict(),os.path.join(weights_folder,f"{t}.pt"))
    torch.save(trainer.best_weights,os.path.join(best_weights_folder,f"{t}_best_weights.pt"))

    logger.info('Goodbye')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, choices=DatasetDescriptor.LOOKUP.keys(), required=True)

    # training hyperparameters
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)

    # hardware related parameters
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=2)

    # model persistance parameters
    parser.add_argument('--weights-folder', type=str, required=True)
    parser.add_argument('--histories-folder', type=str, required=True)

    args = parser.parse_args()

    main(args)