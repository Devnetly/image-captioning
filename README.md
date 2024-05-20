# Image Captioning with Transformer-Based Architecture

## Overview

This project is part of a university course on Natural Language Processing (NLP). The objective is to develop a model that can generate captions for images using a transformer-based architecture. We have utilized a Data-efficient Image Transformer (DeiT) for the encoder and a standard transformer decoder. The Flickr30k dataset was used for training, with additional text preprocessing and image resizing and augmentation techniques applied.

## Members

- Abdelnour FELLAH: [ab.fellah@esi-sba.dz](mailto:ab.fellah@esi-sba.dz)
- Abderrahmane Benounene: [a.benounene@esi-sba.dz](mailto:a.benounene@esi-sba.dz)
- Adel Abdelkader Mokadem: [aa.mokadem@esi-sba.dz](mailto:aa.mokadem@esi-sba.dz)
- Meriem Mekki: [me.mekki@esi-sba.dz](mailto:me.mekki@esi-sba.dz)
- Yacine Lazreg Benyamina: [yl.benyamina@esi-sba.dz](mailto:yl.benyamina@esi-sba.dz)

## Setup

To set up the project, please follow these steps:

```sh
git clone https://github.com/your-repo/image-captioning.git
cd image-captioning
conda create -n automatic-image-captioning
conda activate automatic-image-captioning
pip install -r requirements.txt
```

## Training

To train the model,follow the steps below : 

```sh
cd src/training
python train.py [-h] --dataset {flickr30k} [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--weight-decay WEIGHT_DECAY] [--epochs EPOCHS] [--num-workers NUM_WORKERS] [--prefetch-factor PREFETCH_FACTOR] --weights-folder WEIGHTS_FOLDER --histories-folder HISTORIES_FOLDER
```

## Inference

To generate caption for a set of images in folder,follow these steps : 

```sh
cd src/inference
inference.py [-h] [--dataset {flickr30k}] [--model {transformer}] --checkpoint CHECKPOINT [--source SOURCE] --destination DESTINATION
```

## Run the associated app

To run the app associated with the project : 

```sh
cd app
streamlit run main.py
```