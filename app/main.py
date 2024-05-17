import streamlit as st
import numpy as np
import torch
import os
from PIL import Image
from torchvision.transforms import Compose,Resize,ToTensor
from src.utils import seed_everything,Tokenizer,read_json,ImageCaptionGenerator
from src.utils.model import Transformer
from time import sleep

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 3rem;
            padding-right: 50%;
        }
    </style>
""", unsafe_allow_html=True)

class GLOBAL:

    IMG_SIZE = 384
    SEED = 8

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(ROOT_DIR, 'model')

    CONFIG_FILE = os.path.join(MODEL_DIR, 'config.json')
    WEIGHTS_FILE = os.path.join(MODEL_DIR, 'weights.pt')
    TOKENIZER_FILE = os.path.join(MODEL_DIR, 'vocab.pkl')

print(GLOBAL.TOKENIZER_FILE)

seed_everything(GLOBAL.SEED)

@st.cache_resource
def load(config):

    preprocessor = Compose([
        Resize((GLOBAL.IMG_SIZE,GLOBAL.IMG_SIZE)),
        ToTensor()
    ])

    tokenizer : Tokenizer = Tokenizer.load(GLOBAL.TOKENIZER_FILE)

    model = Transformer(
        **config,
        vocab_size=len(tokenizer.vocab),
        device=GLOBAL.DEVICE,
        pad_idx=tokenizer.vocab.pad_idx
    ).to(GLOBAL.DEVICE)

    model.load_state_dict(torch.load(GLOBAL.WEIGHTS_FILE))

    return preprocessor,tokenizer,model


uploaded_file = None
config = read_json(GLOBAL.CONFIG_FILE)
preprocessor,tokenizer,model = load(config)

with st.sidebar:
    st.markdown("### Upload your image from here")
    uploaded_file = st.file_uploader(label="",type=["png","jpg","jpeg"])

st.markdown("# Automatic Caption Generator")

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    np_image = np.array(image)

    st.image(np_image)
    predict_button = st.button("Generate caption")

    if predict_button:

        generator = ImageCaptionGenerator(
            model=model, 
            tokenizer=tokenizer, 
            img=image,
            max_len=int(config['max_len']), 
            device=GLOBAL.DEVICE, 
            preprocessor=preprocessor
        )

        t = st.empty()
        caption = []

        for word in generator:
            caption.append(word)
            caption[0] = caption[0].title()
            t.markdown("%s" % ' '.join(caption))
            sleep(0.05)
        
        t.markdown("%s" % ' '.join(caption) + '.')
else:
    st.info("No image is uploaded.")