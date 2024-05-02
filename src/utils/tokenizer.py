from .vocab import Vocab
from nltk import word_tokenize
from typing import Any, Optional,Callable
from tqdm import tqdm
from joblib import load,dump
from torch import tensor

class Tokenizer(Callable):

    def __init__(self,
        sos_token : str = '<sos>',
        eos_token : str = '<eos>',
        unk_token : str = '<unk>',
        pad_token : str = '<pad>',
        min_freq : int = 8,
        preprocessor : Optional[Callable] = None
    ) -> None:
        
        self.vocab = Vocab(sos_token,eos_token,unk_token,pad_token,min_freq)
        self.preprocessor = preprocessor

    def fit(self, docs : list[str]):
        
        for doc in tqdm(docs):

            if self.preprocessor is not None:
                doc = self.preprocessor(doc)

            for word in word_tokenize(doc):
                self.vocab.add_word(word)

        return self
    
    def __call__(self, doc : str) -> Any:
        
        if doc is not None:
            doc = self.preprocessor(doc)

        tokens = word_tokenize(doc)
        tokens = [self.vocab.sos_idx] + [self.vocab[word] for word in tokens] + [self.vocab.eos_idx]
        tokens = tensor(tokens)

        return tokens
    
    def save(self, f : str):
        return dump(self, f)

    def load(f : str):
        return load(f)
