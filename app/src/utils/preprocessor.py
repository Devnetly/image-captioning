import re
import string
from typing import Any,Callable


class TextPreprocessor(Callable):

    def __init__(self) -> None:
        
        spetial_chars = string.punctuation
        escaped_chars = [re.escape(c) for c in spetial_chars]
        self.spetial_chars_regex = re.compile(f"({'|'.join(escaped_chars)})")

    def __call__(self, doc : str) -> str:
        
        doc = doc.replace('"','')
        doc = doc.lower()
        doc = doc.strip()
        
        doc = re.sub(r"\d+", "", doc)
        doc = re.sub(self.spetial_chars_regex," ",doc)
        doc = re.sub(" +"," ", doc)

        return doc
