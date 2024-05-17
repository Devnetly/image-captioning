class Vocab:
    
    def __init__(self,
        sos_token : str = '<sos>',
        eos_token : str = '<eos>',
        unk_token : str = '<unk>',
        pad_token : str = '<pad>',
        min_freq : int = 8
    ) -> None:
        
        self.str_2_idx = {}
        self.idx_2_str = {}
        self.freqs = {}
        self.idx = 0
        
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.min_freq = 0
        
        self.add_word(self.pad_token)
        self.add_word(self.sos_token)
        self.add_word(self.eos_token)
        self.add_word(self.unk_token)

        self.min_freq = min_freq

        self.sos_idx = self.str_2_idx[sos_token]
        self.eos_idx = self.str_2_idx[eos_token]
        self.unk_idx = self.str_2_idx[unk_token]
        self.pad_idx = self.str_2_idx[pad_token]
                
    def add_word(self, word : str):
        
        if word not in self.freqs:
            self.freqs[word] = 1
        else:
            self.freqs[word] += 1

        if word not in self.str_2_idx and self.freqs[word] >= self.min_freq:
            self.str_2_idx[word] = self.idx
            self.idx_2_str[self.idx] = word
            self.idx += 1
    
    def __getitem__(self, word : str) -> int:
        return self.str_2_idx.get(word, self.unk_idx)

    def __len__(self):
        return len(self.str_2_idx)