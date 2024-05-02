import timm
import torch
from torch import nn
from timm.models.layers import trunc_normal_

class Encoder(nn.Module):
        
    def __init__(self, model_name, out_dim : int, pretrained : bool = True):
        super().__init__()
        
        self.out_dim = out_dim

        self.model = timm.create_model(model_name, num_classes=0, 
                                           global_pool='', pretrained=pretrained)
        self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):
        features = self.model(x)
        return self.bottleneck(features[:, 1:])
    
class PosEmbeddings(nn.Module):
    
    def __init__(self, max_len : int,dim : int):
        super().__init__()
        
        self.max_len = max_len
        self.dim = dim
        
        self.weight = nn.Parameter(torch.randn(1, max_len, dim) * .02)
        
        self.init_weights()
        
    def forward(self, x):
        return x + self.weight
    
    def init_weights(self):
        trunc_normal_(self.weight, std=.02)

class Mask(nn.Module):
    
    def __init__(self, 
        device : torch.device,
        pad_idx : int
    ):
        super().__init__()
        
        self.device = device
        self.pad_idx = pad_idx
        
    def forward(self, target):
        
        target_len = target.shape[1]
        
        mask = torch.ones(size=(target_len, target_len), device=self.device)
        mask = torch.triu(mask)
        mask = mask == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                
        target_padding_mask = (target == self.pad_idx)
        
        return mask, target_padding_mask
    

class Decoder(nn.Module):
    
    def __init__(self,
        vocab_size : int, 
        encoder_length : int, 
        dim : int, 
        max_len : int,
        num_heads : int, 
        num_layers : int,
        device : torch.device,
        pad_idx : int
    ):
        super().__init__()
        
        self.mask = Mask(device,pad_idx)
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=dim)
        
        self.decoder_pos_embed = PosEmbeddings(max_len-1, dim)
        self.decoder_pos_drop = nn.Dropout(p=0.05)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim, vocab_size)
        
        self.encoder_pos_embed = PosEmbeddings(encoder_length, dim)
        self.encoder_pos_drop = nn.Dropout(p=0.05)
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if not 'encoder_pos_embed' in name and 'decoder_pos_embed' not in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
    def forward(self, encoder_out, target):
                
        target_mask, target_padding_mask = self.mask(target)
        
        target_embedding = self.embedding(target)
        target_embedding = self.decoder_pos_embed(target_embedding)
        target_embedding = self.decoder_pos_drop(target_embedding)
        
        encoder_out = self.encoder_pos_embed(encoder_out)
        encoder_out = self.encoder_pos_drop(encoder_out)
        
        encoder_out = encoder_out.transpose(0, 1)
        target_embedding = target_embedding.transpose(0, 1)
        
        preds = self.decoder(
            memory=encoder_out,
            tgt=target_embedding,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask.float()
        )
        
        preds = preds.transpose(0, 1)
        
        outputs = self.output(preds)
        
        return outputs
    
class Transformer(nn.Module):
    
    def __init__(self,
        feature_extractor_name : str,
        features_dim : int,
        vocab_size : int,
        encoder_length : int,
        num_heads : int,
        num_layers : int,
        max_len : int,
        device : torch.device,
        pad_idx : int
    ):
        super().__init__()
        
        self.feature_extractor_name = feature_extractor_name
        self.features_dim = features_dim
        self.vocab_size = vocab_size
        self.encoder_length = encoder_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.device = device
        self.pad_idx = pad_idx
        
        self.encoder = Encoder(
            model_name=self.feature_extractor_name,
            out_dim=features_dim,
            pretrained=True
        )
        
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            encoder_length=self.encoder_length,
            dim=self.features_dim,
            max_len=self.max_len,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            device=self.device,
            pad_idx=self.pad_idx
        )
    
    def forward(self, x : tuple[torch.Tensor,torch.Tensor]):
        image, target = x
        encoder_out = self.encoder(image)
        preds = self.decoder(encoder_out, target)
        return preds