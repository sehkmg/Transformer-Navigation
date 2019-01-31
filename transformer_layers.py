import torch
import torch.nn as nn
from transformer_modules import clones, LayerNorm, SublayerConnection

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        src_embed = self.src_embed(src)
        print('')
        print('2. Encoding part')
        print('')
        print('2-1. Source batch embedding + positional encoding (src_embed)')
        print(src_embed) #Tracking
        print('')
        return self.encoder(src_embed, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt_embed = self.tgt_embed(tgt)
        print('3. Decoding part')
        print('')
        print('3-1. Target batch embedding + positional encoding (tgt_embed)')
        print(tgt_embed) #Tracking
        print('')
        print('3-2. Output from encoder (memory)')
        print(memory) #Tracking
        print('')
        return self.decoder(tgt_embed, memory, src_mask, tgt_mask)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
            x = self.norm(x)
            print('2-3. X = X + FF(X)')
            print(x) #Tracking
            print('')
        return x

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        print('2-2. MultiHeadedAttention(src_embed (query), src_embed (key), src_embed (value), src_mask)')
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        print('Merge, take a linear map and add to the src_embed')
        print(x) #Tracking
        print('')
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            x = self.norm(x)
            print('3-5. X = X + FF(X)')
            print(x) #Tracking
            print('')
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        print('3-3. MultiHeadedAttention(tgt_embed (query), tgt_embed (key), tgt_embed (value), tgt_mask)')
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        print('Merge, take a linear map and add to the tgt_embed (tgt_embed_attn)')
        print(x) #Tracking
        print('')
        print('3-4. MultiHeadedAttention(tgt_embed_attn (query), memory (key), memory (value), src_mask)')
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        print('Merge, take a linear map and add to the tgt_embed_attn')
        print(x) #Tracking
        print('')
        return self.sublayer[2](x, self.feed_forward)
