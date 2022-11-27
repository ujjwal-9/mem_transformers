import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        # token_ids_batch has shape (B, S/T), where B - batch size, S/T max src/trg token-sequence length
        # Final shape will be (B, S/T, D) where D is the model dimension, every token id has associated vector
        embeddings = self.embeddings_table(token_ids_batch)

        # (stated in the paper) multiply the embedding weights by the square root of model dimension
        # Page 5, Chapter 3.4 "Embeddings and Softmax"
        return embeddings * math.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work so feel free to change it!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        # Register buffer because we want to save the positional encodings table inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where S/T max src/trg token-sequence length, D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D) when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings)


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class UpdateAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, **kwargs):
        super(UpdateAttn, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.ff = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_emb = PositionalEncoding(d_model)

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output += self.ff(output)
        output = self.layer_norm(output)
        output += self.pos_emb(output)
        return output

class OutputAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, **kwargs):
        super(OutputAttn, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        return output

class CrossAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, **kwargs):
        super(OutputAttn, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        return output


class MemTransformerLMEncoder(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1):
        
        super(MemTransformerLMEncoder, self).__init__()
        
        self.n_token = n_token
        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_mem = 16

        self.word_emb = Embedding(n_token, d_model)

        self.n_layer = n_layer
        
        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Linear(n_head * d_head, d_model, bias=False)
        self.update_attention_mha = UpdateAttn(n_head, d_model, d_head, dropout)
        self.output_attention_mha = OutputAttn(n_head, d_model, d_head, dropout)
        self.mem = self._initiate_memory()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
    
    def _initiate_memory(self, bsz):
        param = next(self.parameters())
        mem = torch.zeros(bsz, self.d_mem, self.d_model, 
                          dtype=param.dtype, device=param.device)
        return mem

    def update_memory(self, mem, hids):
        mem = self.update_attention(mem, hids)
        mem = torch.cat([mem, hids], dim=1)
        mem = mem[:, -self.d_mem:, :]
        return mem

    def update_attention(self, mem, hids):
        # [bsz x d_mem x d_model] + [bsz x n_head x d_head] -> [bsz x d_mem x d_model]
        new_mem = self.update_attention_mha(hids, mems=mem)
        return new_mem

    def output_attention(self, mem, hids):
        output = self.output_attention_mha(mem, mems=hids)
        return output
        

    def _forward(self, data, *mems):
        bsz = data.sz(0)
        if len(mems) == 0:
            mems = self._initiate_memory(bsz)

        hids = self.word_emb(data)
        hids = self.pos_emb(hids)
        hids = self.dec_attn(hids)
        mems = self.update_memory(mems, hids)
        attn_output = self.output_attention(mems, hids)
        out_attn = self.layer_norm(hids + attn_output)
        output = self.layer_norm2(self.ff(out_attn) + out_attn)
        return output, mems

    def forward(self, data, *mems):
        output, mems = self._forward(data, *mems)
        return output, mems


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1):
        
        super(MemTransformerLMEncoder, self).__init__()
        
        self.n_token = n_token
        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_mem = 16

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, 
                                          div_val=div_val)

        self.n_layer = n_layer
        
        self.encoder = MemTransformerLMEncoder(n_token, n_layer, n_head, d_model, d_head, d_inner)
        
        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Linear(n_head * d_head, d_model, bias=False)
        self.update_attention_mha = UpdateAttn(n_head, d_model, d_head, dropout)
        self.output_attention_mha = OutputAttn(n_head, d_model, d_head, dropout)
        
        self.cross_attn_mha = CrossAttn(n_head, d_model, d_head, dropout)

        self._initiate_memory()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
    
    def _initiate_memory(self, bsz):
        param = next(self.parameters())
        mem = torch.zeros(bsz, self.d_mem, self.d_model, 
                          dtype=param.dtype, device=param.device)
        return mem

    def update_memory(self, mem, hids):
        mem = self.update_attention(mem, hids)
        mem = torch.cat([mem, hids], dim=1)
        mem = mem[:, -self.d_mem:, :]
        return mem

    def update_attention(self, mem, hids):
        # [bsz x d_mem x d_model] + [bsz x n_head x d_head] -> [bsz x d_mem x d_model]
        new_mem = self.update_attention_mha(hids, mems=mem)
        return new_mem

    def output_attention(self, mem, hids):
        output = self.output_attention_mha(mem, mems=hids)
        return output
        
    def forward_encoder(self, data, mems_enc):
        output, mems_enc = self.encoder(data, mems_enc)
        return output, mems_enc

    def forward(self, data, target, mems_enc, mems_dec):
        bsz = data.sz(0)
        if len(mems_dec) == 0:
            mems = self._initiate_memory(bsz)
        
        output_enc, mems_enc = self.forward_encoder(data, mems_enc)
        
        hids = self.word_emb(target)
        hids = self.pos_emb(hids)
        hids = self.dec_attn(hids)
        mems = self.update_memory(mems_dec, hids)
        attn_output = self.output_attention(mems_dec, hids)
        output_dec = self.layer_norm1(hids + attn_output)
        output_attn = self.cross_attn_mha(output_enc, mems=output_dec)
        output_attn += self.layer_norm2(output_dec + output_attn)
        output_dec = self.layer_norm3(self.ff(output_attn) + output_attn)
        return output_enc, output_dec, mems_enc, mems_dec