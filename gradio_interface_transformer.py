import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import gradio as gr
import re

# ---- Learnable positional embedding ----

class Learnable_pos_emb(nn.Module):

    def __init__(self, max_seq_len, d_model, device):

        super().__init__()
        self.device = device

        self.pos_emb = nn.Embedding(max_seq_len, d_model, device=device)

    def forward(self, x):

        T = x.size(1)  # sequence length
        pos = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)

        return self.pos_emb(pos)  # (1, T, d_model)
    


# ---- Transformer Seq2Seq Model ----

class TransformerSeq2Seq_model(nn.Module):

    def __init__(
        self, hindi_roman_vocab2idx, hindi_devanagari_vocab2idx, src_vocab_size, tgt_vocab_size, max_seq_len,
        num_enc_layer=6, num_dec_layer=6, d_model=256,
        n_heads=4, ffn_dim=1024, activ_func='gelu', dropout_p=0.3,
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.nhead = n_heads
        self.num_encoder_layers = num_enc_layer
        self.num_decoder_layers = num_dec_layer
        self.dim_feedforward = ffn_dim
        self.dropout = dropout_p
        self.activation = activ_func
        self.hindi_roman_vocab2idx = hindi_roman_vocab2idx
        self.hindi_devanagari_vocab2idx = hindi_devanagari_vocab2idx


        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, device=device)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, device=device)
        self.pos_embedding = Learnable_pos_emb(max_seq_len, d_model, device=device)
        self.emb_norm = nn.LayerNorm(d_model, device=device)

        # Transformer backbone
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_enc_layer,
            num_decoder_layers=num_dec_layer,
            dim_feedforward=ffn_dim,
            dropout=dropout_p,
            activation=activ_func,
            batch_first=True,
            norm_first=False,
            device=device,
        )

        # Final vocab projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, device=device)

    # ---- Internal mask functions ----
    def _generate_square_subsequent_mask(self, sz):

        mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1).bool()

        return mask

    def _create_padding_mask(self, seq, pad_idx):

        return (seq == pad_idx)  # shape (B, T)

    # ---- Forward training ----
    def forward(self, src, tgt):

        # embeddings + positional
        src_emb = self.emb_norm(self.src_embedding(src) + self.pos_embedding(src))
        tgt_emb = self.emb_norm(self.tgt_embedding(tgt) + self.pos_embedding(tgt))

        # masks
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))
        src_padding_mask = self._create_padding_mask(src, self.hindi_roman_vocab2idx['<pad>'])
        tgt_padding_mask = self._create_padding_mask(tgt, self.hindi_devanagari_vocab2idx['<pad>'])

        # forward through transformer
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        # project to vocab
        return self.fc_out(out)  # (B, T, vocab_size)

    # ---- Greedy decoding ----
    def greedy_decode(self, src, max_len, bos_idx, eos_idx, pad_idx):
        self.eval()
        B = src.size(0)
    
        src_emb = self.emb_norm(self.src_embedding(src) + self.pos_embedding(src))
        src_padding_mask = self._create_padding_mask(src, pad_idx)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
    
        ys = torch.ones(B, 1, device=self.device, dtype=torch.long) * bos_idx
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
    
        for i in range(max_len):
            tgt_emb = self.emb_norm(self.tgt_embedding(ys) + self.pos_embedding(ys))
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1))
    
            out = self.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask
            )
    
            logits = self.fc_out(out[:, -1, :])
            next_tok = torch.argmax(logits, dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_tok], dim=1)
    
            # update finished
            finished = finished | (next_tok.squeeze(1) == eos_idx)
            if finished.all():
                break
    
        return ys[:, 1:]  # remove initial BOS

    # ---- Beam search decoding ---
    def beam_search_decode(self, src, max_len, bos_idx, eos_idx, pad_idx, beam_width=3):
        self.eval()
        B = src.size(0)
    
        src_emb = self.emb_norm(self.src_embedding(src) + self.pos_embedding(src))
        src_padding_mask = self._create_padding_mask(src, pad_idx)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
    
        sequences = [[([bos_idx], 0.0)] for _ in range(B)]
        finished_batches = [False] * B  # track which batch is done
    
        for _ in range(max_len):
            if all(finished_batches):
                break  # stop early if all batches finished
    
            for b_idx in range(B):
                if finished_batches[b_idx]:
                    continue  # skip finished batch
    
                candidates = sequences[b_idx]
                temp_candidates = []
    
                for seq, score in candidates:
                    if seq[-1] == eos_idx:
                        temp_candidates.append((seq, score))
                        continue
    
                    ys = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
                    tgt_emb = self.emb_norm(self.tgt_embedding(ys) + self.pos_embedding(ys))
                    tgt_mask = self._generate_square_subsequent_mask(ys.size(1))
    
                    out = self.transformer.decoder(
                        tgt=tgt_emb,
                        memory=memory[b_idx:b_idx+1],
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=src_padding_mask[b_idx:b_idx+1]
                    )
    
                    logits = self.fc_out(out[:, -1, :])
                    probs = F.log_softmax(logits, dim=-1)
                    topk_probs, topk_idx = probs.topk(beam_width)
    
                    for k in range(beam_width):
                        temp_candidates.append((seq + [topk_idx[0, k].item()], score + topk_probs[0, k].item()))
    
                # keep top-k
                sequences[b_idx] = sorted(temp_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
                # check if top sequence ended with eos
                if all(seq[-1] == eos_idx for seq, _ in sequences[b_idx]):
                    finished_batches[b_idx] = True
    
        # return best sequence (keep eos)
        best_seqs = []
        for b_idx in range(B):
            best_seq = max(sequences[b_idx], key=lambda x: x[1])[0]
            best_seqs.append(best_seq[1:])  # remove only BOS
    
        # return torch.tensor(best_seqs, device=self.device)
        max_len_seq = max(len(seq) for seq in best_seqs)
        padded_seqs = []

        for seq in best_seqs:
            seq = seq + [pad_idx] * (max_len_seq - len(seq))  # pad to max length
            padded_seqs.append(seq)

        return torch.tensor(padded_seqs, device=self.device)


# ----- Load your saved model and data -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "Transformer_model_full_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract vocabularies and config
hindi_roman_vocab2idx = checkpoint['src_vocab2idx']
hindi_devanagari_vocab2idx = checkpoint['tgt_vocab2idx']
hindi_roman_idx2vocab = checkpoint['src_idx2vocab']
hindi_devanagari_idx2vocab = checkpoint['tgt_idx2vocab']

config = checkpoint['config']
# device = checkpoint['device']
max_seq_len = checkpoint['max_common_seq_len']

# Rebuild the model
model = TransformerSeq2Seq_model(
    hindi_roman_vocab2idx=hindi_roman_vocab2idx,
    hindi_devanagari_vocab2idx=hindi_devanagari_vocab2idx,
    src_vocab_size=len(hindi_roman_vocab2idx),
    tgt_vocab_size=len(hindi_devanagari_vocab2idx),
    max_seq_len=max_seq_len,
    num_enc_layer=config['num_enc_layers'],
    num_dec_layer=config['num_dec_layers'],
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    ffn_dim=config['ffn_dim'],
    activ_func=config['activation'],
    dropout_p=config['dropout'],
    device=device
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


# Get common special token indices (fall back to reasonable defaults)
SRC_PAD = hindi_roman_vocab2idx["<pad>"]

TGT_BOS = hindi_devanagari_vocab2idx["<bos>"]
TGT_EOS = hindi_devanagari_vocab2idx["<eos>"]
TGT_PAD = hindi_devanagari_vocab2idx["<pad>"]

def tokenize_and_pad(src_text, vocab2idx, max_len, pad_idx):
    """
    Tokenize at character-level, map to indices, then pad/truncate to max_len.
    Returns a 1-D list of length == max_len.
    """
    tokens = list(src_text.strip())  # char-level tokens
    idxs = [vocab2idx[ch] for ch in tokens]

    # Truncate if longer than max_len
    if len(idxs) > max_len:
        idxs = idxs[:max_len]
    # Pad on the right
    pad_count = max_len - len(idxs)
    if pad_count > 0:
        idxs = idxs + [pad_idx] * pad_count

    return idxs

def detokenize(indices, idx2vocab, skip_tokens=("<pad>", "<bos>", "<eos>")):
    """
    Convert indices to tokens and join. Skip BOS/EOS/PAD tokens.
    """
    out_tokens = []

    for ind in indices:
        tok = idx2vocab.get(int(ind), "")
        if tok in skip_tokens:
            continue
        out_tokens.append(tok)

    return "".join(out_tokens)

import torch.nn.functional as F  # <-- add this

def transliterate_text_padded(text, mode="greedy"):
    if not text or not text.strip():
        return ""

    src_text = text.strip().lower()
    src_idx = tokenize_and_pad(src_text, hindi_roman_vocab2idx, max_seq_len, SRC_PAD)
    src_tensor = torch.tensor([src_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        if mode == "greedy":
            out_idxs = model.greedy_decode(src_tensor, max_len=max_seq_len, bos_idx=TGT_BOS, eos_idx=TGT_EOS, pad_idx=SRC_PAD)
        else:
            out_idxs = model.beam_search_decode(src_tensor, max_len=max_seq_len, bos_idx=TGT_BOS, eos_idx=TGT_EOS, pad_idx=SRC_PAD)

    out_seq = out_idxs[0].cpu().numpy()
    result = detokenize(out_seq, hindi_devanagari_idx2vocab)
    return result.strip()

with gr.Blocks(title="Hindi Transliteration Transformer (padded)") as demo:
    gr.Markdown("### Roman → Devanagari Transliteration (Transformer Model)")
    gr.Markdown("Uses model max_seq_len with padding/truncation.")

    mode = gr.Radio(["greedy", "beam"], value="greedy", label="Decoding Mode")
    input_text = gr.Textbox(label="Enter Roman Hindi text", placeholder="e.g. namaste or aap kaise ho")
    output_text = gr.Textbox(label="Devanagari Output")
    btn = gr.Button("Transliterate")
    btn.click(fn=lambda t, m: transliterate_text_padded(t, mode=m),
              inputs=[input_text, mode], outputs=output_text)

demo.launch()