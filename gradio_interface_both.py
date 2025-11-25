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


class Encoder(nn.Module):

  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, device, bi_dir = False):

    super(Encoder, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.device = device
    self.bi_dir = bi_dir

    self.dropout = nn.Dropout(p)
    self.embedding = nn.Embedding(input_size, embedding_size)
    # self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, batch_first = True)
    self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=self.bi_dir)

    if self.bi_dir:
      self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
      self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

  def forward(self, x):

    x = x.to(self.device)

    if self.bi_dir:

      embedding = self.embedding(x)
      embedding = self.dropout(embedding)

      outputs, (hidden_state, cell_state) = self.lstm(embedding)

      # Reshape to [num_layers, num_directions, batch_size, hidden_size]
      h_n = hidden_state.view(self.num_layers, 2, x.size(0), self.hidden_size)
      c_n = cell_state.view(self.num_layers, 2, x.size(0), self.hidden_size)

      # Concatenate forward and backward hidden states
      h_n = torch.cat((h_n[:, 0, :, :], h_n[:, 1, :, :]), dim=2)  # [num_layers, batch_size, hidden_size*2]
      c_n = torch.cat((c_n[:, 0, :, :], c_n[:, 1, :, :]), dim=2)

      # Project to original hidden size
      h_n = self.fc_hidden(h_n)  # [num_layers, batch_size, hidden_size]
      c_n = self.fc_cell(c_n)

      return h_n, c_n

    embedding = self.embedding(x)

    outputs, (hidden_state, cell_state) = self.lstm(embedding)

    return hidden_state, cell_state
  
  
class Decoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p, device):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        # self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, batch_first = True)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):

        x = x.to(self.device)
        # embedding = self.dropout(self.embedding(x))
        embedding = self.embedding(x)

        outputs, (hidden_state_n, cell_state_n) = self.lstm(embedding, (hidden_state, cell_state))

        predictions = self.fc(outputs)

        return predictions, hidden_state_n, cell_state_n

    def greedy_batch(self, ht, ct, target_len, bos_idx, eos_idx, pad_idx):

        B = ht.size(1)
        device = ht.device

        # Pre-allocate output tensor filled with <pad>
        outputs = torch.full((B, target_len), pad_idx, dtype=torch.long, device=device)

        # Track which sequences have finished (generated <eos>)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)

        # Initialize first input token (<bos>) for all sequences
        input_token = torch.full((B,), bos_idx, dtype=torch.long, device=device)

        for t in range(target_len):

            # Forward one step through the decoder
            logits, ht, ct = self.forward(input_token.unsqueeze(1), ht, ct)
            logits = logits.squeeze(1)               # [B, vocab_size]

            # Greedy selection
            next_word = logits.argmax(dim=-1)        # [B]

            # Update only unfinished sequences
            not_finished = ~finished
            if not_finished.any():
                outputs[not_finished, t] = next_word[not_finished]
                lengths[not_finished] += 1
                finished = finished | (next_word == eos_idx)

            # Stop if all sequences finished
            if finished.all():
                break

            # Next input token
            input_token = next_word

        return outputs, lengths.tolist()

    def beam_search_batch(self, ht, ct, target_len, bos_idx, eos_idx, pad_idx, beam_width=3):

        B = ht.size(1)   # batch size
        device = ht.device

        # Each batch will store its beam results separately
        all_batch_results = []

        for b in range(B):
            # Initialize beam with <bos>
            beams = [([bos_idx], 0.0, ht[:, b:b+1, :], ct[:, b:b+1, :])]  # (sequence, log_prob, hidden, cell)

            for _ in range(target_len):
                new_beams = []
                for seq, log_prob, h, c in beams:
                    if seq[-1] == eos_idx:
                        new_beams.append((seq, log_prob, h, c))
                        continue

                    inp = torch.tensor([seq[-1]], dtype=torch.long, device=device).unsqueeze(0)  # [1,1]
                    logits, h_new, c_new = self.forward(inp, h, c)   # [1,1,vocab_size]
                    logits = logits.squeeze(1)                      # [1, vocab_size]
                    log_probs = torch.log_softmax(logits, dim=-1)   # log-probabilities

                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                    for i in range(beam_width):
                        next_token = topk_indices[0, i].item()
                        next_score = log_prob + topk_log_probs[0, i].item()
                        new_seq = seq + [next_token]
                        new_beams.append((new_seq, next_score, h_new, c_new))

                # Keep only top-k beams
                new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                beams = new_beams

            # Pick best sequence (highest score)
            best_seq = beams[0][0]

            # Pad to target_len
            if len(best_seq) < target_len:
                best_seq = best_seq + [pad_idx] * (target_len - len(best_seq))

            all_batch_results.append(best_seq[:target_len])

        # Convert to tensor [B, target_len]
        return torch.tensor(all_batch_results, dtype=torch.long, device=device)


class Seq2Seq_Character(nn.Module):

  def __init__(self, encoder, decoder, device):

    super(Seq2Seq_Character, self).__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.device = device

  def forward(self, source, target, tfr=0.5):

    source = source.to(self.device)
    target = target.to(self.device)

    batch_size = source.shape[0]
    target_len = target.shape[1]
    target_vocab_size = self.decoder.fc.out_features


    outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

    ht, ct = self.encoder(source)
    input_token = target[:, 0]  # <bos>

    for t in range(1, target_len):
        output, ht, ct = self.decoder(input_token.unsqueeze(1), ht, ct)
        output = output.squeeze(1)
        outputs[:, t, :] = output
        teacher_force = random.random() < tfr
        top1 = output.argmax(1)
        input_token = target[:, t] if teacher_force else top1

    return outputs
  
  
# Simple validator for words containing only a-z (lowercase)
def is_valid_word(word: str) -> bool:
    """
    Returns True if `word` is non-empty and contains only letters a-z (ASCII lowercase).
    """
    if not isinstance(word, str):
        return False
    w = word.strip().lower()
    if w == "":
        return False
    # match one or more lowercase ASCII letters only
    return bool(re.fullmatch(r'[a-z]+', w))



# ----- Load your saved model and data -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load data ----- (same for both models)
saved_data = torch.load("./seq2seq_data_bigreedy.pth", map_location=device, weights_only=False)
hindi_roman_vocab2idx = saved_data['input_char2idx']
hindi_roman_idx2vocab = saved_data['input_idx2char']
hindi_devanagari_vocab2idx = saved_data['output_char2idx']
hindi_devanagari_idx2vocab = saved_data['output_idx2char']
hindi_roman_max_len = saved_data['input_max_len']
hindi_devanagari_max_len = saved_data['output_max_len']

checkpoint_bilstm = torch.load("./seq2seq_model_bigreedy.pth", map_location=device)
checkpoint_lstm   = torch.load("./seq2seq_model_greedy.pth", map_location=device)  # LSTM version
config_bilstm = checkpoint_bilstm['config']
config_lstm   = checkpoint_lstm['config']



# ----- BILSTM -----
encoder_bilstm = Encoder(
    input_size=config_bilstm['encoder_input_size'],
    embedding_size=config_bilstm['encoder_embedding_size'],
    hidden_size=config_bilstm['hidden_size'],
    num_layers=config_bilstm['encoder_num_layers'],
    p=config_bilstm['dropout'],
    device=device,
    bi_dir=config_bilstm['bidirectional']
)
decoder_bilstm = Decoder(
    input_size=config_bilstm['decoder_input_size'],
    embedding_size=config_bilstm['decoder_embedding_size'],
    hidden_size=config_bilstm['hidden_size'],
    output_size=config_bilstm['decoder_input_size'],
    num_layers=config_bilstm['decoder_num_layers'],
    p=config_bilstm['dropout'],
    device=device
)
encoder_bilstm.load_state_dict(checkpoint_bilstm['encoder_state_dict'])
decoder_bilstm.load_state_dict(checkpoint_bilstm['decoder_state_dict'])
model_bilstm = Seq2Seq_Character(encoder_bilstm, decoder_bilstm, device)
model_bilstm.to(device).eval()

# ----- LSTM -----
encoder_lstm = Encoder(
    input_size=config_lstm['encoder_input_size'],
    embedding_size=config_lstm['encoder_embedding_size'],
    hidden_size=config_lstm['hidden_size'],
    num_layers=config_lstm['encoder_num_layers'],
    p=config_lstm['dropout'],
    device=device,
    bi_dir=False  # LSTM is not bidirectional
)
decoder_lstm = Decoder(
    input_size=config_lstm['decoder_input_size'],
    embedding_size=config_lstm['decoder_embedding_size'],
    hidden_size=config_lstm['hidden_size'],
    output_size=config_lstm['decoder_input_size'],
    num_layers=config_lstm['decoder_num_layers'],
    p=config_lstm['dropout'],
    device=device
)
encoder_lstm.load_state_dict(checkpoint_lstm['encoder_state_dict'])
decoder_lstm.load_state_dict(checkpoint_lstm['decoder_state_dict'])
model_lstm = Seq2Seq_Character(encoder_lstm, decoder_lstm, device)
model_lstm.to(device).eval()




# ----- Decoding function -----
bos_idx = hindi_devanagari_vocab2idx["<bos>"]
eos_idx = hindi_devanagari_vocab2idx["<eos>"]
pad_idx = hindi_devanagari_vocab2idx["<pad>"]

def transliterate(hindi_roman, decoder_method="greedy", beam_width=3, model_type="BILSTM"):

    # word-level check
    if not isinstance(hindi_roman, str) or hindi_roman.strip() == "":
        return "type correct input"
    hindi_roman = hindi_roman.strip().lower()
    if " " in hindi_roman or not is_valid_word(hindi_roman):
        return "type correct input"
    
    # Choose model
    if model_type == "BILSTM":
        model = model_bilstm
    elif model_type == "LSTM":
        model = model_lstm
    else:
        return "Invalid model type. Choose 'BILSTM' or 'LSTM'."

    # Tokenize input
    tokenized_r = [hindi_roman_vocab2idx.get(ch, hindi_roman_vocab2idx["<pad>"]) for ch in hindi_roman]
    tokenized_r = torch.tensor(tokenized_r, dtype=torch.long, device=device)
    
    if len(tokenized_r) < hindi_roman_max_len:
        pad_len = hindi_roman_max_len - len(tokenized_r)
        pad_tensor = torch.full((pad_len,), hindi_roman_vocab2idx["<pad>"], dtype=torch.long, device=device)
        tokenized_r = torch.cat([tokenized_r, pad_tensor])
    else:
        tokenized_r = tokenized_r[:hindi_roman_max_len]
    
    tokenized_r = tokenized_r.unsqueeze(0)

    # Encode
    with torch.no_grad():
        ht, ct = model.encoder(tokenized_r)
        max_len = hindi_devanagari_max_len

        # Decode
        if decoder_method == 'greedy':
            predicted_seq_tensor, _ = model.decoder.greedy_batch(ht, ct, target_len=max_len,
                                                                bos_idx=bos_idx, eos_idx=eos_idx, pad_idx=pad_idx)
        else:
            predicted_seq_tensor = model.decoder.beam_search_batch(ht, ct, target_len=max_len,
                                                                bos_idx=bos_idx, eos_idx=eos_idx, pad_idx=pad_idx,
                                                                beam_width=beam_width)
            
    pred_seq = predicted_seq_tensor[0]
    pred_chars = [hindi_devanagari_idx2vocab[idx.item()] for idx in pred_seq if idx.item() not in [pad_idx, eos_idx, bos_idx]]
    return "".join(pred_chars)


with gr.Blocks() as iface:
    gr.Markdown("## Hindi Roman → Devanagari Transliteration")

    # Word-level
    with gr.Row():
        with gr.Column():
            word_input = gr.Textbox(label="Word Input (Hindi Roman)")
            word_model = gr.Dropdown(["BILSTM", "LSTM"], label="Model", value="BILSTM")
            word_decoder = gr.Radio(["greedy", "beam"], label="Decoder Method", value="greedy")
            word_beam_width = gr.Slider(1, 10, value=3, step=1, label="Beam Width (only for beam search)")
            word_button = gr.Button("Transliterate Word")
        with gr.Column():
            word_output = gr.Textbox(label="Predicted Devanagari Word")

    word_button.click(
        fn=transliterate,
        inputs=[word_input, word_decoder, word_beam_width, word_model],
        outputs=[word_output]
    )

    gr.Markdown("---")

    # Sentence-level
    def transliterate_sentence_input(sentence_input, decoder_method="greedy", beam_width=3, model_type="BILSTM"):

        if not isinstance(sentence_input, str) or sentence_input.strip() == "":
            return "type correct input"
        
        # sentence-level: validate each token
        words = sentence_input.strip().split()
        for w in words:
            if not is_valid_word(w):
                return "type correct input"
            
        devanagari_words = [transliterate(word, decoder_method, beam_width, model_type) for word in words]
        return " ".join(devanagari_words)

    with gr.Row():
        with gr.Column():
            sentence_input = gr.Textbox(label="Sentence Input (Hindi Roman)")
            sentence_model = gr.Dropdown(["BILSTM", "LSTM"], label="Model", value="BILSTM")
            sentence_decoder = gr.Radio(["greedy", "beam"], label="Decoder Method", value="greedy")
            sentence_beam_width = gr.Slider(1, 10, value=3, step=1, label="Beam Width (only for beam search)")
            sentence_button = gr.Button("Transliterate Sentence")
        with gr.Column():
            sentence_output = gr.Textbox(label="Predicted Devanagari Sentence")

    sentence_button.click(
        fn=transliterate_sentence_input,
        inputs=[sentence_input, sentence_decoder, sentence_beam_width, sentence_model],
        outputs=[sentence_output]
    )

iface.launch()
