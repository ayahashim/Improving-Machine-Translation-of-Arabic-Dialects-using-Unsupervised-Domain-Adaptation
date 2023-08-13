

import random

import torch
import torch.nn as nn

import torch.nn.functional as F
from transformers import ( AutoModel, AutoTokenizer)

teacher_force_ratio =0.5
MAX_LENGTH=200
model_name = "aubmindlab/bert-base-arabertv02-twitter"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)



SOS_token = tokenizer.cls_token_id
EOS_token = tokenizer.sep_token_id
PAD_token = tokenizer.pad_token_id


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))

        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        #scores shape (N,L,1)
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        #weights shape is N,1,L
        #context shape is N,1,hid_size
        return context, weights

class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, query, keys):
        scores = torch.bmm(query, keys.permute(0,2,1))



        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        #weights shape is N,1,L
        #context shape is N,1,hid_size
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, attention = "Bahdanau"):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)

        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        if attention == "Bahdanau":
          self.attention = BahdanauAttention(hidden_size)
        else:
          self.attention = DotAttention()
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, MAX_LENGTH = MAX_LENGTH):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            use_teacher_forcing = True if random.random() < teacher_force_ratio else False
            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                if use_teacher_forcing:
                    decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
                else:
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
