# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:25:24 2023

@author: ayaha
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
from arabert.preprocess import ArabertPreprocessor
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import ( AutoModel, AutoTokenizer)
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from Data_preprocessing import data_preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

import time
import math

MAX_LENGTH = 200
model_name = "aubmindlab/bert-base-arabertv02-twitter"
arabic_prep = ArabertPreprocessor(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
SOS_token = tokenizer.cls_token_id
EOS_token = tokenizer.sep_token_id
PAD_token = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_file = "./Data/data_src_train_MAG"
vocab_file = "./Data/data_all"
data_trgt_file = "./Data/data_trgt_train_MAG"
class Lang:
    def __init__(self, name):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.word2index = { self.tokenizer.pad_token : self.tokenizer.pad_token_id}
        self.word2count = {}
        self.index2word = {self.tokenizer.pad_token_id: self.tokenizer.pad_token}
        self.n_words = 1 # Count PAD token

    def addSentence(self, sentence):
        for word in self.tokenizer.tokenize(sentence, add_special_tokens= True):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.tokenizer.convert_tokens_to_ids(word)
            self.word2count[word] = 1
            self.index2word[self.tokenizer.convert_tokens_to_ids(word)] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFKC', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s= arabic_prep.preprocess(s)
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)

    s = re.sub(r"[^a-zA-Z؀-ۿ?.!,¿]+", " ", s)
    s = re.sub(r"([.!?])", r" \1", s)
 
    return s

def readLangs(lang1, lang2, reverse=False, label="src"):
    print("Reading lines...")

    # Read the file and split into lines
    if label=="vocab":
        lines = open(vocab_file, encoding='utf-8').\
        read().strip().split('\n')
    if label =="src":
        lines = open(data_file, encoding='utf-8').\
        read().strip().split('\n')
    if label =="trgt":
        lines = open(data_trgt_file, encoding='utf-8').\
        read().strip().split('\n')


    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs





def prepareData(lang1, lang2, reverse=True, label="src"):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, label)
    print("Read %s sentence pairs" % len(pairs))
    pairs = pairs[1:]
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        input_lang.addSentence(pair[1])
    output_lang = input_lang
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs




def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word,0) for word in tokenizer.tokenize(sentence)]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)



def get_dataloader(batch_size, label ="src"):
    input_lang, output_lang, _ = prepareData('ar', 'arz', label="vocab")
    if label == "src":
        _, _, pairs = prepareData('ar', 'arz', label="src")
    elif label == "trgt":
        _, _, pairs = prepareData('ar', 'arz', label="trgt")
  
    pairs = pairs[1:]
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader




def beam_search(encoder, decoder, sentence, input_lang, output_lang, beam_width=5, max_length=MAX_LENGTH):
    with torch.no_grad():
        sentence = normalizeString(sentence)
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        # Initialize the list to hold the completed beams
        completed_beams = []

        # Start the beam search process

        candidate_beams = []
        decoder_hidden = encoder_hidden

        # Get the top-k candidates and their probabilities
        topk_probs, topk_indices = decoder_outputs.squeeze().topk(beam_width)
        topk_probs = topk_probs.T #(beam, max_length)
        topk_indices = topk_indices.T

        # Expand the current beam with the top-k candidates

        for i in range(beam_width):
          decoder_output=[]
          candidate_prob =0.0
          for j in range(MAX_LENGTH):

            new_candidate_idx = topk_indices[i][j].item()
            decoder_output.append(new_candidate_idx)

            candidate_prob += topk_probs[i][j].item()
            new_decoder_input = torch.tensor([decoder_output], device=device)
            new_beam_score =(candidate_prob)

            # Check if the candidate is the end token
            if new_candidate_idx == EOS_token:
                completed_beams.append((decoder_output, new_beam_score))
                break
            if j == (MAX_LENGTH-1):
              completed_beams.append((decoder_output, new_beam_score))

        # Sort the completed beams and select the one with the highest score
        completed_beams.sort(key=lambda x: x[1], reverse=True)

        decoded_words = [output_lang.index2word.get(word,tokenizer.unk_token)for word in completed_beams[0][0]]

        return decoded_words  # No attention scores for beam search
def evaluateRandomly(encoder, decoder,input_lang, output_lang, data= None,beam =1, n=None):
    outputs = []
    targets =[]
    i=0
    for idx in data.index:
        if n != None:
            if i == n:
                break

        targets.append([normalizeString(data.target_lang[idx])])
        print('input', data.source_lang[idx])
        print('output', data.target_lang[idx])
        output_words = beam_search(encoder, decoder,data.source_lang[idx], input_lang, output_lang, beam_width= beam)

        output_sentence = tokenizer.convert_tokens_to_string(output_words)
        output_sentence= output_sentence.replace("[SEP]", "")
        output_sentence = output_sentence.replace("[PAD]", "")
        output_sentence = output_sentence.replace("[CLS]","")
        outputs.append(output_sentence)

        print('prediction', output_sentence)
        print('')
        i=i+1
    return targets, outputs

def load_checkpoit(checkpoint_file, model, optimizer):
    print("Loading checkpoint...")
    checkpoint= torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    
def save_checkpoit(checkpoint_file, model, optimizer):    
    print("Saving checkpoint...")
    torch.save({
                
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            
                }, checkpoint_file)



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
