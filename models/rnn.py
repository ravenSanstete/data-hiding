from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nnfunc import NNFunc, copy_param_val

class uniRNN(nn.Module):
    def __init__(self, vocab_size = 20002):
        super(uniRNN, self).__init__()
        self.embed = 300
        self.hidden_size = 200
        self.num_filters = 256
        self.dropout = 0.5
        self.num_classes = 2
        self.n_vocab = vocab_size
        self.char_embedding = nn.Embedding(self.n_vocab, self.embed)
        self.lstm = nn.LSTM(self.embed, self.hidden_size, batch_first = True)
        self.output_layer = nn.Linear(self.hidden_size, self.num_classes)
        

    def forward(self, sentence_batch, params, hidden = None):
        # # sentence_batch = Variable(sentence_batch)
        # print(f'sentence_batch.size() is {sentence_batch.size()}')
        # print(sentence_batch.shape)
        copy_param_val(self, params)

        if len(sentence_batch.size()) == 2:
            char_embedding = self.char_embedding(sentence_batch)
        else:
            shape = sentence_batch.size()
            sentence_batch_flat = sentence_batch.view(shape[0] * shape[1], shape[2])
            char_embedding = torch.mm(sentence_batch_flat, self.char_embedding.weight)
            char_embedding = char_embedding.view(shape[0], shape[1], char_embedding.size()[-1])

        char_embedding = F.tanh(char_embedding)
        # char_embedding.shape = [batch_size, seq_size, embed_size]
        if not hidden:
            lstm_out, new_hidden = self.lstm(char_embedding)
        else:
            lstm_out, new_hidden = self.lstm(char_embedding, hidden)
        # lstm_out.shape = [batch_size, seq_size, hidden_size]
        lstm_out = lstm_out.contiguous()
        # print(lstm_out.shape)
        lstm_out = lstm_out[:,-1,:]
        # lstm_out.shape = [batch_size, hidden_size]
        logits = self.output_layer(lstm_out)
        # logits.shape = [batch_size, output_size]
        return logits


class biRNN(nn.Module):
    def __init__(self, vocab_size = 20002):
        super(biRNN, self).__init__()
        self.embed = 300
        self.hidden_size = 200
        self.num_filters = 256
        self.dropout = 0.5
        self.num_classes = 2
        self.n_vocab = vocab_size
        self.char_embedding = nn.Embedding(self.n_vocab, self.embed)
        self.lstm = nn.LSTM(self.embed, self.hidden_size, batch_first = True, bidirectional=True)
        self.output_layer = nn.Linear(2 * self.hidden_size, self.num_classes)
        

    def forward(self, sentence_batch, params, hidden = None, **kwargs):
        # sentence_batch = Variable(sentence_batch)
        copy_param_val(self, params, **kwargs)
        if len(sentence_batch.size()) == 2:
            char_embedding = self.char_embedding(sentence_batch)
        else:
            shape = sentence_batch.size()
            sentence_batch_flat = sentence_batch.view(shape[0] * shape[1], shape[2])
            char_embedding = torch.mm(sentence_batch_flat, self.char_embedding.weight)
            char_embedding = char_embedding.view(shape[0], shape[1], char_embedding.size()[-1])
            
        char_embedding = F.tanh(char_embedding)
        
        if not hidden:
            lstm_out, new_hidden = self.lstm(char_embedding)
        else:
            lstm_out, new_hidden = self.lstm(char_embedding, hidden)
        lstm_out = lstm_out.contiguous()
        lstm_out = lstm_out[:,-1,:] + lstm_out[:,0,:]
        logits = self.output_layer(lstm_out)

        return logits