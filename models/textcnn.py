from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nnfunc import NNFunc, copy_param_val


class TextCNNFunc(NNFunc):
    def __init__(self, **kwargs):
        super().__init__()
        self.state_dict = OrderedDict({
            "embedding.weight": [20002, 300],
            "convs.0.weight": [256, 1, 2, 300],
            "convs.0.bias": [256],
            "convs.1.weight": [256, 1, 3, 300],
            "convs.1.bias": [256],
            "convs.2.weight": [256, 1, 4, 300],
            "convs.2.bias": [256],
            "fc.weight": [2, 768],
            "fc.bias": [2]
        })


    def conv_and_pool(self, x, params, conv_name):
        x = F.relu(F.conv2d(x, params[f"{conv_name}.weight"], params[f"{conv_name}.bias"])).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
        
    def __call__(self, x, params):
        params = self.instantiate(params)
        x = x.transpose(0, 1)
        x = F.embedding(x, params['embedding.weight'])
        x = x.unsqueeze(1)
        # x = x.unsqueeze(1)
        x = torch.cat([self.conv_and_pool(x, params, f"convs.{i}") for i in range(3)], 1)
        x = F.dropout(x)
        
        x = F.linear(x, params['fc.weight'], params['fc.bias'])
        return x



class TextCNN(nn.Module):
    def __init__(self, vocab_size = 20002, **kwargs):
        super(TextCNN, self).__init__()
        self.filter_sizes = (2, 3, 4)
        self.embed = 300
        self.num_filters = 256
        self.dropout = 0.5
        self.num_classes = 2
        self.n_vocab = vocab_size
        ## 通过padding_idx将<PAD>字符填充为0
        self.embedding = nn.Embedding(self.n_vocab, self.embed) #, padding_idx=word2idx['<PAD>'])
        ## in_channels, out_channels, kernel_size(h,w)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        ## x.size(2) is the kernel size: [batch_size, self.num_filters, x.size(2)] -> [batch_size, self.num_filters, 1]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
        
    def forward(self, x, params = None, **kwargs):
        if params is not None:
            copy_param_val(self, params, **kwargs)
        x = x.transpose(0, 1)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
