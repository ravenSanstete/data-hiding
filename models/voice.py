from typing import OrderedDict
import torch
import torch.nn as nn
from .nnfunc import NNFunc, copy_param_val

class voiceG(nn.Module):
    def __init__(self,nz=100,nc=1):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
            # input (100,1)
            ('conv1',nn.ConvTranspose1d(nz, 64, 3, 1 , 0 , bias=False)),
            ('bn1',nn.BatchNorm1d(64)),
            ('relu1',nn.ReLU(True)),
            # (64,3)

            ('conv2',nn.ConvTranspose1d(64, 64, 27, 3 , 0,bias=False, dilation=2)),
            ('bn2',nn.BatchNorm1d(64)),
            ('relu2',nn.ReLU(True)),
            # (64,59)

            ('conv3',nn.ConvTranspose1d(64, 32, 38, 3 , 0,bias=False, dilation=2)),
            ('bn3',nn.BatchNorm1d(32)),
            ('relu3',nn.ReLU(True)),
            # (32,249)

            ('conv4',nn.ConvTranspose1d(32, 32, 52, 3 , 2,bias=False, dilation=5)),
            ('bn4',nn.BatchNorm1d(32)),
            ('relu4',nn.ReLU(True)),
            # (32,996)
 
            ('conv5',nn.ConvTranspose1d(32, nc, 80, 16 , 0,bias=False)),
            ('tanh',nn.Tanh())
            # (1,16000)
            ])
        )

    def forward(self,x,params = None,**kwargs):
        if params is not None:
            copy_param_val(self, params, **kwargs)
        x = self.main(x)
        return x