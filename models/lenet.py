from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .nnfunc import NNFunc, copy_param_val


class LeNetFunc(NNFunc):
    def __init__(self, feat_dim = 400, out_dim = 10, **kwargs):
        ## feat_dim 400 for 32 x 32, 256 for 28 x 28
        super().__init__()
        self.state_dict = OrderedDict({
            "conv1.weight": [6,3,5,5],
            "conv2.weight": [16,6,5,5],
            "fc1.weight": [120, feat_dim],
            "fc1.bias": [120],
            "fc2.weight": [84, 120],
            "fc2.bias": [84],
            "fc3.weight": [out_dim, 84],
            "fc3.bias": [out_dim]
        })

        
    def __call__(self, x, params):
        params = self.instantiate(params)
        x = F.conv2d(x, params['conv1.weight'])
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(x, params['conv2.weight'])
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.linear(x,
                    params['fc1.weight'],
                    params['fc1.bias'])
        x = F.relu(x)
        x = F.linear(x,
                     params['fc2.weight'],
                     params['fc2.bias'])
        x = F.relu(x)
        x = F.linear(x,
                     params['fc3.weight'],
                     params['fc3.bias'])
        return x




    
class LeNet(nn.Module):
    def __init__(self, feat_dim = 400, out_dim = 10, **kwargs):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, bias = False)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias = False)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(feat_dim, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, out_dim)
        # conv = nn.Conv2d(6, 16, kernel_size=5,bias = True, stride = self.first_m.stride, padding = self.first_m.padding)
        # conv = conv.to(self.device)
        # conv.weight = Parameter(true_weight)
        # conv.bias = self.first_m.bias
        # x_2 = conv(x_1)

        
    def forward(self, x, params=None, **kwargs):
        if params is not None:
            copy_param_val(self, params, **kwargs)
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


    
