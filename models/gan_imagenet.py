from typing import OrderedDict
import torch
import torch.nn as nn
from .nnfunc import NNFunc, copy_param_val


'''
class g_imagenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
            ('conv1',nn.ConvTranspose2d(256,512,4,2,0,bias=False)),
            ('relu1',nn.LeakyReLU(0.2)),

            ('conv2',nn.ConvTranspose2d(512,256,4,1,0,bias=False,dilation=2)),
            ('relu2',nn.LeakyReLU(0.2)),

            ('conv3',nn.ConvTranspose2d(256,128,4,3,0,bias=False)),
            ('relu3',nn.LeakyReLU(0.2)),

            ('conv4',nn.ConvTranspose2d(128,64,4,1,0,bias=False,dilation=2)),
            ('relu4',nn.LeakyReLU(0.2)),

            ('conv5',nn.ConvTranspose2d(64,32,4,3,0,bias=False)),
            ('relu5',nn.LeakyReLU(0.2)),

            ('conv6',nn.Conv2d(32,3,7,bias=False)),
            ('tanh',nn.Tanh()),

        ]))

    def forward(self,x):
        return self.main(x)
'''

# modified based on DC-GAN to fit (3,224,224)
class Generator(nn.Module):
    def __init__(self,nc=3,nz=100,ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(OrderedDict([
            # input is Z, going into a convolution
            ('conv1',nn.ConvTranspose2d( nz, ngf * 8, 7, 1, 0, bias=False)),
            ('bn1',nn.BatchNorm2d(ngf * 8)),
            ('relu1',nn.ReLU(True)),
            # state size. (ngf*8) x 7 x 7
            ('conv2',nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
            ('bn2',nn.BatchNorm2d(ngf * 4)),
            ('relu2',nn.ReLU(True)),
            # state size. (ngf*4) x 14 x 14
            ('conv3',nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
            ('bn3',nn.BatchNorm2d(ngf * 2)),
            ('relu3',nn.ReLU(True)),
            # state size. (ngf*2) x 28 x 28
            ('conv4',nn.ConvTranspose2d( ngf * 2, ngf, 4, 4, 0, bias=False)),
            ('bn4',nn.BatchNorm2d(ngf)),
            ('relu4',nn.ReLU(True)),
            # state size. (ngf) x 112 x 112
            ('conv5',nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)),
            ('tanh',nn.Tanh())
            # state size. (nc) x 224 x 224
            ])
        )

    def forward(self,x,params = None,**kwargs):
        if params is not None:
            copy_param_val(self, params, **kwargs)
        return self.main(x)

'''
x = torch.randn(1,100,1,1)
g = Generator()
output = g(x)
print(output.shape)
'''
