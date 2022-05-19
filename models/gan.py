import os, time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from .nnfunc import copy_param_val, model_grad_to_pool, NNFunc, _total_param_count


## from [-1, 1] to [0, 255] uint8
def mnist_denormalize(x):
    x = (0.5 * (x + 1)) * 255.0
    return x.to(torch.uint8)

## from [-1, 1] to [0, 255] uint8
def cifar10_denormalize(x):
    dm, ds = torch.FloatTensor([[[0.4915]], [[0.4823]], [[0.4468]]]), torch.FloatTensor([[[0.2023]],[[0.1994]],[[0.2010]]])
    dm, ds = dm.to(x.device), ds.to(x.device)
    x = x.mul_(ds).add_(dm)
    x = x * 255.0
    return x.to(torch.uint8)
    


def show(img, name, ax = None):
    img = (img.float() / 255.0)
    img.clamp(0.0, 1.0)
    if(img.shape[0] == 1):
        # if only has one channel
        img = img.squeeze(0)

    
    if(ax is None):
        fig, ax = plt.subplots(figsize=(4, 4))
    if(len(img.shape) == 3):
        img = img.permute(1, 2, 0)
    npimg = img.numpy()
    # print(npimg.shape)
    # npimg = ((npimg - npimg.min()) * 255 / (npimg.max() - npimg.min())).astype('uint8')
    ax.imshow(npimg)


def compare_show(imgs_a, imgs_b, name):
    ## convert the images
    ncols = 16
    nrows = (len(imgs_a) // 16 + 1) * 2
    fig, axs = plt.subplots(figsize = (ncols*4, nrows*4), ncols = ncols, nrows = nrows)
    if(len(imgs_a) == 1):
        show(imgs_a[0], "", axs[0][0])
        show(imgs_b[0], "", axs[1][0])
    else:
        for i in range(len(imgs_a)):    
            show(imgs_a[i], "", axs[2*(i//16) + 0][i % 16])
            show(imgs_b[i], "", axs[2*(i//16) + 1][i % 16])
    plt.savefig("img/result_{}.png".format(name))
    plt.cla()
    plt.close("all")
    print("Results stored @ {}".format(name))



# class GAN(nn.Module):
#     def __init__(self, d_g = 128, d_d = 128, device = torch.device('cpu'), image_channel = 1, **kwargs):
#         super(GAN, self).__init__()
#         self.G = generator(d_g, image_channel)
#         self.D = discriminator(d_d, image_channel)
#         self.device = device
#         self.eval_counter = 0
#         ## control the discriminator with other free parameters
#         self.D_opt = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))

#         self.D.weight_init(0.0, 0.02)
        

#     def _forward_impl(self, x):
#         z_ = torch.randn((x.shape[0], 100)).view(-1, 100, 1, 1)
#         z_ = z_.to(self.device)
#         return self.G(z_)

        
#     def forward(self, x, params):
#         copy_param_val(self.G, params)
#         return self._forward_impl(x)

#     def loss_grad_step(self, x, y, params, perm_key, lr = 0.1, opt = None, iter_no = 0, weight_decay = 5e-4):
#         self.zero_grad()
#         copy_param_val(self.G, params[perm_key])
#         # D loss grad step
        
#         self.D.zero_grad()
#         y_real_ = torch.ones(x.shape[0])
#         y_fake_ = torch.zeros(x.shape[0])
#         x, y_real_, y_fake_ = x.to(self.device), y_real_.to(self.device), y_fake_.to(self.device)

#         D_result = self.D(x).squeeze()
#         D_real_loss = F.binary_cross_entropy(D_result, y_real_)
        
#         G_result = self._forward_impl(x)
#         D_result = self.D(G_result).squeeze()

#         D_fake_loss = F.binary_cross_entropy(D_result, y_fake_)
#         D_train_loss = D_real_loss + D_fake_loss
        
#         D_train_loss.backward()
#         ## update the params related with the discriminator
#         self.D_opt.step()

#         self.G.zero_grad()
#         ## G loss grad step
#         G_result = self._forward_impl(x)
#         D_result = self.D(G_result).squeeze()
#         G_train_loss = F.binary_cross_entropy(D_result, y_real_)
#         G_train_loss.backward()
        
#         ## update the parameters related with the generator
#         with torch.no_grad():
#             G_grad = model_grad_to_pool(self.G, params)
#             grad = opt[1].calc_grad(G_grad, lr[1])
#             params[perm_key] = params[perm_key] - grad
#         return (D_train_loss + G_train_loss).item(), params

#     def eval_step(self, x, y, params, metric, iter_no = 0, instance_name = 'default'):
#         self.eval_counter += 1
#         if(iter_no <= 3): ## only evaluate for 10 batches
#             DEVICE = torch.device('cuda:0')
#             G_result = self.forward(x, params)
#             G_result = G_result.cpu()
#             is_one_channel = (G_result.shape[1] == 1)
#             if(is_one_channel):
#                 G_result = torch.repeat_interleave(G_result, 3, dim = 1)
#                 x = torch.repeat_interleave(x, 3, dim = 1)
#                 G_result = mnist_denormalize(G_result)
#                 x = mnist_denormalize(x)
#             else:
#                 ## denormalize with CIFAR10
#                 G_result = cifar10_denormalize(G_result)
#                 x = cifar10_denormalize(x)

#             G_result = G_result.to(DEVICE)
#             x = x.to(DEVICE)
#             metric.inception.to(DEVICE)
#             # print(torch.min(G_result), torch.max(G_result))
#             metric.update(G_result, real = False)
#             metric.update(x, real=True)
#             if(iter_no == 0):
#                 compare_show(G_result.cpu(), x.cpu(), name = f'gan_{instance_name}_{self.eval_counter}')
#         else:
#             pass

#     def total_param_count(self):
#         return _total_param_count(self)
        
        
        
        













# Generator Class (This is already a DCGAN)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128, image_channel = 1):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, image_channel, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

 #Discriminator Class
class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, image_channel = 1):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(image_channel, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
