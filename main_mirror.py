from collections import OrderedDict
from models import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.lenet import LeNetFunc, LeNet
from models.textcnn import TextCNNFunc, TextCNN
from models.resnet import resnet18
from models.m5 import M5
from models.vgg import vgg11, vgg19
from models.nnfunc import is_bn_weight,copy_param_delta
from dataset import get_imdb_dataloader, IMDB, get_cifar10_dataloader, get_speechcommand_dataloader, get_dermamnist_dataloader, get_mnist_dataloader
from itertools import cycle
import torchmetrics
from optim import EasyAdam, EasySGD
from scheduler import CosineAnnealingLR, StepLR

import argparse
parser = argparse.ArgumentParser(description="model_overloading_main")
parser.add_argument('--which_cuda', default=0, type=int, help='the index of cuda to use')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--max_iter', default=100000, type=int, help='the training epoch max iter')
args = parser.parse_args()

device = torch.device(f'cuda:{args.which_cuda}')
print(device)

### the size of weight pool
param_num = 15000_000

def get_grad(model):
    return [param.grad.data for param in model.parameters()]

def _model_grad_to_pool(grad, params, offset):
    params_grad = torch.zeros_like(params)
    grad = grad.view(-1)
    param_size = grad.shape[0]
    repete_count = (param_size + offset) // len(params)
    if(repete_count == 0):
        params_grad[offset:param_size + offset] = grad
        offset += param_size
        return params_grad, offset
    cur = 0
    params_grad[offset:] = grad[cur:cur+len(params) - offset]
    repete_count -= 1
    cur += len(params) - offset
    for i in range(repete_count):
        params_grad += grad[cur:cur+len(params)]
        cur += len(params)
    offset = param_size + offset - (repete_count + 1) * len(params)
    params_grad[:offset] += grad[cur:]
    return params_grad, offset
    
def model_grad_to_pool(model, params, bn_params = None):
    if params is None:
        return None, None
    params_grad = torch.zeros_like(params)
    if bn_params is not None:
        bn_params_grad = torch.zeros_like(bn_params)
    else:
        bn_params_grad = None
    offset = 0
    bn_offset = 0
    for name, param in model.named_parameters():
        if bn_params is not None and is_bn_weight(name):
            local_param_grad, bn_offset = _model_grad_to_pool(param.grad.data, bn_params, bn_offset)
            bn_params_grad += local_param_grad
        else:
            local_param_grad, offset = _model_grad_to_pool(param.grad.data, params, offset)
            params_grad += local_param_grad
    return params_grad, bn_params_grad

def print_banner(s):
    print("=*="*20 + f" {s} " + "=*="*20)

TASK_MODEL_ENTRY = OrderedDict({
    # "cifar10-LeNet": (get_cifar10_dataloader, LeNet, {"feat_dim": 400, "out_dim": 10}), ## 60%
    # "cifar10-VGG": (get_cifar10_dataloader, vgg11, {"num_classes": 10}),
    # "mnist-LeNet": (get_mnist_dataloader, LeNet, {}), ## 98%
    # "dermamnist-LeNet": (get_dermamnist_dataloader, LeNet, {}), ## 75%
    "cifar10-ResNet18": (get_cifar10_dataloader, resnet18, {"num_classes": 10}),  ## 95% 
    # "imdb-TextCNN": (get_imdb_dataloader, TextCNN, {}), ## 89.6%
    # "speechcommand-M5": (get_speechcommand_dataloader, M5, {}), ## >94%
    # "mnist-ResNet18": (get_mnist_dataloader, resnet18, {"num_classes": 10}),  
})

def _evaluate(model, params, test_loader, **kwargs):
    metric = torchmetrics.Accuracy()
    model.eval()
    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.squeeze().cpu().long()
        preds = model(x, params, **kwargs).softmax(dim = -1).cpu()
        acc = metric(preds, y)
    acc = metric.compute()
    return acc
    
def evaluate(task_model_instances, params, **kwargs):
    acc_list = dict()
    for key in task_model_instances:
        print_banner(f"Eval {key.upper()}")
        ##TODO: random perm
        acc = _evaluate(task_model_instances[key][2], params, task_model_instances[key][1], **kwargs)
        print(f"Acc ({key.upper()}): {acc*100:.2f}%")
        acc_list[key] = acc * 100
    return acc_list

def main():
    weight_pool = Variable(torch.zeros((param_num,)))
    nn.init.uniform_(weight_pool, a = -0.1, b = 0.1)

    weight_pool = weight_pool.to(device)
    weight_pool.requires_grad = True
    weight_pool = None

    bn_weight_pool = Variable(torch.zeros((10000,)), requires_grad = True)
    nn.init.ones_(bn_weight_pool)
    bn_weight_pool = bn_weight_pool.to(device)
    bn_weight_pool = None 
    
    bn_bias_pool = Variable(torch.zeros((10000,)), requires_grad = True)
    nn.init.zeros_(bn_bias_pool)
    bn_bias_pool = bn_bias_pool.to(device)
    bn_bias_pool = None
    
    # parameters for gradient average 
    a_pool = Variable(torch.ones((param_num,)), requires_grad = True)
    b_pool = Variable(torch.zeros((param_num,)), requires_grad = True)
    

    task_model_instances = OrderedDict()
    best_test_acc_list = dict()
    for key in TASK_MODEL_ENTRY:
        kwargs = TASK_MODEL_ENTRY[key][2]
        train_loader, test_loader = TASK_MODEL_ENTRY[key][0](**kwargs)
        model = TASK_MODEL_ENTRY[key][1](**kwargs)
        # print(f"{key}:{model.total_param()}")
        perm_key = torch.range(0, param_num-1).long().to(device) # 
        # perm_key = torch.randperm(param_num).to(device)
        print("Before cycling, the {} train loader has {} batches.".format(key, len(train_loader)))
        task_model_instances[key] = (cycle(train_loader), test_loader, model, perm_key)
        best_test_acc_list[key] = 0
    
    num_iters = args.max_iter
    lr = args.lr

    for key in task_model_instances:
        # optimizer = optim.Adam(task_model_instances[key][2].parameters(), lr = args.lr)
        optimizer = optim.SGD(task_model_instances[key][2].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    adam = EasyAdam()
    bn_adam = EasyAdam()
    adam_mirror = torch.optim.Adam([a_pool], lr = lr)
    scheduler_mirror = torch.optim.lr_scheduler.CosineAnnealingLR(adam_mirror, T_max=num_iters//500)
    
    for idx in range(num_iters):
        if weight_pool is not None:
            weight_pool.grad = None
        for key in task_model_instances:
            # print_banner(key.upper())
            train_loader = task_model_instances[key][0]
            model = task_model_instances[key][2]
            model.to(device)
            model.train()
            model.zero_grad()
            perm_key = task_model_instances[key][3]
            x, y = next(train_loader)
            x = x.to(device)
            y = y.squeeze().to(device).long()
            ## TODO: random perm
            loss = F.cross_entropy(model(x, params = weight_pool, bn_params = bn_weight_pool, a=a_pool, b=b_pool), y)
            loss.backward()
            if weight_pool is None:
                optimizer.step()
            # for name, param in model.named_parameters():
            #     print(param.grad)
            #     break
            param_grad, bn_param_grad = model_grad_to_pool(model, weight_pool, bn_params=bn_weight_pool)
            ## accumulate the gradient over tasks
            with torch.no_grad():
                if weight_pool is not None:
                    if(weight_pool.grad is None):
                        weight_pool.grad = torch.zeros_like(param_grad)
                    weight_pool.grad[perm_key] += param_grad
                    if bn_param_grad is not None:
                        if(bn_weight_pool.grad is None):
                            bn_weight_pool.grad = torch.zeros_like(bn_param_grad)
                        bn_weight_pool.grad += bn_param_grad
            # print(param_grad.norm())
            # loss.backward()
            # print(f"Grad Norm: ", weight_pool.grad.norm())
            if((idx % 100) == 0):
                print(f"{key}\t Iter = {idx+1}\tLoss = {loss:.8f}\t LR = {lr:.6f}")
        if(idx % 1000 == 0):
            cur_test_acc_list = evaluate(task_model_instances, weight_pool, bn_params = bn_weight_pool)
            for key in cur_test_acc_list:
                best_test_acc_list[key] = max(best_test_acc_list[key], cur_test_acc_list[key])
            if bn_weight_pool is not None:
                print('Check weather the bn_weight_pool has been updated')
                print(f'min is {bn_weight_pool.min().item()} max is {bn_weight_pool.max().item()} mean is {bn_weight_pool.mean().item()}')
            if weight_pool is not None:
                print('the min/max/mean of current weight_pool')
                print(f'min is {weight_pool.min().item()} max is {weight_pool.max().item()} mean is {weight_pool.mean().item()}')
            print(best_test_acc_list)
            # lr = scheduler.get_lr(lr, idx // 100)
            scheduler_mirror.step()
            lr = adam_mirror.state_dict()['param_groups'][0]['lr']
        if weight_pool is not None:
            grad = adam.calc_grad(weight_pool.grad, lr, p = weight_pool.data)
            weight_pool = weight_pool - grad
            if bn_weight_pool is not None:
                bn_grad = bn_adam.calc_grad(bn_weight_pool.grad, lr, p = weight_pool.data)
                bn_weight_pool = bn_weight_pool - bn_grad
        # optimizer.step()    

import numpy as np
if __name__ == '__main__':
    # model = resnet18(num_classes=10)
    # total = 0
    # for name, param in model.named_parameters():
    #     if name.find('bn') != -1 and name.find('weight') != -1 or name.find('shortcut.1.weight') != -1:
    #         print(name)
    #         print(param.min())
    #         total += np.prod(param.shape)
    # print(total)
    main()
