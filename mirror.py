import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

from main_y import get_grad, model_grad_to_pool, print_banner, get_param_group
from optim import EasyAdam, EasySGD
from itertools import cycle
from models.nnfunc import copy_param_val, copy_param_delta

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--which_cuda', default=0, type=int, help='the index of cuda to use')
parser.add_argument('--use_scheduler', action='store_true', help='whether use schedule during training')
parser.add_argument('--use_decay_momentum', action='store_true', help='whether use dacay and momentum during training')
parser.add_argument('--overload', action='store_true', help='whether overloading')
parser.add_argument('--diff', action='store_true', help='show the diff between w and w/o overload')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--max_iter', default=200, type=int, help='the training epoch max iter')
parser.add_argument('--param_num', default=15000000, type=int, help='the size of weight group')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
print(args)

device = 'cuda:{}'.format(args.which_cuda)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
from models.resnet import resnet18
net = resnet18(num_classes=10)
net = net.to(device)
net_oral =resnet18(num_classes=10) 
net_oral = net_oral.to(device)

criterion = nn.CrossEntropyLoss()
optims = [EasySGD()]
if args.use_decay_momentum is True:
    print('---------use optimizer SGD decay and momentum----------')
    optimizer_oral = optim.SGD(net_oral.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
    optimizer_oral = optim.SGD(net_oral.parameters(), lr = args.lr)
    optimizer= optim.SGD(net.parameters(), lr = args.lr)

if args.use_scheduler is True:
    print('--------use scheduler---------')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter)
    scheduler_oral = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_oral, T_max=args.max_iter)


def model_param_compare(a, b):
    a_params = b_params = []
    a_buffers = b_buffers = []
    for a_name, a_param in a.named_parameters():
        a_params.append(a_param)
    for b_name, b_param in b.named_parameters():
        b_params.append(b_param)
    
    for a_buffer in a.buffers():
        a_buffers.append(a_buffer)
    for b_buffer in b.buffers():
        b_buffers.append(b_buffer)

    for i in range(len(a_params)):
        if torch.norm(a_params[i].view(-1) - b_params[i].view(-1)) != 0:
            print('________________Different Param________________')
            print(a_params[i].view(-1)[:10])
            print(b_params[i].view(-1)[:10])
            return False
    for i in range(len(a_buffers)):
        if torch.sum(torch.abs(a_buffers[i].view(-1) - b_buffers[i].view(-1))) != 0:
            print('________________Different Buffer________________')
            print(a_buffers[i].view(-1)[:10])
            print(b_buffers[i].view(-1)[:10])
            return False
    for batch_idx, (inputs, targets) in enumerate(testloader): 
        print('check output')
        inputs = inputs.to(device)
        print(a(inputs).view(-1)[:10])
        print(b(inputs).view(-1)[:10])
        break
    return True

model_param_compare(net, net_oral)

# Training
def train(epoch, weight_group = None):
    print('\nEpoch: %d' % epoch)
    net.train()
    net_oral.train()
    train_loss = train_loss_oral = 0
    correct = correct_oral = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        net.zero_grad()
        net_oral.zero_grad()
        if args.diff is False:
            copy_param_val(net, params = weight_group)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            p_id = 'oral'
            grad = copy_param_delta(net, params = weight_group)
            weight_group[p_id] = weight_group[p_id] - grad[p_id]
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | LR: %.4f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), optimizer.state_dict()['param_groups'][0]['lr'], 100.*correct/total, correct, total))
        else: # diff
            optimizer_oral.zero_grad()
            outputs_oral = net_oral(inputs)
            loss_oral = criterion(outputs_oral, targets)
            loss_oral.backward()
            optimizer_oral.step()
            train_loss_oral += loss_oral.item()
            _, predicted_oral = outputs_oral.max(1)
            correct_oral += predicted_oral.eq(targets).sum().item() 

            # outputs = net(inputs,params = weight_group)
            # loss = criterion(outputs, targets)
            # loss.backward()
            # param_grads = model_grad_to_pool(net, weight_group)
            # p_id=0
            # if(weight_group[p_id].grad is None):
            #     weight_group[p_id].grad = torch.zeros_like(param_grads[p_id])
            # weight_group[p_id].grad += param_grads[p_id]
            # grad = optims[p_id].calc_grad(weight_group[p_id].grad, args.lr, p = weight_group[p_id].data)
            # weight_group[p_id] = weight_group[p_id] - grad

            copy_param_val(net, params = weight_group)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


            p_id = 'oral'
            grad = copy_param_delta(net, params = weight_group)
            weight_group[p_id] = weight_group[p_id] - grad
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), ' Loss: %.3f | LR: %.4f | Acc: %.3f%% (%d/%d)  Loss_oral: %.3f | LR_oral: %.4f | Acc_oral: %.3f%% (%d/%d) '
                        % (train_loss/(batch_idx+1), optimizer.state_dict()['param_groups'][0]['lr'], 100.*correct/total, correct, total, 
                           train_loss_oral/(batch_idx+1), optimizer_oral.state_dict()['param_groups'][0]['lr'], 100.*correct_oral/total, correct_oral, total))

def test(epoch, weight_group = None):
    # print(model_param_compare(net, net_oral))
    global best_acc
    net.eval()
    net_oral.eval()
    test_loss_oral = test_loss = 0
    correct_oral = correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs,params = weight_group)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_oral = net_oral(inputs)
            loss_oral = criterion(outputs_oral, targets)
            test_loss_oral+= loss_oral.item()
            _, predicted_oral = outputs_oral.max(1)
            correct_oral += predicted_oral.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | LR: %.4f | Acc: %.3f%% (%d/%d) Loss_oral: %.3f | LR_oral: %.4f | Acc_oral: %.3f%% (%d/%d) '
                         % (test_loss/(batch_idx+1), optimizer.state_dict()['param_groups'][0]['lr'], 100.*correct/total, correct, total, 
                           test_loss_oral/(batch_idx+1), optimizer_oral.state_dict()['param_groups'][0]['lr'], 100.*correct_oral/total, correct_oral, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

if __name__ == '__main__':
    if args.diff is True:
        args.overload = True
    if args.overload:
        print("initial weight group, overloading")
        weight_group = get_param_group(oral = True, device = device, model_oral = net_oral, params_num= args.param_num)
        group_num = len(weight_group) if weight_group is not None else 0
    else:
        weight_group = None

    for epoch in range(start_epoch, start_epoch + args.max_iter):
        if(weight_group is None):
            print("___________________Weight group is EMPTY => ORAL _____________________")
        train(epoch, weight_group)
        test(epoch, weight_group)
        if args.use_scheduler is True:
            scheduler.step()
            scheduler_oral.step()
            # args.lr = optimizer.state_dict()['param_groups'][0]['lr']