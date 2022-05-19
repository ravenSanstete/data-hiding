
from ast import arg
from collections import OrderedDict
from typing import Dict, Text
from defense import defense_to_weight_pool

from models import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from train_one_batch import * 
from models.lenet import LeNet
from models.textcnn import TextCNNFunc, TextCNN
from models.resnet import resnet18
from models.m5 import M5
from models.vgg import vgg11, vgg16, vgg19
from models.dcgan import discriminator,generator
from models.gan_imagenet import Generator
from models.nnfunc import *
from models.voice import voiceG
from dataset import get_imdb_dataloader, IMDB, get_cifar10_dataloader_gan, get_cifar10_dataloader,get_facescrub_dataloader, get_speechcommand_dataloader, get_dermamnist_dataloader, get_mnist_dataloader,get_imagenet_dataloader,get_gtsrb_dataloader
from itertools import cycle
import random
from utils import progress_bar
from TaskModel import TaskModel, WeightGroup
from train_one_batch import get_buffer_param
import os
import torchvision.utils as vutils
import argparse

## TODO: Test_interval of each task_model is the same, should they be different?
test_interval = 391 ## for cifar-resnet

## task_model_entry = (dataloader_func, model, model_kwargs, train_one_batch_fn, \
#                           optimizer, optim_kwargs, 
#                           scheduler, sch_kwargs, loss_fn) # the best acc of full model
TASK_MODEL_ENTRY = OrderedDict({
    "cifar10-ResNet18": (get_cifar10_dataloader, resnet18, {"num_classes": 10}, train_one_batch_for_clf, \
                             torch.optim.SGD, {"lr":0.1, "momentum":0.9, "weight_decay":5e-4}, \
                             torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()),  ## 95.65% <- SGD with lr 0.1
    # "imdb-TextCNN": (get_imdb_dataloader, TextCNN, {}, train_one_batch_for_clf, \
    #                        torch.optim.SGD, {"lr":0.1, "momentum":0.9, "weight_decay":5e-4}, \
    #                        torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()),  ## 90.04% <- SGD with lr 0.01
    # "speechcommand-M5": (get_speechcommand_dataloader, M5, {}, train_one_ebatch_for_clf, \
    #                         torch.optim.SGD, {"lr":0.1, "momentum":0.9, "weight_decay":5e-4}, \
    #                         torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()), ## 96.92% <- SGD with lr 0.1
    "cifar-onlyG": (get_cifar10_dataloader_gan,generator,{}, train_one_batch_onlyG,
 	                torch.optim.Adam,{"lr":2e-4,"betas":(0.5,0.999)},
 	                torch.optim.lr_scheduler.CosineAnnealingLR, {}, Image_Pair_Loss),
    # "imdb-onlyG": (None,Generator,{"nc":1}, train_one_batch_onlyG,
 	#                 torch.optim.Adam,{"lr":1e-3,"betas":(0.5,0.999)},
 	#                 torch.optim.lr_scheduler.CosineAnnealingLR, {}, Image_Pair_Loss),
    #"Imagenet-onlyG":(get_imagenet_dataloader,Generator,{},train_one_batch_onlyG,
    #                    torch.optim.Adam,{"lr":1e-3,"betas":(0.5,0.999)},
 	#                   torch.optim.lr_scheduler.CosineAnnealingLR, {}, Image_Pair_Loss),
    "Speech-onlyG":(get_speechcommand_dataloader,voiceG,{},train_one_batch_onlyG,
                   torch.optim.Adam,{"lr":2e-4,"betas":(0.5,0.999)},
                   torch.optim.lr_scheduler.CosineAnnealingLR, {}, Image_Pair_Loss),
    "Face-onlyG":(get_facescrub_dataloader,Generator,{},train_one_batch_onlyG,
                     torch.optim.Adam,{"lr":1e-3,"betas":(0.5,0.999)},
                 torch.optim.lr_scheduler.CosineAnnealingLR, {}, Image_Pair_Loss)
    })

## weight_group_table = (param_num, initial_func, param_min, param_max)
WEIGHT_GROUP_TABLE = OrderedDict({
    "normal":(1500_0000, nn.init.uniform_, {"a":-0.1, "b":0.1}),
    "ones": (20000, nn.init.ones_, {}),
    "zeros": (20000, nn.init.zeros_, {}),
    # "embedding": (1000_0000, nn.init.normal_, {"mean":0.0, "std":1.0}), # For Embedding
    })

def print_weight_group_info(weight_group):
    for key in weight_group.get_keys():
        w = weight_group.get(key)
        print(f'weight_group: {key: <12} size is {WEIGHT_GROUP_TABLE[key][0] / 10000:>8}w, \
        | min is {w.min().item():+.4f} | max is {w.max().item():+.4f} | mean is {w.mean().item():+.4f} |\
        {WEIGHT_GROUP_TABLE[key][1].__name__:<10} | {WEIGHT_GROUP_TABLE[key][2]}')
        # w[:10])

def print_task_model_instances_info(task_model_instances = None):
    print('TASK_MODEL_ENTRY SETTING is:')
    for key in TASK_MODEL_ENTRY:
        print(f'{key:<10} | {TASK_MODEL_ENTRY[key][4].__name__} | {TASK_MODEL_ENTRY[key][5]} \
        | {TASK_MODEL_ENTRY[key][6].__name__} | {TASK_MODEL_ENTRY[key][7]} | {TASK_MODEL_ENTRY[key][8]}')
    print('task_model_instances are:')
    for key in task_model_instances:
        TM = task_model_instances[key]
        print(f'{key:<10} | batch_size is {TM.batch_size} | batch_num is {TM.batch_num} | perm_key is {TM.perm_key} | Scheduler.T_max is {TM.T_max}')

# TODO: add voice save and if-else to decide which was trained
def save_all_iamge(task_model_instances = None,weight_group = None):
    for key in TASK_MODEL_ENTRY:
        if 'onlyG' in key:
            if 'Speech' in key:
                copy_param_val(task_model_instances[key].model, params = weight_group)
                task_model_instances[key].final_save_all_voice()
            elif 'imdb' in key:
                copy_param_val(task_model_instances[key].model, params = weight_group)
                task_model_instances[key].final_save_all_text()
            else:
                # model_state_dict = torch.load('./checkpoint/2000k_2k_2k_cifar10-ResNet18_0.1_Face-onlyG_0.001/Face-onlyG.pth')['net']
                # task_model_instances[key].model.load_state_dict(model_state_dict)
                copy_param_val(task_model_instances[key].model, params = weight_group)
                task_model_instances[key].final_save_all_image()
        else:
            continue

# TODO: epoch save
def save_epoch(task=None):
    if 'Speech' not in task.name:
        task.save_GAN_image()
    else:
        task.save_GAN_voice()

def get_trial_name(task_model_instances = None):
    info = ''
    for key in WEIGHT_GROUP_TABLE:
        info += '{}k_'.format(WEIGHT_GROUP_TABLE[key][0]//1000)
    for key in task_model_instances:
        info += '{}_{}_'.format(key, task_model_instances[key].origin_lr)
    return info[:-1]
    
def get_param_group(device = None, oral_init = False, model_oral = None, params_num = None):
    weight_group = WeightGroup()
    if oral_init is True:
        # Initial parameters of ResNet18 from pytorch
        param_num = 1500_0000 if params_num is None else params_num
        weight_pool = Variable(torch.zeros((param_num,)))
        model_test = model_oral if model_oral is not None else resnet18(num_classes=10)
        offset = 0
        temp = None 
        for name, param in model_test.named_parameters():
            lens = np.prod(param.shape)
            temp = param
            print(f'{name} min is {param.min().item()} max is {param.max().item()} mean is {param.mean().item()}')
            temp = temp.view(-1)
            weight_pool[offset:offset+lens] = temp
            offset += lens
        weight_pool = weight_pool.to(device)
        # weight_pool.requires_grad = True
        weight_group.set(key = 'oral', v = weight_pool)
    else:
        for key in WEIGHT_GROUP_TABLE:
            param_num, initial_func, param_kwargs = WEIGHT_GROUP_TABLE[key][:4]
            weight_pool = Variable(torch.zeros((param_num,))) # , requires_grad = True)
            initial_func(weight_pool, **param_kwargs)
            weight_pool = weight_pool.to(device)
            weight_group.set(key = key, v = weight_pool)

    # a_pool = Variable(torch.ones((params_num,)), requires_grad = True)
    # b_pool = Variable(torch.zeros((params_num,)), requires_grad = True)
    print_weight_group_info(weight_group)
    return weight_group

def print_banner(s):
    print("=*="*20 + f" {s} " + "=*="*20)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(idx = None, max_iter = None, **kwargs):
    print_bar_info = ''
    for key in task_model_instances:
        task_model_instances[key].train_idx += 1
        task_model_instances[key].model.train()
        try:
            inputs, targets = next(task_model_instances[key].train_dataloader_iter)
        except:
            task_model_instances[key].train_dataloader_iter = iter(task_model_instances[key].train_loader)
            inputs, targets = next(task_model_instances[key].train_dataloader_iter)

        train_loss = task_model_instances[key].train_one_batch(task_model_instances[key], inputs, targets,**kwargs)
        task_model_instances[key].train_loss += train_loss
        print_bar_info += '| %s Loss: %.3f |' % (key, task_model_instances[key].train_loss/(task_model_instances[key].train_idx+1))
    progress_bar(idx, max_iter, print_bar_info)

## TODO: Put this function into TASK MODEL ENTRY
## TODO: Models should be initialized (including buffers in DCNN), which can help reduce the memory during traing and testing.
def test(idx = None, weight_group = None, best_acc_sum_for_all_task = None, save = False, test_trainset_acc = False, resnet_interval =None,image_save_interval = None,args =None):
    ## idx == -1: Directly testing the acc of task_model_intances from the saved weight_group.pth
    cur_acc_sum = 0
    cur_acc_list = {k:0 for k in best_acc_sum_for_all_task[0]}
    for key in task_model_instances: 
        if ("GAN" in key or 'onlyG' in key) and (idx+1)% image_save_interval == 0:
            task_model_instances[key].train_loss = 0 
            task_model_instances[key].train_idx = 0
            #task_model_instances[key].name = f'{task_model_instances[key].name}_{args.filter_pruning}_{args.prune_ratio}'
            # SAVE EPOCH HAS BEEN MASKED
            # save_epoch(task_model_instances[key])
            continue
        if ("GAN" not in key and 'onlyG' not in key) and (idx+1) % resnet_interval == 0 :
            device = task_model_instances[key].device
            dataloader = task_model_instances[key].train_loader if test_trainset_acc else task_model_instances[key].test_loader 
            #model_state_dict = torch.load('./checkpoint/2000k_2k_2k_cifar10-ResNet18_0.1_Face-onlyG_0.001/cifar10-ResNet18.pth')['net']
            #task_model_instances[key].model.load_state_dict(model_state_dict)
            ## When to activate the test_fn for intance[key] ?
            copy_param_val(task_model_instances[key].model, params = weight_group)
            # print(task_model_instances['cifar10-ResNet18'].model.bn1.running_mean)
            #print(task_model_instances['cifar10-ResNet18'].model.state_dict()['linear.weight'])
            task_model_instances[key].train_loss = 0 
            task_model_instances[key].train_idx = 0
            #if idx >= 0:
            task_model_instances[key].model.eval()
            #else: ## TODO: other methods (e.g. Traing a model G to memorize the 10k buffer parameters of resnet)
                #task_model_instances[key].model.train() 
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.to(device), targets.to(device).long()
                    total += targets.size(0)
                    outputs = task_model_instances[key].model(inputs)
                    loss = task_model_instances[key].loss_fn(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    progress_bar(batch_idx, len(dataloader), '| Epoch: %s | Loss: %.3f | LR: %.4f | %s | Acc: %.3f%% |'
                                % (task_model_instances[key].test_epoch, test_loss/(batch_idx+1), task_model_instances[key].optimizer.state_dict()['param_groups'][0]['lr'], 
                                key, 100.*correct/total))
                cur_acc_list[key] = 100.*correct/total
                cur_acc_sum += cur_acc_list[key]
                task_model_instances[key].test_epoch += 1
                if idx >= 0:
                    task_model_instances[key].scheduler_step()

            if cur_acc_sum > best_acc_sum_for_all_task[1] and idx >= 0:
                with torch.no_grad():
                    subdir = f'./checkpoint/{get_trial_name(task_model_instances)}'
                    if not os.path.isdir(subdir):
                        os.mkdir(subdir)
                    for key in task_model_instances:
                        best_acc_sum_for_all_task[0][key] = cur_acc_list[key]
                        copy_param_val(task_model_instances[key].model, params = weight_group)
                        save_models(model = task_model_instances[key].model, acc = cur_acc_list[key], \
                                    idx = idx, subdir = subdir, device = task_model_instances[key].device, key = key, save = save)
                        best_acc_sum_for_all_task[1] = cur_acc_sum
                    weight_group.save(path = f'{subdir}/weight_group.pth')
                print(f'cur_best_acc_sum is {best_acc_sum_for_all_task[1]}: {best_acc_sum_for_all_task[0]}')

def get_task_model_instances(args = None) -> Dict[str, TaskModel]:
    task_model_instances = OrderedDict()
    for key in TASK_MODEL_ENTRY:
        dataloader_func, model, model_kwargs, train_one_batch_fn, optimizer_fn, optim_kwargs, scheduler_fn, _, loss_fn = TASK_MODEL_ENTRY[key]
        task_model_instances[key] = TaskModel(key, dataloader_func, model, model_kwargs, train_one_batch_fn, optimizer_fn, optim_kwargs, scheduler_fn, \
                                                loss_fn, torch.device(f'cuda:{args.which_cuda}'), T_max = args.max_iter // 391)
        task_model_instances[key].further_init(pair_size = args.pair_size,epoch_interval = args.save_img_epoch) ## For GAN
    print('Task_model_instances have been built.')
    return task_model_instances


def main(args = None):
    setup_seed(100)
    global task_model_instances
    device = torch.device(f'cuda:{args.which_cuda}')
    task_model_instances = get_task_model_instances(args)
    best_acc_sum_for_all_task = [dict({k:0 for k in task_model_instances}), 0] ## [acc for each task_model, acc_sum]
    if args.test is True:
        subdir = subdir = f'./checkpoint/{get_trial_name(task_model_instances)}' 
        weight_group_ = torch.load(f'{subdir}/weight_group_final.pth',map_location='cuda:0')
        weight_group_ = defense_to_weight_pool(task_model_instances, weight_group_, args)
        weight_group = WeightGroup()
        for key in weight_group_:
            weight_group.set(key, weight_group_[key].to(device))
        # 直接用cifa10原始的buffer放到里面去跑一下
        print(f'In the weight group, the param num is {weight_group.lens()}')
        get_buffer_param(task_model_instances, weight_group, use_test_dataloader=False)
        # test(-1, weight_group, best_acc_sum_for_all_task, args.save_model)
        test(-1, weight_group, best_acc_sum_for_all_task, args.save_model,resnet_interval = 391,image_save_interval= 1 ,args=args)
        #save_all_iamge(task_model_instances,weight_group)
        return
    else:
        weight_group = get_param_group(device, args.oral_init)
        print_task_model_instances_info(task_model_instances)
        for idx in range(args.max_iter):
            train(idx, args.max_iter, weight_group = weight_group, device = device)
            if (idx+1)%391==0 or (idx+1)%(args.pair_size*args.save_img_epoch)==0 : 
                test(idx, weight_group, best_acc_sum_for_all_task, args.save_model,resnet_interval = 391,image_save_interval=  args.pair_size*args.save_img_epoch,args=args)
        print_weight_group_info(weight_group)
        print_task_model_instances_info(task_model_instances)
        print('The acc of each models with the best acc sum:\n', best_acc_sum_for_all_task[0])
        subdir = f'./checkpoint/{get_trial_name(task_model_instances)}'
        weight_group.save(f'{subdir}/weight_group_final.pth')
        # save all images
        save_all_iamge(task_model_instances,weight_group)
        for key in task_model_instances:
            copy_param_val(task_model_instances[key].model, params = weight_group)
            save_models(model = task_model_instances[key].model, acc = None, idx = None, subdir = subdir, device = task_model_instances[key].device, key = key, save = True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model_overloading_main")
    parser.add_argument('--which_cuda', default=0, type=int, help='the index of cuda to use')
    parser.add_argument('--max_iter', default=78200, type=int, help='the training epoch max iter')
    parser.add_argument('--oral_init', action='store_true', help='whether use oral initialization')
    parser.add_argument('--save_model', action='store_true', help='whether to save each model')
    parser.add_argument('--test', action='store_true', help='whether to test the weight_group')
    parser.add_argument('--pair_size',default=1,type=int,help='the number of batches used to map in GAN')
    parser.add_argument('--save_img_epoch',default=10,type=int,help='the number of epoch to save generator-img')
    # defense
    parser.add_argument('--defense_method', default=None, type=str, help='the defense methods include: quant, prune, noise, finetune')
    parser.add_argument('--prune_ratio', default=0, type=float, help='the ratio of params pruning')
    parser.add_argument('--filter_pruning', action='store_true', help='prune the filter or weight (i.e. FP or WP)')
    parser.add_argument('--recover_for_pruning', action='store_true', help='whether to recover nonzeros for pruning)') 
    args = parser.parse_args()
    print(args)
    main(args)