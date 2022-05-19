import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import prune
import tqdm

from models.nnfunc import copy_param_val
from TaskModel import WeightGroup,model_to_weight_group
#from utils import progress_bar_local, print_bar_train_info
from train_one_batch import get_buffer_param



## For pruning
def prune_tensors(tensors, prune_ratio = None):
    weights=np.abs(tensors.cpu().numpy())
    weightshape=weights.shape
    rankedweights=weights.reshape(weights.size).argsort()
    
    num = weights.size
    prune_num = int(np.round(num*prune_ratio))
    count=0
    masks = np.zeros_like(rankedweights)
    for n, rankedweight in enumerate(rankedweights):
        if rankedweight > prune_num:
            masks[n]=1
        else: count+=1
    print("total weights:", num)
    print("weights pruned:",count)
    masks=masks.reshape(weightshape)
    weights=masks*weights
    return torch.from_numpy(weights).to(dtype=torch.float32), masks

def prune_weight_pool(weight_group = None, keys=['normal'], prune_ratio = 0.2):
    print('Pruning the whole weight_pool {}.'.format(keys))
    difference_norm = 0
    for key in keys:
        weight_group_clone = weight_group[key].clone()
        weight_group[key], _ = prune_tensors(weight_group[key], prune_ratio = prune_ratio)
        difference_norm += torch.norm(weight_group[key] - weight_group_clone)
    print('After pruning, the diffetence_norm is {}'.format(difference_norm))
    return weight_group

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
    num_zeros = 0
    num_elements = 0
    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
    sparsity = num_zeros / num_elements
    return num_zeros, num_elements, sparsity

def measure_global_sparsity(model, weight=True, bias=False,
                            conv2d_use_mask=False, linear_use_mask=False):
    num_zeros = 0
    num_elements = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements
        elif isinstance(module, torch.nn.Linear):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements
    sparsity = num_zeros / num_elements
    return num_zeros, num_elements, sparsity

def remove_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
    return model

def prune_carrier(task_model = None, weight_group_ = None, prune_ratio = 0.2, filter_pruning = False, recover_for_pruning = False):
    print('Pruning the carrier model {}'.format(task_model.name))
    ## weight pool to carrier
    model = task_model.model.to(task_model.device)
    weight_group = WeightGroup()
    for key in weight_group_:
        weight_group.set(key, weight_group_[key])
    copy_param_val(model, params = weight_group)
    for name, module in model.named_modules():
        # 需要反过来 weight purning和filter purning写反了
        if filter_pruning is True and isinstance(module, nn.Conv2d): ## WP or isinstance(module, nn.Linear))
            print(f'prune the module of {name}')
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
        if filter_pruning is False and (isinstance(module, nn.Conv2d)): ## FP
            prune.ln_structured(module, name='weight', amount=prune_ratio, dim=0, n=1) # Ln-norm
    remove_parameters(model)
    num_zeros, num_elements, sparsity = measure_global_sparsity(model)
    print('num_zeros %d \t num_elements %d \t sparsity %.3f' % (num_zeros, num_elements, sparsity))

    ## carrier to weight pool
    print('After purning, the weight pool is constrcuted from the carrier.')
    if recover_for_pruning:
        weight_group = wp_recover_from_redundancy(model, weight_group_)
    else:
        weight_group = model_to_weight_group(model,{key: weight_group_[key].shape[0] for key in weight_group_})

    for key in weight_group:
        print(f'{key}.len is {weight_group[key].shape[0]}')
    
    difference_norm = 0
    for key in weight_group:
        lens = min(weight_group[key].shape[0], weight_group_[key].shape[0])
        difference_norm += torch.norm(weight_group[key][:lens] - weight_group_[key][:lens])
    print('After {} pruning, the diffetence_norm is {}'.format('filter' if filter_pruning else 'weight', difference_norm))
    return weight_group

def wp_recover_from_redundancy(model = None, weight_group_ = None):
    ## TODO：wp_start_pos
    ## Only for simple model, e.g. resnet
    weight_group = model_to_weight_group(model,{key: weight_group_[key].shape[0] for key in weight_group_}, clip = False)

    print('Origin wp from pth')
    ori_wp_num = {key: weight_group_[key].shape[0] for key in weight_group_}
    print(ori_wp_num)
    print('The model param after pruning')
    model_wp_num = {key: weight_group[key].shape[0] for key in weight_group}
    print(model_wp_num)

    for key in weight_group_:
        zero_num = torch.sum(weight_group_[key] == 0).item()
        print(f'The origin wp {key} has {zero_num} zeros.')
    prune_zero_num = 0
    for key in weight_group:
        zero_num = torch.sum(weight_group[key][:ori_wp_num[key]] == 0).item() 
        print(f'The wp from pruned model param {key} has {zero_num} zeros.')
        prune_zero_num += zero_num 
    
    print('*************** Let\'s recover it from nonzeros. *******************')
    for key in weight_group:
        padding_zeros = torch.zeros(ori_wp_num[key] - model_wp_num[key] % ori_wp_num[key]).to('cuda:0')
        weight_group[key] = torch.cat([weight_group[key], padding_zeros], dim=0)
        weight_group[key] = weight_group[key].view(-1, ori_wp_num[key]) 
        nonzero_nums = torch.sum(weight_group[key] != 0, dim=0) 

        temp = torch.nan_to_num(torch.div(torch.sum(weight_group[key],dim=0), nonzero_nums))
        # diff = torch.norm(torch.sum(weight_group[key],dim=0)-torch.mul(temp, nonzero_nums))
        # print(f'local diff is {diff}')
        weight_group[key] =  temp 
    
    recover_zero_num = 0
    for key in weight_group:
        zero_num = torch.sum(weight_group[key] == 0).item() 
        print(f'The wp {key} from pruned model after recovering has {zero_num} zeros.')
        recover_zero_num += zero_num
    print(f'*************** Recover {prune_zero_num - recover_zero_num} nonzeros. *******************')
    return weight_group

def defense_to_weight_pool(task_model_instances = None, weight_group_ = None, args = None):
    ## weight_group_: saved weight group
    defense_method = args.defense_method
    if defense_method is None:
        return weight_group_
    print(f'____________________Test with defense method {defense_method}____________________')
        # weight_group_ = add_noise_to_carrier(weight_group_, carrier_param_num = task_model_instances["cifar10-ResNet18"].param_num)
    if defense_method == 'prune':
        weight_group_ = prune_carrier(task_model_instances['cifar10-ResNet18'], weight_group_, args.prune_ratio, args.filter_pruning, args.recover_for_pruning)
        # weight_group_ = prune_weight_pool(weight_group_, args.prune_ratio)
    else:
        print(f'{args.defense_method} does not exist, only support to one of (quant, prune, noise, finetune)')
        exit()
    return weight_group_