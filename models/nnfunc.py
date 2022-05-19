from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## util for param_group
def is_bn_weight(name):
    return (name.find('bn') != -1 and name.find('weight') != -1) or name.find('shortcut.1.weight') != -1

def is_bn_bias(name):
    return (name.find('bn') != -1 and name.find('bias') != -1) or name.find('shortcut.1.bias') != -1

def param_name_to_key(name):
    if is_bn_weight(name):
        return 'ones'
    elif is_bn_bias(name):
        return 'zeros'
    # elif 'embedding' in name:
    #     return 'embedding'
    else: ## for normal params
        return 'normal'

def get_params(params_origin, param_size, offset, a = None, b = None):
    params = params_origin.clone() ## !!!!!!!!!!!
    new_params = []
    repete_count = (param_size + offset) // len(params)
    if(repete_count == 0):
        return params[offset:param_size + offset], param_size + offset
    
    new_params.append(params[offset:])
    repete_count -= 1
    rr = repete_count
    for i in range(repete_count):
        if a == None and b == None:
            new_params.append(params)
        else:
            new_params.append(params*a[i]+b[i]) # 
    offset = param_size + offset - (repete_count+1) * len(params)  
    new_params.append(params[:offset])
    return torch.cat(new_params), offset
    
def copy_param_val(model, params, **kwargs):
    # torch.manual_seed(params.shape[0])
    offset = dict({k:0 for k in params.get_keys()})
    for name, param in model.named_parameters():
        key = 'oral' if 'oral' in params.get_keys() else param_name_to_key(name)
        param_val, offset[key] = get_params(params.get(key), np.prod(param.shape), offset[key], **kwargs)
        param.data = param_val.reshape(param.shape).data

## TODO: group, the length of params_delta
def _copy_param_delta(net_param, params, offset):
    params_delta = torch.zeros_like(params)
    net_param = net_param.view(-1)
    param_size = net_param.shape[0]
    repete_count = (param_size + offset) // len(params)
    if(repete_count == 0):
        params_delta[offset:param_size + offset] = net_param - params[offset:param_size+offset]
        offset += param_size
        return params_delta, offset
    cur = 0
    params_delta[offset:] = net_param[cur:cur+len(params) - offset]-params[offset:]
    repete_count -= 1
    cur += len(params) - offset
    for i in range(repete_count):
        params_delta += net_param[cur:cur+len(params)] - params
        cur += len(params)
    offset = param_size + offset - (repete_count + 1) * len(params)
    params_delta[:offset] += net_param[cur:] - params[:offset]
    return params_delta, offset

def copy_param_delta(model, params, use_bn = False, **kwargs):
    params_delta = dict({k: torch.zeros_like(params.get(k)) for k in params.get_keys()})
    offset = dict({k: 0 for k in params.get_keys()})
    for name, param in model.named_parameters():
        key = 'oral' if 'oral' in params.get_keys() else param_name_to_key(name)
        local_param_delta, offset[key] = _copy_param_delta(param.data, params.get(key), offset[key])
        # print(name)
        # print(param.view(-1)[:10])
        # print(local_param_delta[:10])
        params_delta[key] += local_param_delta
    return params_delta

def accumulate_param_delta(model_delta, local_model_delta, alpha_group = None):
    # TODO: alpha = lr?
    if alpha_group is None:
        alpha_group = dict()
        for key in local_model_delta:
            alpha_group[key] = torch.ones(1)
    for key in local_model_delta:
        if not (key in model_delta):
            model_delta[key] = local_model_delta[key] 
        else: 
            model_delta[key] += local_model_delta[key] 
    return model_delta

def upadte_weight_group(weight_group, param_delta):
    for key in param_delta: ## len(param_delta) may less than len(weight_group)
        weight_group.add(key, param_delta[key])
    return weight_group

import os
def save_models(model = None, acc = None, idx = None, subdir = None, device = None, key = None, save = False):
    if not save:
        return
    print(f'Saving {key} with the largest acc sum ......')
    # save *.pth
    state = {
        'net': model.cpu().state_dict(), ## if net is still in the GPU, weight_group will be saved at the same time.
        # 'acc': acc,
        # 'idx': idx, 
    }
    torch.save(state, f'{subdir}/{key}.pth')
    model.to(device)
    
class NNWrapper(nn.Module):
    def __init__(self, baseObject):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__

    def forward(self, *input, params, **kwargs):
        copy_param_val(self, params)
        self.forward(self, *input, **kwargs)
        
class NNFunc():
    def __init__(self):
        self.state_dict = None
        pass

    def __call__(self, x, params):
        pass

    def total_param(self):
        count = np.sum([np.prod(self.state_dict[key]) for key in self.state_dict])
        return count
    
    def instantiate(self, params):
        state_dict_instance = OrderedDict()
        offset = 0
        for key in self.state_dict:
            param_size = np.prod(self.state_dict[key])
            state_dict_instance[key], offset  = get_params(params, param_size, offset)
            state_dict_instance[key] = state_dict_instance[key].view(self.state_dict[key])
            
        return state_dict_instance 
