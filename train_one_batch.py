from models.nnfunc import *

## For classification
def train_one_batch_for_clf(task_model = None, inputs = None, targets = None, weight_group = None, device = None):
    model = task_model.model
    optimizer = task_model.optimizer
    loss_fn = task_model.loss_fn
    inputs, targets = inputs.to(device), targets.to(device).long()
    model.zero_grad()
    copy_param_val(model, params = weight_group)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    param_delta = dict()
    local_param_delta = copy_param_delta(model, params = weight_group)
    param_delta = accumulate_param_delta(param_delta, local_param_delta)
    weight_group = upadte_weight_group(weight_group, param_delta)
    train_loss = loss.item() ## .item(): tensor -> float
    _, predicted = outputs.max(1)
    return train_loss

## For GAN and generator
def Image_Pair_Loss(real_images,g_iamges):
    num_pair = real_images.shape[0]
    loss  = torch.mean(torch.norm(torch.sub(real_images,g_iamges).reshape(num_pair,-1),dim=1))
    return loss

def train_one_batch_onlyG(task_model = None, inputs = None, targets = None, weight_group = None, device = None):
    '''
    model = task_model.model
    optimizer = task_model.optimizer
    loss_fn = task_model.loss_fn
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    copy_param_val(model, params = weight_group)
    task_model.g_image_pair = model(task_model.fix_noise)
    image_pair_loss = loss_fn(task_model.g_image_pair, task_model.image_pair)
    task_model.GAN_image_pair_loss_list.append(image_pair_loss.item())
    image_pair_loss.backward()
    optimizer.step()
    param_delta = dict()
    local_param_delta = copy_param_delta(model, params = weight_group)
    param_delta = accumulate_param_delta(param_delta, local_param_delta)
    weight_group = upadte_weight_group(weight_group, param_delta)
    image_pair_loss = image_pair_loss.item()
    return image_pair_loss
    '''
    '''
    # multi-batches
    model = task_model.model
    optimizer = task_model.optimizer
    loss_fn = task_model.loss_fn
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    copy_param_val(model, params = weight_group)
    batch_idx = task_model.idx%task_model.pair_size
    task_model.g_image_pair = model(task_model.fix_noise[batch_idx])
    image_pair_loss = loss_fn(task_model.g_image_pair, task_model.image_pair[batch_idx])
    task_model.GAN_image_pair_loss_list.append(image_pair_loss.item())
    image_pair_loss.backward()
    optimizer.step()
    param_delta = dict()
    local_param_delta = copy_param_delta(model, params = weight_group)
    param_delta = accumulate_param_delta(param_delta, local_param_delta)
    weight_group = upadte_weight_group(weight_group, param_delta)
    image_pair_loss = image_pair_loss.item()
    task_model.idx +=1 
    # for plot : generating identical mapping
    task_model.g_image_pair = model(task_model.fix_noise[0])
    return image_pair_loss
    '''
    # shuffle -  multi-batches 
    model = task_model.model
    optimizer = task_model.optimizer
    loss_fn = task_model.loss_fn
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    copy_param_val(model, params = weight_group)
    # print(torch.sum(inputs))
    task_model.g_image_pair = model(inputs.to(device))
    image_pair_loss = loss_fn(task_model.g_image_pair, targets.to(device))
    task_model.GAN_image_pair_loss_list.append(image_pair_loss.item())
    image_pair_loss.backward()
    optimizer.step()
    param_delta = dict()
    local_param_delta = copy_param_delta(model, params = weight_group)
    param_delta = accumulate_param_delta(param_delta, local_param_delta)
    weight_group = upadte_weight_group(weight_group, param_delta)
    image_pair_loss = image_pair_loss.item()

    # for plot : generating identical mapping
    task_model.g_image_pair = model(task_model.fix_noise[0:task_model.batch_size].to(device))
    return image_pair_loss

def train_one_batch_gan(task_model = None, inputs = None, targets = None, weight_group = None, device = None):
    G = task_model.model
    D = task_model.D
    optimizer = task_model.optimizer
    optimizerD = task_model.optimizerD
    loss_fn = task_model.loss_fn
    G.train()
    D.train()
    real_image = inputs.to(device)
    noise = torch.randn(real_image.shape[0],100,1,1,device=device) 
    copy_param_val(G, params = weight_group)
    task_model.fake_image = G(noise)
    optimizer.zero_grad()
    optimizerD.zero_grad()
    fake_output = D(task_model.fake_image.detach()) # backward process stoped here
    real_output = D(real_image)
    real_label = torch.ones(real_output.shape[0],device=device)
    fake_label = torch.zeros(fake_output.shape[0],device=device)
    d_loss = loss_fn(fake_output,fake_label) + loss_fn(real_output,real_label)
    d_loss.backward()
    optimizerD.step()
    # train generator
    G.zero_grad()
    optimizer.zero_grad()
    fake_output = D(task_model.fake_image)
    g_loss = loss_fn(fake_output,real_label)
    task_model.g_image_pair = G(task_model.fix_noise)
    image_pair_loss = Image_Pair_Loss(task_model.image_pair,task_model.g_image_pair)
    task_model.GAN_image_pair_loss_list.append(image_pair_loss.item())
    g_loss_total = g_loss + image_pair_loss
    g_loss_total.backward()
    optimizer.step()
    param_delta = dict()
    local_param_delta = copy_param_delta(G, params = weight_group)
    param_delta = accumulate_param_delta(param_delta, local_param_delta)
    weight_group = upadte_weight_group(weight_group, param_delta)
    image_pair_loss = image_pair_loss.item()
    g_loss = g_loss.item()
    d_loss = d_loss.item()
    return image_pair_loss


def get_buffer_param(task_model_instance, weight_group, use_test_dataloader = False):
    for key in task_model_instance:
        model = task_model_instance[key].model
        dataloader = task_model_instance[key].test_loader if use_test_dataloader else task_model_instance[key].train_loader 
        copy_param_val(model, params = weight_group)
        model = model.to(task_model_instance[key].device)
        ## The model initialized with weight group has default buffers.
        print('________________________{} buffer params from {} dataloader________________________'.format(key, 'test' if use_test_dataloader else 'train'))
        model.train()
        for x, _ in dataloader:
            _ = model(x.to(task_model_instance[key].device))
        model.eval()
        task_model_instance[key].model = model
    # except:
    #     key = task_model_instance.name
    #     if 'ones' not in task_model_instance.param_num:
    #         print('________________________{} No buffer params________________________'.format(key))
    #         return
    #     model = task_model_instance.model
    #     dataloader = task_model_instance.test_loader if use_test_dataloader else task_model_instance.train_loader 
    #     copy_param_val(model, params = weight_group, rand_perm=task_model_instance.rand_perm, wp_start_pos = task_model_instance.wp_start_pos)
    #     model = model.to(task_model_instance.device)
    #     ## The model initialized with weight group has default buffers.
    #     print('________________________{} buffer params from {} dataloader________________________'.format(key, 'test' if use_test_dataloader else 'train'))
    #     model.train()
    #     for x, _ in dataloader:
    #         _ = model(x.to(task_model_instance.device))
    #     model.eval()