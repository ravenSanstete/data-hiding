import os
from joblib import parallel_backend
from torch.autograd.grad_mode import F
from torch.serialization import save
import torchvision.utils as vutils
import numpy as np
import torch
from models.dcgan import discriminator,generator
from dataset import PairImage, PairVoice,PairText
import torchaudio
import matplotlib.pyplot as plt

class WeightGroup(object):
    ''' Privacy variable  __weight_group'''
    __weight_group = dict()
    @classmethod
    def get(self, key):
        return self.__weight_group[key]
    @classmethod
    def lens(self):
        return {key:self.__weight_group[key].shape[0] for key in self.__weight_group.keys()}
    @classmethod
    def get_keys(self):
        return self.__weight_group.keys()
    @classmethod
    def set(self, key, v):
        self.__weight_group[key] = v
    @classmethod
    def add(self, key, v):
        self.__weight_group[key] = self.__weight_group[key] + v
    @classmethod 
    def save(self, path):
        # for key in self.get_keys():
        torch.save(self.__weight_group, path)

def model_to_weight_group(model = None, weight_group_num = None, wp_start_pos = 0, return_dict = True, clip = True):
    weight_group= dict({'normal':[],'ones':[], 'zeros':[]})
    for name, module in model.named_modules():
        print(name)
        if not hasattr(module, 'weight') and (not hasattr(module, 'bias')):
            continue
        weight_key = 'normal'
        bias_key =  'normal'
        if type(module) in [torch.nn.modules.batchnorm.BatchNorm2d,torch.nn.modules.batchnorm.BatchNorm1d]:
            weight_key = 'ones'
            bias_key = 'zeros'
        if hasattr(module, 'weight_ih_l0') and type(module.bias) is bool: ## For RNN
            weight_group[weight_key].append(module.weight_ih_l0.data.view(-1).clone())
            weight_group[weight_key].append(module.weight_hh_l0.data.view(-1).clone())
            weight_group[bias_key].append(module.bias_ih_l0.data.view(-1).clone())
            weight_group[bias_key].append(module.bias_hh_l0.data.view(-1).clone())
        else:
            weight_group[weight_key].append(module.weight.data.view(-1).clone())
            if hasattr(module, 'bias') and module.bias is not None:
                weight_group[bias_key].append(module.bias.data.view(-1).clone())
    ## cat and clip
    ## TODO: randperm; 
    #  if the model is carrier, there is no need to randperm
    for key in weight_group:
        if clip:
            weight_group[key] = torch.cat(weight_group[key],dim=0)[0 :weight_group_num[key]]
        else:
            weight_group[key] = torch.cat(weight_group[key],dim=0) 
    if not return_dict:
        ## return as WeightGroup
        WP = WeightGroup()
        for key in weight_group:
            WP.set(key, weight_group[key])
    else:
        ## return as dict
        WP = weight_group
    return WP

def plot_wave(waveform,path):
    plt.figure()
    plt.plot(waveform.t().cpu().numpy())
    plt.savefig(path)
    plt.close()

class TaskModel():
    def __init__(self,key = None, dataloader_func = None,
                model = None, model_kwargs = None, train_one_batch_fn = None, optimizer_fn = None, optim_kwargs = None, 
                scheduler_fn = None, loss_fn = None, device = None,
                T_max = None, **kwargs):
        self.name = key
        self.device = device
        if dataloader_func:
            train_loader, test_loader = dataloader_func()
            self.train_loader = train_loader
            self.test_loader = test_loader
        else:
            self.train_loader = None
            self.test_loader = None
        self.model = model(**model_kwargs).to(device)
        self.optimizer = optimizer_fn(self.model.parameters(), **optim_kwargs)
        self.loss_fn = loss_fn
        ## To calculate the average train loss during training
        self.train_idx = 0
        self.train_loss = 0
        ## Use to print the info of test phase
        self.test_epoch = 0
        self.best_test_acc = 0
        if self.train_loader:
            self.batch_size = self.train_loader.batch_size
            self.batch_num = len(self.train_loader)
            self.train_dataloader_iter= iter(self.train_loader)
        else:
            self.batch_size = 32
        self.train_one_batch = train_one_batch_fn

        ## TODO: The scheduler dose not step simultanuously.
        # self.max_iter = max_iter
        self.T_max = T_max
        self.scheduler = scheduler_fn(self.optimizer, T_max = self.T_max)
        self.origin_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        self.perm_key = None ## TODO

        ## For GAN
        self.D = None
        self.optimizerD = None
        self.num_pair = 0
        self.fix_noise = None
        self.g_image_pair = None
        self.image_pair = None
        self.fake_image = None
        self.GAN_image_pair_loss_list = []
        self.pair_size = None
        self.idx = 0 # training iter index
        self.epoch_interval = None
        self.noise_dataset = None

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    ## For GAN
    def further_init(self, pair_size = 0,epoch_interval = 0):
        ''' # one-batch
        if 'GAN' in self.name or 'onlyG' in self.name:
            self.num_pair = pair_size*128
            self.fix_noise = torch.randn(self.num_pair,100,1,1,device=self.device)
            print(f'fix noise:{self.fix_noise.shape}')
            # matching pair (nosie,tarining-set image)
            for i,(data,_) in enumerate(self.train_loader):
                if i==0:
                    image_pair = data
                else:
                    image_pair = torch.cat((image_pair,data),dim=0)
                if i+1>=pair_size:
                    break
            self.image_pair = image_pair.to(self.device)
            vutils.save_image(self.image_pair[0:64],\
                f'real_image_pair_{self.num_pair}.png',normalize=True)
            print(f'image pair:{self.image_pair.shape}')
            self.D = discriminator().to(self.device)
            self.optimizerD = torch.optim.Adam(self.D.parameters(),lr=1e-3,betas=(0.5,0.999))
            print("GAN related generation complete.")
        '''
        '''
        # multiple-batch
        if 'GAN' in self.name or 'onlyG' in self.name:
            self.pair_size = pair_size
            self.num_pair = self.pair_size*self.batch_size
            fix_noise_list = []
            image_pair_list = []

            for i,(data,_) in enumerate(self.train_loader):
                noise = torch.randn(self.batch_size,100,1,1)
                fix_noise_list.append(noise)
                image_pair_list.append(data)
                if i+1>=pair_size:
                    break
            
            self.fix_noise = torch.stack(fix_noise_list).to(self.device)
            self.image_pair = torch.stack(image_pair_list).to(self.device)
            
            print(f'fix noise:{self.fix_noise.shape}')
            print(f'image pair:{self.image_pair.shape}')
            vutils.save_image(self.image_pair[0][0:self.batch_size],\
                f'real_image_pair_{self.num_pair}.png',normalize=True)

            self.D = discriminator().to(self.device)
            self.optimizerD = torch.optim.Adam(self.D.parameters(),lr=2e-4,betas=(0.5,0.999))
            print("GAN related generation complete.")
        '''
        # shuffle multiple-batch
        if 'GAN' in self.name or 'onlyG' in self.name:
            if 'Speech' in self.name:
                pair_size = 16
                print(f'Speech Pair_size {pair_size}')
                dataset = PairVoice(self,pair_size)
            elif 'Face' in self.name:
                pair_size = 2
                print(f'Face Pair_size {pair_size}')
                dataset = PairImage(self,pair_size)
            elif 'cifar' in self.name:    
                pair_size = 32
                print(f'Cifar10 Pair_size {pair_size}')
                dataset = PairImage(self,pair_size)
            else:
                print(f'imdb Pair_size {pair_size}')
                dataset = PairText(self,pair_size)
    

                
            self.noise_dataset = dataset
            print(f'fix noise:{self.fix_noise.shape}')
            print(f'pair:{self.image_pair.shape}')
            print("GAN related generation complete.")
            self.epoch_interval = epoch_interval
            self.D = discriminator().to(self.device)
            self.optimizerD = torch.optim.Adam(self.D.parameters(),lr=1e-3,betas=(0.5,0.999))
            # os:memory allocate error -> num_workers may need to be set to zero
            self.train_loader = torch.utils.data.DataLoader(self.noise_dataset, batch_size=self.batch_size, shuffle=True, num_workers = 0)
            self.train_dataloader_iter = iter(self.train_loader)
            # 保存初始选定的图像/音频
            # if 'Speech' not in self.name:
            #     vutils.save_image(self.image_pair[0:self.batch_size],\
            #         f'real_image_pair_{self.name}_{self.num_pair}.png',normalize=True)
            # else:
            #     plot_wave(self.image_pair[0],f'real_voice_{self.name}_{self.num_pair}.png')

    def save_GAN_image(self):
        path_1 =  f'./g_pair_img/{self.name}/{self.num_pair}'
        path_2 =  f'./g_pair_img_single/{self.name}/{self.num_pair}'

        if not os.path.exists(path_1):
            os.makedirs(path_1)
        if not os.path.exists(path_2):
            os.makedirs(path_2)

        if not 'onlyG' in self.name:
            vutils.save_image(self.fake_image[0:self.batch_size],f'g_fake_img/{self.name}_{self.test_epoch}_{self.num_pair}.png',normalize=True)
        vutils.save_image(self.g_image_pair[0:self.batch_size],path_1+f'/{self.test_epoch}.png',normalize=True)
        vutils.save_image(self.g_image_pair[0],path_2+f'/{self.test_epoch}.png',normalize=True)
        self.test_epoch += self.epoch_interval

    #TODO: save generated 1 voice waveform plot
    def save_GAN_voice(self):
        path_1 =  f'./g_pair_waveform/{self.name}/{self.num_pair}'

        if not os.path.exists(path_1):
            os.makedirs(path_1)

        path = path_1+f'/{self.test_epoch}.png'    
        waveform = self.g_image_pair[0].detach()
        plot_wave(waveform,path)
        self.test_epoch += self.epoch_interval


    def final_save_all_image(self):
        model = self.model
        model.eval()
        idx = 1
        path_1 = f'./generated_all_image/{self.name}/{self.num_pair}'
        path_2 = f'./paired_all_image/{self.name}/{self.num_pair}'

        if not os.path.exists(path_1):
            os.makedirs(path_1)
        if not os.path.exists(path_2):
            os.makedirs(path_2)
        self.train_loader = torch.utils.data.DataLoader(self.noise_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 0)
        for i,(noise,img) in enumerate(self.train_loader):
            noise = noise.to(self.device)
            pair_img = img.to(self.device)
            generated_image = model(noise)
            for j in range(self.batch_size):
                vutils.save_image(generated_image[j],path_1+f'/{idx}.png',normalize=True)
                vutils.save_image(pair_img[j],path_2+f'/{idx}.png',normalize=True)
                idx += 1
        
        #np.save(f'pair_lost_list/{self.name}_{self.num_pair}.npy',np.array(self.GAN_image_pair_loss_list))
        print('all image save compelete')

    #TODO: save all voice and all waveform
    def final_save_all_voice(self):
        model = self.model
        model.eval()
        idx = 1
        path_1 = f'./generated_all_voice/{self.name}/{self.num_pair}'
        path_2 = f'./paired_all_voice/{self.name}/{self.num_pair}'
        path_3 = f'./generated_all_waveform/{self.name}/{self.num_pair}'
        path_4 = f'./paired_all_waveform/{self.name}/{self.num_pair}'

        if not os.path.exists(path_1):
            os.makedirs(path_1)
        if not os.path.exists(path_2):
            os.makedirs(path_2)
        if not os.path.exists(path_3):
            os.makedirs(path_3)
        if not os.path.exists(path_4):
            os.makedirs(path_4)
        for i,(noise,voice) in enumerate(self.train_loader):
            noise = noise.to(self.device)
            pair_voice = voice.to(self.device)
            generated_voice = model(noise)
            for j in range(self.batch_size):
                torchaudio.save(path_1+f'/{idx}.wav',generated_voice[j].cpu(),16000)
                torchaudio.save(path_2+f'/{idx}.wav',pair_voice[j].cpu(),16000)
                #plot_wave(generated_voice[j].detach(),path_3+f'/{idx}.png')
                #plot_wave(pair_voice[j].detach(),path_4+f'/{idx}.png')
                idx += 1
        
        #np.save(f'pair_lost_list/{self.name}_{self.num_pair}.npy',np.array(self.GAN_image_pair_loss_list))
        print('all voice save compelete')

    def final_save_all_text(self):
        model = self.model
        model.eval()
        idx = 1
        path_1 = f'./generated_all_text/{self.name}/{self.num_pair}'

        if not os.path.exists(path_1):
            os.makedirs(path_1)

        generated_list = []
        for i,(noise,voice) in enumerate(self.train_loader):
            noise = noise.to(self.device)
            generated_text = model(noise)
            for j in range(self.batch_size):
                for k in range(224):
                    embed = generated_text[j][0][k][0:200].detach().cpu()
                    generated_list.append(embed)
                idx += 1
        
        torch.save(generated_list,f'{path_1}/generated_all_embedding.pth')
        print('all text save compelete')


if __name__ == '__main__':
    pass