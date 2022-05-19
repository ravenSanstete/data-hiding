# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim
# from torchtext import data
# from torchtext import datasets
# from torchtext.data import Dataset
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import Dataset, DataLoader,random_split
import torchvision.utils as vutils

# import torchaudio
# from torchaudio.datasets import SPEECHCOMMANDS

# import time
# import random
# from models import *
# import tqdm
# import os
# import dill

import medmnist # https://github.com/MedMNIST/MedMNIST
from medmnist.dataset import DermaMNIST


import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets
from torchtext.data import Dataset

import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

#import matplotlib.pyplot as plt

import time
import random
from models import *
import os
import dill

from models.vgg import vgg11
from models.m5 import M5
from models.lenet import LeNet
from models.textcnn import TextCNN

# https://zhuanlan.zhihu.com/p/64934558

root = './data/text'

class IMDBDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)


def dump_examples(train, test, suffix = None):
    # save the examples
    train_path, test_path = os.path.join(root, suffix+'_train'), os.path.join(root, suffix+'_test')
    if not os.path.exists(train_path):
        with open(train_path, 'wb')as f:
            dill.dump(train.examples, f)
    if not os.path.exists(test_path):
        with open(test_path, 'wb')as f:
            dill.dump(test.examples, f)

def load_split_datasets(fields, suffix = None):
    # load the examples
    train_path, test_path = os.path.join(root, suffix+'_train'), os.path.join(root, suffix+'_test')
    with open(train_path, 'rb')as f:
        train_examples = dill.load(f)
    with open(test_path, 'rb')as f:
        test_examples = dill.load(f)

    # 恢复数据集
    train = IMDBDataset(examples=train_examples, fields=fields)
    test = IMDBDataset(examples=test_examples, fields=fields)
    return train, test

class IMDB():
    def  __init__(self, device = None):
        super(IMDB, self).__init__()
        self.dataset_name = 'IMDB'
        self.batch_size = 128
        self.vocabulary_size = 20000
        self.random_seed = 123
        # torch.manual_seed(self.random_seed)
    
    def get_dataloader(self):
        TEXT = data.Field(tokenize='spacy',tokenizer_language="en_core_web_sm") # include_lengths=True) # necessary for packed_padded_sequence
        LABEL = data.LabelField(dtype=torch.float)
        text_vocab_path = os.path.join(root, self.dataset_name+'_text_vocab')
        label_vocab_path = os.path.join(root, self.dataset_name+'_label_vocab')

        if os.path.exists(os.path.join(root, self.dataset_name +'_train')) and os.path.exists(text_vocab_path):
            print('load the examples...')
            fields = {'text': TEXT, 'label': LABEL}
            train_data, test_data = load_split_datasets(fields = fields, suffix = self.dataset_name)
            with open(text_vocab_path, 'rb')as f:
                TEXT.vocab = dill.load(f)
            with open(label_vocab_path, 'rb')as f:
                LABEL.vocab = dill.load(f)
        else:
            print('generate the examples...')
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            dump_examples(train_data, test_data, suffix = self.dataset_name)
            TEXT.build_vocab(train_data, max_size=self.vocabulary_size)
            LABEL.build_vocab(train_data)
            with open(text_vocab_path, 'wb')as f:
                dill.dump(TEXT.vocab, f)
            with open(label_vocab_path, 'wb')as f:
                dill.dump(LABEL.vocab, f)

        print(f'Num Train: {len(train_data)}')
        print(f'Num Test: {len(test_data)}')

        print(f'Vocabulary size: {len(TEXT.vocab)}')
        print(f'Number of classes: {len(LABEL.vocab)}')
        
        train_loader, test_loader = data.BucketIterator.splits(
            (train_data, test_data), 
            batch_size=self.batch_size,
            sort_within_batch=True, # necessary for packed_padded_sequence
            )
        return train_loader, test_loader

## The dataset SPEECHCOMMANDS is a torch.utils.data.Dataset version of the dataset.
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    # [Bacth_size, s, hz] -> [Bacth_size, pad(hz), channels] -> [Bacth_size, channels, hz], channels.value is between (-1, 1)
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def get_SC_dataloader(batch_size = 64, new_sample_rate = None):
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    def collate_fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
        tensors, targets = [], []
        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            ## The Sample fRate is 16000 before transform.
            ## The operation of transform is much slower in CPU the GPU.
            if new_sample_rate is not None:
                print('new sample rate')
                transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
                tensors += [transform(waveform)]
            else:
                tensors += [waveform]
            targets += [torch.tensor(labels.index(label))]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    ## not CUDA
    num_workers = 2
    pin_memory = False
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader

## export 
def get_mnist_dataloader(batch_size = 32, **kwargs):
    img_size = 32
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] , [0.5])])
    #Defining the transforms to applied on the image ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_cifar10_dataloader(batch_size = 64, **kwargs):
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
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0) # 2
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0) # 2
    return train_loader, test_loader

def get_cifar10_dataloader_gan():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False, num_workers=2)
    return train_loader,None

'''
data_loader,_ = get_cifar10_dataloader_gan()
for i,(data,_) in enumerate(data_loader):
    dataset = data
    break
vutils.save_image(dataset[0:64],f'cifar_test.png',normalize=True)
'''

def get_dermamnist_dataloader(BATCH_SIZE=100, data_size = 32): # 28
    BATCH_SIZE=16
    transform = transforms.Compose([
        transforms.Resize(data_size),
        transforms.CenterCrop(data_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = DermaMNIST('data', split='train', transform=transform) # transform is only applied to images, not labels
    val_dataset = DermaMNIST('data', split='val', transform=transform)
    test_dataset = DermaMNIST('data', split='test', transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader

def get_imdb_dataloader(**kwargs):
    imdb = IMDB()
    train_loader, test_loader = imdb.get_dataloader()
    return train_loader, test_loader

def get_speechcommand_dataloader(**kwargs):
    return get_SC_dataloader(**kwargs)

def get_gtsrb_dataloader(batch_size=24, **kwargs):
    gtsrb_data_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
    ])

    train_data_path = "./data/gtsrb/Train"
    train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform = gtsrb_data_transforms)

    # Divide data into training and validation (0.8 and 0.2)
    ratio = 0.8
    n_train_examples = int(len(train_data) * ratio)
    n_val_examples = len(train_data) - n_train_examples

    train_data, val_data = random_split(train_data, [n_train_examples, n_val_examples])
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size = batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size = batch_size)

    return train_loader,val_loader



PREFIX = '/home/myang_20210409/data'
def get_imagenet_dataloader(batch_size = 32):
    #datadir = os.path.join(PREFIX + "/imagenet/", 'val')
    datadir = os.path.join(PREFIX + "/imagenette2/", 'train')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform_imagenet = transforms.Compose([
                       transforms.RandomResizedCrop(224),
                       transforms.ToTensor(),
                       normalize,])
    # data_shape = [3, 224, 224]
    dataset = torchvision.datasets.ImageFolder(datadir,transform=transform_imagenet)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                num_workers=1,pin_memory=True,drop_last=True)
    return data_loader,None
'''
data_loader,_ = get_imagenet_dataloader()
print(data_loader.batch_size)
for i,(data,label) in enumerate(data_loader):
    print(data.shape)
    break
'''

def get_facescrub_dataloader(batch_size = 64, width = 224):
    datadir = PREFIX + "/facescrub/20face/"
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    dataset = torchvision.datasets.ImageFolder(datadir,
        transforms.Compose([
                       transforms.Resize(width),
                       transforms.ToTensor(),
                       normalize,
                   ]))

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                   num_workers=2,pin_memory=True,drop_last=True)

    return loader,None

class PairImage(torch.utils.data.Dataset):
    def __init__(self,taskModel,pair_size):
        super(PairImage, self).__init__()
        taskModel.pair_size = pair_size
        taskModel.num_pair = pair_size*taskModel.batch_size
        self.num_pair = taskModel.num_pair
        taskModel.fix_noise = torch.randn(taskModel.num_pair,100,1,1)
        self.fix_noise = taskModel.fix_noise
        # matching pair (nosie,tarining-set image) 
        for i,(data,_) in enumerate(taskModel.train_loader):
            if i==0:
                image_pair = data
            else:
                image_pair = torch.cat((image_pair,data),dim=0)
            if i+1>=pair_size:
                break
        taskModel.image_pair = image_pair
        self.image_pair = taskModel.image_pair

    def __getitem__(self, index):
        return self.fix_noise[index],self.image_pair[index]
    
    def __len__(self):
        return self.num_pair

class PairVoice(torch.utils.data.Dataset):
    def __init__(self,taskModel,pair_size):
        super(PairVoice, self).__init__()
        taskModel.pair_size = pair_size
        taskModel.num_pair = pair_size*taskModel.batch_size
        self.num_pair = taskModel.num_pair
        taskModel.fix_noise = torch.randn(taskModel.num_pair,100,1)
        self.fix_noise = taskModel.fix_noise
        # matching pair (nosie,tarining-set image)
        for i,(data,_) in enumerate(taskModel.train_loader):
            if i==0:
                voice_pair = data
            else:
                voice_pair = torch.cat((voice_pair,data),dim=0)
            if i+1>=pair_size:
                break
        taskModel.image_pair = voice_pair
        self.voice_pair = taskModel.image_pair

    def __getitem__(self, index):
        return self.fix_noise[index],self.voice_pair[index]
    
    def __len__(self):
        return self.num_pair

class PairText(torch.utils.data.Dataset):
    def __init__(self,taskModel,pair_size):
        super(PairText, self).__init__()
        self.embedding = torch.load('dataset_embed.pth')
        print(len(self.embedding))
        taskModel.pair_size = pair_size
        taskModel.num_pair = pair_size*taskModel.batch_size
        taskModel.batch_num = pair_size
        self.num_pair = taskModel.num_pair
        taskModel.fix_noise = torch.randn(taskModel.num_pair,100,1,1)
        self.fix_noise = taskModel.fix_noise
        # matching pair (nosie,tarining-set image) 
        embedding_list = []
        for i in range(self.num_pair):
            new_embedding = torch.randn(1,224,224)
            for j in range(224):
                new_embedding[0][j][0:200] = self.embedding[i*224+j][2] # substitude
            embedding_list.append(new_embedding.unsqueeze(0))
        image_pair = torch.cat(embedding_list,dim=0)
        taskModel.image_pair = image_pair
        self.image_pair = taskModel.image_pair

    def __getitem__(self, index):
        return self.fix_noise[index],self.image_pair[index]
    
    def __len__(self):
        return self.num_pair