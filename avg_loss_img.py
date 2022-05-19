import torch
import torchvision.transforms as transforms
import argparse
import numpy as np
import os
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--key',default='Face-onlyG',type=str,help='name')
parser.add_argument('--pair_num',default=128,type=int,help='total num of the noise-image pair')
parser.add_argument('--prune_ratio', default=0, type=float, help='the ratio of params pruning')
parser.add_argument('--filter_pruning', action='store_true', help='prune the filter or weight (i.e. FP or WP)')
args = parser.parse_args()

key = args.key
pair_num = args.pair_num
# path1 = f'./generated_all_image/{key}/{pair_num}'
# path2 = f'./paired_all_image/{key}/{pair_num}'
path1 = f'./generated_all_image/{key}_{args.filter_pruning}_{args.prune_ratio}/{pair_num}'
path2 = f'./paired_all_image/{key}_{args.filter_pruning}_{args.prune_ratio}/{pair_num}'

device = 'cuda:0'
transform = transforms.Compose([
    #transforms.PILToTensor(),
    transforms.ToTensor()
])
lost_list = []
for idx in range(1,pair_num+1):
    img_generated = torch.unsqueeze(transform(Image.open(path1+f'/{idx}.png')).to(device),0)
    img_paired = torch.unsqueeze(transform(Image.open(path2+f'/{idx}.png')).to(device),0)
    #print(img_generated.shape)
    #print(img_generated)
    v = torch.mean(torch.norm(torch.sub(img_generated,img_paired).reshape(1,-1),dim=1))
    lost_list.append(v.item())    

if not os.path.exists('./mse_list'):
    os.mkdir('./mse_list')


#np.save(f'mse_list/mse_{key}_{pair_num}.npy',np.array(lost_list))

print(f"{key}\t{pair_num}\tmean:{sum(lost_list)/len(lost_list)}")


