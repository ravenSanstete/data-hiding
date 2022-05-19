import torch
import torchaudio
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--key',default='Speech-onlyG',type=str,help='name')
parser.add_argument('--pair_num',default=512,type=int,help='total num of the noise-pair')
args = parser.parse_args()

key = args.key
pair_num = args.pair_num

#path1 = f'./generated_all_voice/{key}/{pair_num}'
#path2 = f'./paired_all_voice/{key}/{pair_num}'

path1 = f'/home/myang_20210409/yyf/model_overloading/checkpoint/rc_so_4096_new/g_pair_voice/wav/'
loss_list = []
for idx in range(1,pair_num+1):
    voice_generated,_ = torchaudio.load(path1+f'/{idx}.wav',channels_first=True)
    voice_paired,_ = torchaudio.load(path1+f'/{idx}_gt.wav',channels_first=True)
    loss =torch.mean(torch.norm(torch.sub(voice_generated,voice_paired),dim=1))
    loss_list.append(loss.item())

if not os.path.exists('./mse_list'):
    os.mkdir('./mse_list')
np.save(f'mse_list/mse_{key}_{pair_num}.npy',np.array(loss_list))


print(f'{key}\t{pair_num}\t{sum(loss_list)/len(loss_list)}')