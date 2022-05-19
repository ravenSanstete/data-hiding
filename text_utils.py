import torch
import torch.nn as nn
import numpy as np
import torchtext
from torchtext import datasets,data  
from torchtext.vocab import GloVe
import glob
import io
import os
import re
import pandas

# TODO:
# look up table 
# containing words,Punctuation,number,<unk>
# def get_lookUpTable():
#     embeddings_index = dict()
#     embedding_dim = 200

#     with open('data/glove.6B.200d.txt') as f:
#         for line in f:
#             word, coefs = line.split(maxsplit=1)
#             coefs = np.fromstring(coefs, "f", sep=" ")
#             embeddings_index[word] = coefs
#             print(word)

#     print(f'Loaded {len(embeddings_index)} word vectors.')
#     return embeddings_index 

class Glove_(GloVe):
    def __init__(self,name,dim) -> None:
        super().__init__(name,dim)

    def __call__(self, token):
        if token in self.stoi:
            index = self.stoi[token]
            vector = self.vectors[index]
            return index,vector
        else:
            return -1,self.unk_init(torch.Tensor(self.dim))

    # convert all tokens into embeddings
    def get_embeddings(self):
        def tokenize(line):
            return re.split("[^a-z^A-Z^0-9]",line)
        tokens = []
        #cop = re.compile("[^a-z^A-Z^0-9]")

        c=0
        for label in ['pos', 'neg']:
            for fname in glob.iglob(os.path.join('data/imdb/aclImdb/train', label, '*.txt')):
                with io.open(fname, 'r', encoding="utf-8") as f:
                    text = f.readline()
                    tokens += tokenize(text)
                    c+=1
                if c>=5000:
                    break

        #print(tokens)
        print(len(tokens))
        # save word,index and embeddings
        dataset_embeddings = []
        count = 0
        for token in tokens:
            token = token.lower()
            if token == "":
                continue
            index,vector = self(token)
            if index == -1:
                count +=1 
                token = '<unk>'
            dataset_embeddings.append((token,index,vector))
            # print(len(dataset_embeddings))
        # print(dataset_embeddings)
        print(f'<unk> number : {count}')
        torch.save(dataset_embeddings,'dataset_embed.pth')

    def search_closest(self,embedding):
        distance_to_unk = torch.norm(torch.sub(embedding,torch.zeros_like(embedding)),p=2)
        current_index = -1
        distance_min = distance_to_unk
        token = '<unk>'

        for i,v in enumerate(self.vectors):
            distance = torch.norm(torch.sub(embedding,v),p=2)
            if distance < distance_min:
                distance_min = distance
                current_index = i 
        if current_index!= -1 :
            token = self.itos[current_index]
        print(token)
        return token,current_index
         
    # convert all recovered embeddings to tokens
    def get_tokens(self):
        embeddings = torch.load('generated_all_text/imdb-onlyG/32/generated_all_embedding.pth')
        print(len(embeddings))
        recover = []
        for embedding in embeddings:
            # search dict
            token,index = self.search_closest(embedding)
            recover.append((token,index))
        # save word 
        torch.save(recover,f'recover_{len(embeddings)}.pth')


    def recover_rate(self,num_words):
        recover = torch.load(f'recover_{num_words}.pth')
        dataset = torch.load(f'dataset_embed.pth')
        c = 0 
        for r,d in zip(recover,dataset):
            if r[1] == d[1]:
                c+=1
        print(f'recover rate : {float(c/num_words)}')

# find embedding
def get_embedding():
    vec = Glove_(name='6B',dim=200)
    
    #vec.get_embeddings()
    # embed = torch.tensor([ 2.6805e-01,  3.6032e-01, -3.3200e-01, -5.4642e-01, -5.0451e-01,
    #     -1.3461e-02, -8.0432e-01, -2.4214e-01,  5.3736e-01,  7.7581e-01,
    #     -3.2554e-01,  4.8300e-01,  8.4265e-01,  3.7780e-01, -1.4767e-01,
    #      5.3192e-01, -7.0518e-01,  4.4037e-01,  7.5035e-01, -1.8171e-01,
    #      7.0139e-01,  2.9383e+00,  4.5612e-02, -2.1176e-01,  1.9947e-01,
    #     -4.8175e-01, -2.5815e-01,  4.6200e-01, -5.6841e-03, -3.0563e-01,
    #     -5.7541e-01, -1.9527e-02, -1.3751e-01, -5.9450e-01, -3.8216e-01,
    #     -1.3541e-01, -6.6444e-01, -2.3028e-01, -5.5466e-02,  3.8421e-01,
    #     -1.6888e-01,  5.1462e-02, -2.8293e-01,  4.5076e-01, -3.6464e-01,
    #      3.6101e-01,  1.0935e+00, -1.1947e-01,  4.9729e-02,  4.8765e-02,
    #      4.8944e-01, -3.3138e-04,  1.6365e-01,  4.9743e-01,  3.3814e-01,
    #      1.5570e-02,  2.5762e-01, -5.8483e-01, -5.5821e-01, -2.9092e-01,
    #      2.3611e-01, -2.8951e-01, -3.1919e-01,  6.5705e-02, -3.1602e-01,
    #     -1.2054e-01, -7.7942e-01,  6.0136e-01,  4.4160e-01, -2.7946e-02,
    #      7.3821e-01, -3.1318e-01, -5.3737e-02, -2.6919e-01, -5.6458e-01,
    #     -6.5164e-01, -1.2298e+00, -5.0430e-02, -7.2749e-01,  8.5426e-02,
    #     -1.4811e-01, -1.5080e-01, -4.5213e-01,  3.4224e-01,  9.9421e-02,
    #     -3.8825e-01, -2.6387e-01, -2.5937e-01, -4.5955e-02, -1.5518e+00,
    #      2.7701e-01, -5.0155e-01,  6.3821e-01, -2.1799e-01, -1.5459e-01,
    #      2.0470e-01,  3.7607e-01,  1.3830e-01, -5.9114e-01, -2.0036e-01,
    #     -1.7630e-02, -2.9715e-01,  1.2323e-02, -1.1470e-01,  8.2837e-01,
    #      1.0221e-01, -2.1023e-01,  1.4215e+00, -5.7118e-01,  3.4696e-01,
    #      1.0750e-01,  2.0036e-01,  1.1781e-01, -1.6939e-01, -6.6335e-02,
    #     -9.4572e-02, -2.1243e-01, -6.3982e-02, -3.0773e-01,  1.4120e-01,
    #      5.5169e-01,  1.4343e-01,  7.4784e-01, -3.1253e-01, -1.0610e-01,
    #     -3.3361e-01,  2.8880e-02, -4.1480e-02,  5.8897e-01, -8.4928e-01,
    #     -3.6634e-01,  4.0954e-02,  8.3578e-02,  2.0159e-01, -3.1628e-01,
    #      3.2837e-01,  7.8545e-02,  6.8703e-02, -3.2559e-01, -5.8249e-01,
    #     -2.0688e-01, -4.9981e-01, -9.8690e-02, -5.5841e-01,  2.0955e+00,
    #      6.1811e-01,  2.3829e-01, -5.2405e-01, -1.3310e-01, -9.6439e-03,
    #      6.3668e-01,  9.2328e-01, -7.3654e-01,  2.0892e-01,  6.4937e-02,
    #     -1.6934e-01, -6.1282e-02, -1.3634e-01,  1.5841e-01,  5.0026e-01,
    #     -2.7620e-01,  2.2532e-01,  8.6730e-02, -5.6935e-02, -4.5413e-01,
    #      1.2981e-01, -1.7020e-02,  4.6354e-01, -4.2211e-01, -7.2322e-02,
    #     -6.1887e-02, -4.0143e-01,  6.6137e-01, -4.6212e-02,  2.7799e-01,
    #      1.3435e-01, -7.7289e-01, -2.1018e-01,  1.0200e+00,  4.4495e-01,
    #      1.1312e+00, -1.7580e-01, -6.7577e-01,  2.8609e-01, -6.0770e-01,
    #     -6.5499e-01,  3.8634e-02,  4.8288e-01,  3.5732e-01,  2.4206e-01,
    #     -1.8247e-01, -2.7803e-01, -6.0281e-02, -6.6602e-02, -5.5558e-02,
    #     -1.9829e-01,  5.3632e-01,  1.7769e-01,  2.2362e-01,  1.4241e-02])
    # vec.search_closest(embed)
    vec.get_tokens()
    vec.recover_rate(num_words=32*224)

# find closeset word

# 把一个数据集的所有embedding统一
# 生成过程，得到embedding npz
# 再额外恢复，并合

get_embedding()