from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .nnfunc import NNFunc, copy_param_val

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x, params = None, **kwargs):
        if params is not None:
            copy_param_val(self, params, **kwargs)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        # x = F.log_softmax(x, dim=2).squeeze()
        x = x.squeeze()
        return x

# ## Normal training
# from tqdm import tqdm
# import torch.optim as optim
# import torch
# from dataset import get_speechcommand_dataloader

# device = torch.device('cuda:0')
# class M5o(nn.Module):
#     def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
#         super().__init__()
#         self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
#         self.bn1 = nn.BatchNorm1d(n_channel)
#         self.pool1 = nn.MaxPool1d(4)
#         self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
#         self.bn2 = nn.BatchNorm1d(n_channel)
#         self.pool2 = nn.MaxPool1d(4)
#         self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
#         self.bn3 = nn.BatchNorm1d(2 * n_channel)
#         self.pool3 = nn.MaxPool1d(4)
#         self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
#         self.bn4 = nn.BatchNorm1d(2 * n_channel)
#         self.pool4 = nn.MaxPool1d(4)
#         self.fc1 = nn.Linear(2 * n_channel, n_output)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(self.bn1(x))
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu(self.bn2(x))
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = F.relu(self.bn3(x))
#         x = self.pool3(x)
#         x = self.conv4(x)
#         x = F.relu(self.bn4(x))
#         x = self.pool4(x)
#         x = F.avg_pool1d(x, x.shape[-1])
#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=2).squeeze()

# def test_speechcommand():
#     train_loader, test_loader = get_speechcommand_dataloader()
#     model = M5o(n_input=1, n_output=35)

#     model.to(device)
#     print(model) 
#     optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)   
#     log_interval = 20
#     n_epoch = 10

#     pbar_update = 1 / (len(train_loader) + len(test_loader))
#     losses = []
#     def train(model, epoch, log_interval):
#         model.train()
#         for batch_idx, (data, target) in enumerate(train_loader):

#             data = data.to(device)
#             target = target.to(device)

#             # apply transform and model on whole batch directly on device
#             # data = transform(data)
#             output = model(data)

#             # negative log-likelihood for a tensor of size (batch x 1 x n_output)
#             loss = F.nll_loss(output.squeeze(), target)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             ## batch size is 256, evaluate the training acc every 20 batches
#             ## evaluate the testing acc through the whole dataset
#             # print training stats
#             if batch_idx % log_interval == 0:
#                 print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

#             # update progress bar
#             pbar.update(pbar_update)
#             # record loss
#             losses.append(loss.item())
    
#     def number_of_correct(pred, target):
#         # count number of correct predictions
#         return pred.squeeze().eq(target).sum().item()

#     def get_likely_index(tensor):
#         # find most likely label index for each element in the batch
#         return tensor.argmax(dim=-1)

#     def test(model, epoch):
#         model.eval()
#         correct = 0
#         for data, target in test_loader:

#             data = data.to(device)
#             target = target.to(device)

#             # apply transform and model on whole batch directly on device
#             # data = transform(data)
#             output = model(data)

#             pred = get_likely_index(output)
#             correct += number_of_correct(pred, target)

#             # update progress bar
#             pbar.update(pbar_update)

#         print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
#     with tqdm(total=n_epoch) as pbar:
#         for epoch in range(1, n_epoch + 1):
#             train(model, epoch, log_interval)
#             test(model, epoch)
#             scheduler.step()

# if __name__ == '__main__':
#     test_speechcommand()