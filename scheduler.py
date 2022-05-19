import math
import numpy as np


class StepLR():
    def __init__(self,init_lr,step_size,gamma=0.1,last_step=-1):
        self.lr = init_lr
        self.step_size = step_size
        self.gamma = gamma
        self.last_step = last_step

    def get_lr(self):
        if (self.last_step == 0) or (self.last_step % self.step_size != 0):
            return self.lr 
        else:
            return self.lr*self.gamma

    def step(self):
        self.last_step += 1
        self.lr = self.get_lr()

    def get_last_lr(self):
        return self.lr 

            
        

class CosineAnnealingLR():
    def __init__(self, init_lr, T_max, eta_min=0, last_step=-1):
        self.lr = init_lr
        self.step_size = T_max
        self.eta_max = init_lr
        self.eta_min = eta_min
        self.last_step = last_step

    def step(self):
        self.last_step += 1
        self.lr = self.get_lr()

    def get_last_lr(self):
        return self.lr         

    def get_lr(self):
        if(self.last_step == 0):
            return self.lr
        elif (self.last_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.lr + (self.eta_max - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
        return (1+math.cos(math.pi * self.last_step / self.T_max))/ (1+math.cos(math.pi * (self.last_step - 1)/self.T_max)) * (self.lr - self.eta_min) + self.eta_min
    
        
        
        
