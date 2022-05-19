import torch
import math
import warnings

##################################
### The Adam algorithm       ####
##################################
class EasyAdam(object):
    def __init__(self, beta1 = 0.9, beta2=0.999):
        self.step = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.m1 = None
        self.m2 = None

    def calc_grad(self, grad, lr, p = None):
        eps = 1e-8
        if(self.m1 is None):
            self.m1 = torch.zeros_like(grad)
            self.m2 = torch.zeros_like(grad)
        self.step += 1
        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step
        self.m1 = self.beta1 * self.m1.detach() + (1-self.beta1) * grad
        self.m2 = self.beta2 * self.m2.detach() + (1-self.beta2) * (grad ** 2).detach()
        
        m2_debiased = self.m2 / (1 - self.beta2 ** self.step)
        step_size = lr / bias_correction1
        return step_size * (self.m1 / ((self.m2.sqrt()/math.sqrt(bias_correction2)) + eps))

class EasySGD(object):
    def __init__(self, momentum=0.9, weight_decay = 5e-4, dampening = 0, nesterov = False):
        self.step = 0
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening
        self.param_state = dict()

    def calc_grad(self, grad, lr, p):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        d_p = grad
        if self.weight_decay != 0: # 进行正则化
            # add_表示原处改变，d_p = d_p + weight_decay*p.data
            d_p.add_(p.data, alpha=self.weight_decay)
        if self.momentum != 0:
            # param_state = self.state[p] # 之前的累计的数据，v(t-1)
            # 进行动量累计计算
            if 'momentum_buffer' not in self.param_state:
                buf = self.param_state['momentum_buffer'] = torch.clone(d_p).detach()
            else:
                # 之前的动量
                buf = self.param_state['momentum_buffer']
                # buf= buf*momentum + （1-dampening）*d_p
                buf.mul_(self.momentum).add_(d_p, alpha=1 - self.dampening)
            if self.nesterov: # 使用neterov动量
                # d_p= d_p + momentum*buf
                d_p = d_p.add(self.momentum, buf)
            else:
                d_p = buf
        # p = p - lr*d_p
        # p.data.add_(lr, d_p)
        return lr * d_p

class StepLR(object):
    def __init__(self) -> None:
        super().__init__()
    
    def get_lr(self, lr = None, epoch=None, epoch_size = 20, gamma=0.1):
        if epoch % epoch_size == 0:
            lr = lr * gamma
        return lr

