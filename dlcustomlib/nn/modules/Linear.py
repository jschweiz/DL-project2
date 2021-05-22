from ..BaseModule import BaseModule
import torch, math

class Linear(BaseModule):
    def __init__(self, in_size, out_size, init_method = None):
        # initialization
        std = math.sqrt(1 / in_size)
        self.weights = torch.empty(in_size, out_size).uniform_(-std, std)
        self.bias = torch.empty(1, out_size).uniform_(-std, std)
        
    def forward(self, x):
        self.x = x
        return (x @ self.weights) + self.bias

    def backward(self, *output_grad):
        self.dw = self.x.t() @ output_grad[-1]
        self.db = output_grad[-1].sum(axis=0)
        return output_grad[-1] @ self.weights.t()
    
    def step(self, lr, wd):
        self.weights.add_(-lr * self.dw - lr * wd * self.weights)
        self.bias.add_(-lr * self.db - lr * wd * self.bias)

    def requires_grad(self):
        return True
    
    def zero_grad(self):
        self.dw =0
        self.db = 0