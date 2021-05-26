from ..Activation import Activation 
import torch

class Flatten(Activation):
    def __init__(self):
        self.shape = ()

    def forward(self, x):
        self.shape = x.shape
        return torch.ravel(x).reshape(x.shape[0], -1)

    def backward(self, *outputgrad):
        da_curr = outputgrad[-1]
        return da_curr.reshape(self.shape)

    def requires_grad(self):
        return False