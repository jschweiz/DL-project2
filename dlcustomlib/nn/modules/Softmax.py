from ..Activation import Activation
import torch

def softmax_f(x):
        expo = torch.exp(x - torch.max(x))
        return expo /torch.sum(expo, 0)

def softmax_b(x):
        s = softmax_f(x)
        return s * (1 - s)

class Softmax(Activation):
    def __init__(self):
        super().__init__(fwd_func=softmax_f, bwd_func=softmax_b)