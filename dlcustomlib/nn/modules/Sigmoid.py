from ..Activation import Activation
import torch

def sigmoid_f(x):
        return 1 / (1 + torch.exp(-x))

def sigmoid_b(x):
        s = sigmoid_f(x)
        return (s * (1 - s))

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(fwd_func=sigmoid_f, bwd_func=sigmoid_b)

        