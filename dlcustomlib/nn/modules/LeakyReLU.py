from ..Activation import Activation
import torch

class LeakyReLU(Activation):
    def lrelu_f(self, x):
        return torch.max(self.alpha * x, x)

    def lrelu_b(self, x):
        return (x > 0)

    def __init__(self, negative_slope=0.01):
        self.alpha = negative_slope
        super().__init__(fwd_func=self.lrelu_f, bwd_func=self.lrelu_b)