from ..Activation import Activation

def tanh_f(x):
    return (2 / (1 + (-2*x).exp())) - 1

def tanh_b(x):
    return 1 - tanh_f(x)**2

class Tanh(Activation):
    def __init__(self):
        super().__init__(fwd_func=tanh_f, bwd_func=tanh_b)
