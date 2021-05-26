from ..Activation import Activation

def relu_f(x):
    return x * (x > 0)

def relu_b(x):
    return (x > 0)

class ReLU(Activation):        
    def __init__(self):
        super().__init__(fwd_func=relu_f, bwd_func=relu_b)