from .BaseModule import BaseModule

class Activation(BaseModule):
    def __init__(self, fwd_func=lambda x:x, bwd_func=lambda x:x):
        self.fwd_func = fwd_func
        self.bwd_func = bwd_func

    def forward(self, x):
        self.x = x
        return self.fwd_func(x)
        
    def backward(self, *output_grad):
        return self.bwd_func(self.x) * output_grad[-1]

    def requires_grad(self):
        return False

    def zero_grad(self):
        self.x = 0