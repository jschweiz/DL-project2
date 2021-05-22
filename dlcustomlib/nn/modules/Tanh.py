from ..Activation import Activation

class Tanh(Activation):
    def forward(self, x):
        super().forward(x)
        return (2 / (1 + (-2*x).exp())) - 1
    
    def backward(self, *output_grad):
        return (1 - self.forward(self.x)**2) * output_grad[-1]