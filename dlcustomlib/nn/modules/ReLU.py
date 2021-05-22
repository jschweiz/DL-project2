from ..Activation import Activation

class ReLU(Activation):        
    def forward(self, x):
        super().forward(x)
        return x * (x > 0)
    
    def backward(self, *output_grad):
        return (self.x > 0) * output_grad[-1]