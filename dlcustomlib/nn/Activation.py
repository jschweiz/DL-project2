from .BaseModule import BaseModule

class Activation(BaseModule):
    def forward(self, x):
        self.x = x
    
    def backward(self, *gradwrtoutput):
        pass

    def requires_grad(self):
        return False

    def zero_grad(self):
        self.x = 0