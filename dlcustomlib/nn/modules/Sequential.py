from ..BaseModule import BaseModule

class Sequential(BaseModule):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, *output_grad):
        grad_list = [output_grad[-1]]
        for layer in self.layers[::-1]:
            grad_list.append(layer.backward(grad_list[-1]))         
        return output_grad

    def parameters(self):
        return self.layers