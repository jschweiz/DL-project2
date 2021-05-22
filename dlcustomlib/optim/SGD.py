from .Optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, model, lr = 0.01, weight_decay = 0):
        super().__init__(model, lr)
        self.weight_decay = weight_decay

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
        
    def step (self):
        for layer in self.layers:
            if layer.requires_grad():
                layer.step(self.lr, self.weight_decay)