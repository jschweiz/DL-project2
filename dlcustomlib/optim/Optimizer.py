class Optimizer(object):
    def __init__(self, layers, lr):
        self.lr = lr
        self.layers = layers
        
    def step(self):
        raise NotImplementedError