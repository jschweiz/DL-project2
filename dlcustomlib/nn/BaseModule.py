
class BaseModule (object):
    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def param(self):
        return []