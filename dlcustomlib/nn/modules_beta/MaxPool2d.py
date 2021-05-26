from ..Activation import Activation
import torch

class MaxPool2d(Activation):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self._stride = 2
        self._a = None
        self._cache = {}

    def forward(self, x):
        n, c, _,_= x.shape
        pool_h = x[0,0,:,:].shape[0] // self.kernel_size
        pool_w = x[0,0,:,:].shape[1] // self.kernel_size
        input_h = pool_h * self.kernel_size
        input_w = pool_w * self.kernel_size
        output = torch.zeros((n,c,pool_h, pool_w))

        for i in range(n):
            for j in range(c):
                out= x[:input_h, :input_w]
                out = torch.reshape(out, (-1, self.kernel_size))
                out, _ = torch.max(out, dim=1)
                out = torch.reshape(out, (pool_h, self.kernel_size, pool_w))
                out, _ = torch.max(out, dim=1)
                output[i,j,:,:] = out
            
        return output

    def backward(self, *output_grad):
        raise NotImplementedError
