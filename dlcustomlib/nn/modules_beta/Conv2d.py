from ..BaseModule import BaseModule
import torch, math, numpy as np

class Conv2d(BaseModule):

    def __init__(self, in_dim, out_dim, kernel_size, padding = 'valid', stride = 1):
        std = math.sqrt(1 / (in_dim * kernel_size * kernel_size))
        self.weight = torch.empty( kernel_size, kernel_size, in_dim, out_dim).uniform_(-std, std)
        self.bias = torch.empty(out_dim).uniform_(-std, std)
        self._padding = padding
        self._stride = stride
        self.dweight, self.dbias = None, None
        self._a_prev = None
        self._cols = None

    def step(self, lr, wd):
        self.weight.add_(-lr * self.dweight - lr * wd * self.weight)
        self.bias.add_(-lr * self.dbias - lr * wd * self.bias)

    def forward(self, a_prev):
        a_prev = a_prev.permute(0, 2, 3, 1)
        self._a_prev = a_prev
        output_shape = self.calculate_output_dims(input_dims=a_prev.shape)
        _, h_out, w_out, _ = output_shape
        h_f, w_f, _, _ = self.weight.shape
        pad = 0, 0
        a_prev_pad = self.pad(array=a_prev, pad=pad)
        output = torch.zeros(output_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f

                output[:, i, j, :] = torch.sum( a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] * self.weight[np.newaxis, :, :, :], axis=(1, 2, 3))

        to_return = output + self.bias
        to_return = to_return.permute(0, 3, 1, 2)
        return to_return

    def zero_grad(self):
        self.dweight =0
        self.dbias = 0

    def backward(self, *outputgrad):

        da_curr = outputgrad[-1].permute(0, 2, 3, 1)

        _, h_out, w_out, _ = da_curr.shape
        n, h_in, w_in, _ = self._a_prev.shape
        h_f, w_f, _, _ = self.weight.shape
        pad = 0, 0
        a_prev_pad = self.pad(array=self._a_prev, pad=pad)
        output = torch.zeros_like(a_prev_pad, dtype=float)

        self.dbias = da_curr.sum(axis=(0, 1, 2)) / n
        self.dweight = torch.zeros_like(self.weight)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f
                output[:, h_start:h_end, w_start:w_end, :] += torch.sum(
                    self.weight.unsqueeze(0)*da_curr[:, i:i+1, j:j+1].unsqueeze(3),
                    axis=4
                    )
                self.dweight += torch.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end].unsqueeze(4) *
                    da_curr[:, i:i+1, j:j+1].unsqueeze(3),
                    axis=0
                )
        self.dweight /= n
        return output[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in, :].permute(0, 3, 1, 2)


    # useful functions    

    def calculate_output_dims(self, input_dims):
        n, h_in, w_in, _ = input_dims
        h_f, w_f, _, n_f = self.weight.shape
        h_out = (h_in - h_f) // self._stride + 1
        w_out = (w_in - w_f) // self._stride + 1
        return n, h_out, w_out, n_f

    @staticmethod
    def pad(array, pad):
        return torch.tensor(np.pad(
            array=array,
            pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant'
        ))

    def requires_grad(self):
        return True
    
    def zero_grad(self):
        self.dweight =0
        self.dbias = 0
