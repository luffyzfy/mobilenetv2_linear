
import torch
import torch.nn as nn

class ConvX(nn.Conv2d):
    def _init_( self, in_channels, out_channels, kernel_size,stride = 1, padding = 0, dilation = 1, groups = 1, bias = True ):
        super(ConvX , self)._init_(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.bit = 8

    def forward(self,input):

        max_i = input.max()
        min_i = input.min()
        c = part_quant(input, max_i, min_i, 8, mode = 'activation')
        input = (c[0] - c[2]) * c[1]
        #input = c[0]*c[1] + c[2]

        input = torch.nn.Conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return input


def part_quant(x,max,min,bit,mode):
    scale = (max-min)/(2**bit-1)
    if mode == 'activation':
        zero_point = 0 - torch.round(min/scale)
    elif mode == 'weight':
        zero_point = -128 - torch.round(min/scale)
    x_q = torch.round(x/scale+zero_point)
    return x_q, scale, zero_point
