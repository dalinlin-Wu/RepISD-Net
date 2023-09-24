import math
import numpy as np

import torch
import torch.nn as nn

import copy
import os


def ECB_deploy(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        save_path_convert = os.path.join(save_path, 'deploy_yours.pth.tar')
        torch.save(model.state_dict(), save_path_convert)
    return model

class Conv2d(nn.Module):
    def __init__(self, cdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.cdc = cdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.cdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class CDC(nn.Module):
    def __init__(self, cdc, inplane, ouplane):
        super(CDC, self).__init__()
        self.conv1 = Conv2d(cdc, inplane, ouplane, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ouplane)

        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(ouplane)

    def forward(self, x):

        y = self.conv1(x)
        y = self.bn1(y)

        x = self.conv2(x)
        x= self.bn2(x)
        y = y + x
        return y

class CDC_converted(nn.Module):

    def __init__(self, inplane, ouplane):
        super(CDC_converted, self).__init__()

        self.conv1 = nn.Conv2d(inplane, ouplane, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ouplane)

        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(ouplane)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)

        x = self.conv2(x)
        x = self.bn2(x)
        y = y + x
        return y



def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def cdc_conv_bn(cdcs,in_channels, out_channels,  ):
    result = nn.Sequential()
    result.add_module('conv_cdc', CDC(cdcs,in_channels, out_channels))
    return result

def cdc_conv_bn_convert(cdcs,in_channels, out_channels,  ):
    result = nn.Sequential()
    result.add_module('conv_cdc', CDC_converted(in_channels, out_channels))
    return result


class ECB_block(nn.Module):

    def __init__(self, cdcs,in_channels, out_channels, kernel_size,stride=1, padding=0, convert=False,deploy=False,):
        super(ECB_block, self).__init__()
        self.deploy = deploy
        self.convert = convert

        self.in_channels = in_channels
        #
        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()


        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, bias=True)

        else:
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.rbr_identity = None

            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            if self.convert:
                self.rbr_cdc = cdc_conv_bn_convert(cdcs,in_channels, out_channels,)
            else:
                self.rbr_cdc = cdc_conv_bn(cdcs,in_channels, out_channels,)

    def forward(self, x):
        if self.deploy:

            return self.nonlinearity(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        out = self.rbr_dense(x) + self.rbr_cdc(x) + id_out

        out = self.nonlinearity(out)
        return out

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor_ori(self.rbr_dense)
        kernelcdc_3x3, biascdc_3x3,kernelcdc_1x1, biascdc_1x1 = self._fuse_bn_tensor_cdc_after_convert(self.rbr_cdc)
        kernelid, biasid = self._fuse_bn_tensor_ori(self.rbr_identity)
        return kernel3x3 + kernelcdc_3x3 + self._pad_1x1_to_3x3_tensor(kernelcdc_1x1) + kernelid, bias3x3 + biascdc_3x3 + biascdc_1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor_ori(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            #   For the 1x1 or 3x3 branch

            kernel, running_mean, running_var, gamma, beta, eps = branch.conv.weight, branch.bn.running_mean, branch.bn.running_var, branch.bn.weight, branch.bn.bias, branch.bn.eps

        else:
            #   For the identity branch
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                #   Construct and store the identity kernel in case it is used multiple times
                input_dim = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel, running_mean, running_var, gamma, beta, eps = self.id_tensor, branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor_cdc_after_convert(self, branch):
        if branch is None:
            return 0, 0

            ###kernel=3
        kernel_conv1, running_mean_bn1, running_var_bn1, gamma_bn1, beta_bn1, eps_bn1 = branch.conv_cdc.conv1.weight, branch.conv_cdc.bn1.running_mean, branch.conv_cdc.bn1.running_var, branch.conv_cdc.bn1.weight, branch.conv_cdc.bn1.bias, branch.conv_cdc.bn1.eps
        ###kernel=1
        kernel_conv2, running_mean_bn2, running_var_bn2, gamma_bn2, beta_bn2, eps_bn2 = branch.conv_cdc.conv2.weight, branch.conv_cdc.bn2.running_mean, branch.conv_cdc.bn2.running_var, branch.conv_cdc.bn2.weight, branch.conv_cdc.bn2.bias, branch.conv_cdc.bn2.eps


        std_bn1 = (running_var_bn1 + eps_bn1).sqrt()
        t_bn1 = (gamma_bn1 / std_bn1).reshape(-1, 1, 1, 1)

        std_bn2 = (running_var_bn2 + eps_bn2).sqrt()
        t_bn2 = (gamma_bn2 / std_bn2).reshape(-1, 1, 1, 1)

        return kernel_conv1 * t_bn1, beta_bn1 - running_mean_bn1 * gamma_bn1 / std_bn1, kernel_conv2 * t_bn2, beta_bn2 - running_mean_bn2 * gamma_bn2 / std_bn2

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_cdc')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True








