import torch
import torch.nn as nn
import torch.nn.functional as F
def createConvFunc(op_type):
    assert op_type in ['cv', 'cdc',], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cdc': ##中心
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    else:
        print('impossible to be here unless you force that')
        return None

nets = {
    'cdc': {
        'block':  'cdc',},
}

def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)
    print(str(nets[model]))

    pdcs = []
    layer_name = 'block'
    op = nets[model][layer_name]
    pdcs.append(createConvFunc(op))
    return pdcs


def config_model_converted(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))
    pdcs = []
    layer_name = 'block'
    op = nets[model][layer_name]
    pdcs.append(op)
    return pdcs

def convert_cdc(op, weight):
    if op == 'cv':
        return weight
    elif op == 'cdc':
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.view(shape)
        return weight
    raise ValueError("wrong op {}".format(str(op)))

def convert_CDC(state_dict,config):
    cdcs = config_model_converted(config)
    new_dict =  {}
    for pname,p in state_dict.items():
        if 'double_conv.0.rbr_cdc.conv_cdc.conv1.weight' in pname:
            new_dict[pname] = convert_cdc(cdcs[0],p)
        elif 'double_conv.1.rbr_cdc.conv_cdc.conv1.weight' in pname:
            new_dict[pname] = convert_cdc(cdcs[0],p)
        else:
            new_dict[pname] = p
    return new_dict
