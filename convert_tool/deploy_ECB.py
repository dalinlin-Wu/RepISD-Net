
import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from model.RepISD import RepISD
from model.ECB import ECB_deploy
from convert_tool.convert_CDC import config_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='PSPNet Conversion')
parser.add_argument('--load', metavar='LOAD', default='../result_train/NUAA/converted.pth.tar',help='load the weight of CDC-branch convertd model.')
parser.add_argument('--save', metavar='SAVE', default='../result_train/NUAA/',help='where to save the deployed model.')
parser.add_argument('--config', type=str, default='cdc', help='')
parser.add_argument('--convert', type=bool, default=True, help='')
parser.add_argument('--deploy', type=bool, default=False)  ###转换权重的时候 deploy=False


def convert():
    args = parser.parse_args()
    os.makedirs(args.save,exist_ok=True)
    cdc = config_model(args.config)
    train_model = RepISD(cdc[0],bins=(1, 2, 3, 6), dropout=0.1, classes=2,convert=args.convert,deploy=args.deploy)
    print(train_model)

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        print(ckpt.keys())
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    ECB_deploy(train_model, save_path=args.save)


if __name__ == '__main__':
    convert()