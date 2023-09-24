
import argparse
import os
import sys
import time
from tqdm             import tqdm
import torch.utils.data as Data
from torch.optim      import lr_scheduler

from utils.utils import *
from utils.dice_loss import *
from utils.metrics import *

from model.RepISD import RepISD
from convert_tool.convert_CDC import config_model
from dataset.nuaa import SirstDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Training settings
def parse_args():
    parser = argparse.ArgumentParser(description='Traing')
    parser.add_argument('--root', type=str,default='./data/NUAA-SIRST')
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST')
    parser.add_argument('--log_name', type=str, default='RepISD_Net')
    parser.add_argument('--model', type=str, default='RepISD')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256, help='crop image size')
    parser.add_argument('--base_size', type=int, default=256, help='base image size')
    parser.add_argument('--epochs', type=int, default=600, metavar='N',help='number of epochs to train (default: 600)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--min_lr', default=1e-5,type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',help=' Adagrad，Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',help='learning rate (default: 0.1)')

    parser.add_argument('--config', type=str, default='cdc', help=' ')
    parser.add_argument('--convert', type=bool, default=False, help='False for training')
    parser.add_argument('--deploy', type=bool, default=False, help='False for training')

    args = parser.parse_args()

    # make dir for save result
    args.save_dir = make_dir(True, args.dataset, args.log_name)
    # save training log
    save_train_log(args, args.save_dir)
    # the parser
    return args

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args

        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.best_iou = 0
        self.best_nIoU = 0

        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir


        trainset = SirstDataset(args, mode='train')
        valset = SirstDataset(args, mode='val')
        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'RepISD':
            cdc = config_model(args.config)
            model       = RepISD(cdc[0],bins=(1, 2, 3, 6), dropout=0.1, classes=2,convert=args.convert,deploy=args.deploy)

        model           = model.cuda()
        param = count_param(model)
        print(model)
        print(param)
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model
        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(self.model.parameters(), lr=args.lr, weight_decay=1e-4)

        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()

        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data_loader)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()
            labels = labels.to(torch.int64)

            preds = self.model(data)
            preds = preds[:, 1:, :, :]

            loss = 0
            loss += dice_loss(preds, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), preds.size(0))
            tbar.set_description('Epoch %d, lr:%f,training loss %.4f' % (epoch, self.optimizer.param_groups[0]['lr'],losses.avg))
        self.train_loss = losses.avg

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.val_data_loader)
        self.model.eval()
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()

                preds = self.model(data)
                preds = preds[:, 1:, :, :]
                loss = 0
                loss += dice_loss(preds, labels)

                losses.update(loss.item(), preds.size(0))
                self.iou_metric.update(preds, labels)
                self.nIoU_metric.update(preds, labels)
                _, IoU = self.iou_metric.get()
                _, nIoU = self.nIoU_metric.get()


                tbar.set_description('Epoch %d, test loss %.4f, IoU: %.4f, ' % (epoch, losses.avg, IoU,))
            test_loss=losses.avg
        # save high-performance model
        save_model(IoU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss,  epoch, self.model.state_dict())
        if IoU > self.best_iou:
            self.best_iou = IoU

def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)



if __name__ == "__main__":
    args = parse_args()
    main(args)
