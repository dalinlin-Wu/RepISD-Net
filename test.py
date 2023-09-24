from tqdm             import tqdm
import time
import torch.utils.data as Data
# Metric, loss .etc
from utils.utils import *
from utils.metrics import *

from model.RepISD import RepISD
from convert_tool.convert_CDC import config_model,convert_CDC
from dataset.nuaa import SirstDataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--root', type=str,
                        default='./data/NUAA-SIRST')
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST')

    parser.add_argument('--log_name', type=str, default='RepISD_Net')
    parser.add_argument('--model', type=str, default='RepISD')
    parser.add_argument('--config', type=str, default='cdc', help='')
    parser.add_argument('--convert', type=bool, default=True, help='convert the CDC branch')
    parser.add_argument('--deploy', type=bool, default=False, help='False for training')
    parser.add_argument('--st_model', type=str, default='NUAA-SIRST_RepISD_Net_01_02_2023_14_06_49_wDS',
                        help='')

    ####load the inference-time model  convert = True, deploy = True
    parser.add_argument('--model_dir', type=str,default = 'NUAA/deploy.pth.tar')

    parser.add_argument('--crop-size', type=int, default=256, help='crop image size')
    parser.add_argument('--base-size', type=int, default=256, help='base image size')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--workers', type=int, default=0,metavar='N', help='dataloader threads')




    #  hyper params for training

    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for \
                        testing (default: 32)')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=1,help='')

    args = parser.parse_args()

    # the parser
    return args

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.ROC = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1, args.ROC_thr)
        self.best_iou = 0
        self.best_nIoU = 0


        trainset = SirstDataset(args, mode='train')
        valset = SirstDataset(args, mode='val')
        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'RepISD':
            cdc = config_model(args.config)
            model = RepISD(cdc[0], bins=(1, 2, 3, 6), dropout=0.1, classes=2, convert=args.convert, deploy=args.deploy)

        torch.backends.cudnn.benchmark = False
        model           = model.cuda()
        print(model)
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model
        param = count_param(model)
        print('------------------------------------------------')
        FLOPs = calc_flops(model,512)

        if args.deploy:
            checkpoint = torch.load('result/' + args.model_dir)
            self.model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load('result/' + args.model_dir)
            if args.convert:
                self.model.load_state_dict(convert_CDC(checkpoint['state_dict'], args.config))
                new_weights = convert_CDC(checkpoint['state_dict'], args.config)
                convert_path = 'result/NUAA/converted_yours.pth.tar'  ######save the weight of CDC-branch convertd model.
                torch.save(new_weights, convert_path)
                print('save the converted weights ....')
            else:
                checkpoint = torch.load('result/' + args.model_dir)
                self.model.load_state_dict(checkpoint['state_dict'])


        # Test

        self.model.eval()
        tbar = tqdm(self.val_data_loader)
        self.iou_metric.reset()
        self.nIoU_metric.reset()

        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()


                res = []
                torch.cuda.synchronize()
                start = time.time()


                preds = self.model(data)
                pred = preds[:, 1:, :, :]
                torch.cuda.synchronize()  ###等待当前设备上所有流中的所有核心完成。
                end = time.time()
                res.append(end - start)

                num += 1
                self.iou_metric.update(pred, labels)
                self.nIoU_metric.update(pred, labels)
                self.PD_FA.update(pred, labels)

                _, IoU = self.iou_metric.get()
                _, nIoU = self.nIoU_metric.get()


            FA, PD = self.PD_FA.get(len(self.val_data_loader))
            FA = list(FA)
            PD = list(PD)

            if IoU > self.best_iou:
                self.best_iou = IoU
            if nIoU > self.best_nIoU:
                self.best_nIoU = nIoU


            print('IOU : {}'.format(self.best_iou))
            print('FA : {}'.format(FA))
            print('PD : {}'.format(PD))

            time_sum = 0
            for i in res:
                time_sum += i
            FPS = 1.0 / (time_sum / len(res))
            print("FPS: %f" % (FPS))

            save_result_for_test(args.st_model, IoU, nIoU, FA, PD,param,FLOPs,FPS)


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





