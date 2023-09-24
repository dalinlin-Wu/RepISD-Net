import matplotlib.colors
from torchvision      import transforms
import torch.utils.data as Data
import time
# Metric, loss .etc
from utils.utils import *
from utils.metrics import *

# Model

from model.RepISD import RepISD
from convert_tool.convert_CDC import config_model

import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Demo')

    # data and pre-process
    parser.add_argument('--img_demo_dir', type=str, default='demo_img',
                        help='img_demo')
    parser.add_argument('--img_demo_index', type=str, default='Misc_413',
                        help='Misc_413,XDU451')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--log_name', type=str, default='RepISD_Net')
    parser.add_argument('--model', type=str, default='RepISD')


    parser.add_argument('--config', type=str, default='cdc', help=' c_pdc, a_pdc')
    parser.add_argument('--convert', type=bool, default=True, help='')
    parser.add_argument('--deploy', type=bool, default=True, help='')

    parser.add_argument('--model_dir', type=str,default = 'NUAA/deploy.pth.tar')


    args = parser.parse_args()

    # the parser
    return args

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        for i in args.img_demo_index.split(','):
            img_dir = args.img_demo_dir + '/' + i + args.suffix
            # Preprocess and load data
            input_transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize([0.3546, 0.3544, 0.3553], [0.2634, 0.2632, 0.2639])])

            data = DemoLoader(img_dir, base_size=256, crop_size=256, transform=input_transform,
                              suffix=args.suffix)
            img = data.img_preprocess()

            # Choose and load model (this paper is finished by one GPU)
            if args.model == 'RepISD':
                cdc = config_model(args.config)
                model       = RepISD(cdc[0],bins=(1, 2, 3, 6), dropout=0.1, classes=2,convert=args.convert,deploy=args.deploy)

            model           = model.cuda()
            model.apply(weights_init_xavier)
            print("Model Initializing")
            self.model      = model
            self.model = model
            param = count_param(model)
            print('------------------------------------------------')
            FLOPs = calc_flops(model, 512)


            # Load trained model
            if args.deploy:
                checkpoint = torch.load( 'result/' + args.model_dir)
                self.model.load_state_dict(checkpoint)
            # Test
            self.model.eval()
            img = img.cuda()
            img = torch.unsqueeze(img, 0)

            res = []
            torch.cuda.synchronize()
            start = time.time()

            preds = self.model(img)
            pred = preds[:, 1:, :, :]

            torch.cuda.synchronize()  ###等待当前设备上所有流中的所有核心完成。
            end = time.time()
            res.append(end - start)
            time_sum = 0
            for i in res:
                time_sum += i
            FPS = 1.0 / (time_sum / len(res))
            print("FPS: %f" % (FPS))

            save_Pred_GT_visulize(pred, args.img_demo_dir, args.img_demo_index, args.suffix)



def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





