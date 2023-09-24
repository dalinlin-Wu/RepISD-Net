# Basic module
from tqdm             import tqdm

import torch.utils.data as Data
# Metric, loss .etc
from utils.utils import *
from utils.dice_loss import *

# Model

from model.RepISD import RepISD
from convert_tool.convert_CDC import config_model
from dataset.nuaa import SirstDataset,load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--root', type=str,
                        default='./data/NUAA-SIRST')
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST')
    # choose model

    parser.add_argument('--log_name', type=str, default='RepISD_Net')
    parser.add_argument('--model', type=str, default='RepISD')

    parser.add_argument('--config', type=str, default='cdc', help=' c_pdc, a_pdc')
    parser.add_argument('--convert', type=bool, default=True, help='')
    parser.add_argument('--deploy', type=bool, default=True, help='')
    parser.add_argument('--model_dir', type=str,default = 'NUAA/deploy.pth.tar')

    parser.add_argument('--crop-size', type=int, default=256, help='crop image size')
    parser.add_argument('--base-size', type=int, default=256, help='base image size')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')


    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')


    #  可视化的俄时候batch只能是1
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')


    args = parser.parse_args()

    # the parser
    return args

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args

        self.save_prefix = '_'.join([args.model, args.dataset])



        dataset_dir = args.root + '/' + args.dataset
        train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,'idx_427')

        valset = SirstDataset(args, mode='val')
        self.test_data = Data.DataLoader(valset, batch_size=args.batch_size)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'RepISD':
            cdc = config_model(args.config)
            model       = RepISD(cdc[0], bins=(1, 2, 3, 6), dropout=0.1, classes=2,convert=args.convert,deploy=args.deploy)

        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model



        visulization_path = 'result_vis' + '/' + args.log_name  + '_' + args.dataset + '_visulization_result'
        visulization_fuse_path = 'result_vis' + '/' + args.log_name + '_' + args.dataset + '_visulization_fuse'
        make_visulization_dir(visulization_path, visulization_fuse_path)


        # Load trained model
        checkpoint = torch.load('result/' + args.model_dir)  ###depoly  ours
        self.model.load_state_dict(checkpoint)

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)

        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()

                preds = self.model(data)
                pred = preds[:, 1:, :, :]

                save_Pred_GT(pred, labels, visulization_path, val_img_ids, num, '.png')

                num += 1

            total_visulization_generation(dataset_dir,  test_txt, '.png', visulization_path,
                                          visulization_fuse_path)



def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





