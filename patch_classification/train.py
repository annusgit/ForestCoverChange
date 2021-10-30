

from __future__ import print_function
from __future__ import division
from training_functions import *
from model import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--function', dest='function', default='train_net')
    parser.add_argument('--data', dest='data', default='generated_data')
    parser.add_argument('--input_size', dest='input_dim', type=int, default=64)
    parser.add_argument('--workers', dest='workers', type=int, default=4)
    parser.add_argument('-p', '--pretrained_model', dest='pre_model', type=int, default=-1)
    parser.add_argument('--save_data', dest='save_data', default=None)
    parser.add_argument('-s', '--models_dir', dest='models_dir', default=None)
    parser.add_argument('--summary_dir', dest='summary_dir', default=None)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=4)
    parser.add_argument('-l', '--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=500)
    parser.add_argument('-log', '--log_after', dest='log_after', type=int, default=10)
    parser.add_argument('-c', '--cuda', dest='cuda', type=int, default=0)
    parser.add_argument('--device', dest='device', type=int, default=0)
    args = parser.parse_args()

    function_to_call = eval(args.function)
    net = HyperSpectral_Resnet(in_channels=18, out_channels=2)

    # model, images, labels, block_size, input_dim, workers, pre_model,save_dir,
    #       sum_dir, batch_size, lr, log_after, cuda, device
    function_to_call(model=net, generated_data_path=args.data, input_dim=args.input_dim, workers=args.workers,
                     pre_model=args.pre_model, save_data=args.save_data, save_dir=args.models_dir,
                     sum_dir=args.summary_dir, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                     log_after=args.log_after, cuda=args.cuda, device=args.device)

