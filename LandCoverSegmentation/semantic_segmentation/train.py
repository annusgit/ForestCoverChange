from __future__ import print_function
from __future__ import division
import os
import sys
import torch
import argparse
from model import *
from training_functions import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', dest='function', default='train_net')
    parser.add_argument('--data', dest='data', default='generated_data')
    parser.add_argument('--input_size', dest='input_dim', type=int, default=128)
    parser.add_argument('--topology', dest='model_topology', default='Unet')
    parser.add_argument('--bands', dest='bands', nargs='+', type=int)
    parser.add_argument('--classes', dest='classes', nargs='+', type=str)
    parser.add_argument('--workers', dest='workers', type=int, default=4)
    parser.add_argument('--pretrained_model', dest='pre_model', type=str, default='None')
    parser.add_argument('--data_split_lists', dest='data_split_lists', default=None)
    parser.add_argument('--models_dir', dest='models_dir', default=None)
    parser.add_argument('--summary_dir', dest='summary_dir', default=None)
    parser.add_argument('--error_maps_dir', dest='error_maps_path', default=None)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--epochs', dest='epochs', type=int, default=500)
    parser.add_argument('--log_after', dest='log_after', type=int, default=10)
    parser.add_argument('--cuda', dest='cuda', type=int, default=0)
    parser.add_argument('--device', dest='device', type=int, default=0)
    args = parser.parse_args()
    print('\n\n' + "#" * 100)
    print("LOG: The Following Command-Line Parameters Have Been Adopted")
    for index, (key, value) in enumerate(args.__dict__.items(), 1):
        print("{}. {} = {}".format(index, key, value))
    print("#" * 100 + '\n\n')
    function_to_call = eval(args.function)
    net = UNet(topology=args.model_topology, input_channels=len(args.bands), num_classes=len(args.classes))
    function_to_call(model=net, model_topology=args.model_topology, generated_data_path=args.data, input_dim=args.input_dim, bands=args.bands,
                     classes=args.classes, workers=args.workers, pre_model=args.pre_model, data_split_lists=args.data_split_lists, save_dir=args.models_dir,
                     sum_dir=args.summary_dir, error_maps_path=args.error_maps_path, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                     log_after=args.log_after, cuda=args.cuda, device=args.device)
    pass
