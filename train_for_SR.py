import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from MAE_FS_v2.util.datasets import get_loader
from util.pos_embed import interpolate_pos_embed, interpolate_latent_pos_embed
from util.utils import logger_configuration, load_model, save_model

torch.backends.cudnn.benchmark = True
import argparse
from tqdm import tqdm
from engine_sr import train_one_epoch, test

import models_sr
from models_sr import Net

def get_args_parser():
    parser = argparse.ArgumentParser(description='MAE_JSCC')
    # train
    parser.add_argument('--training', action='store_true', help='training or testing')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_model_freq', default=200, type=int)
    parser.add_argument('--test_model_freq', default=10, type=int)
    parser.add_argument('--print_step', default=276, type=int)
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--global_step', default=0, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--batch_size', default=160, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--seed', default=1024, type=int)

    # Dataset
    parser.add_argument('--trainset', type=str, default='DIV2K', choices=['cifar100', 'cifar10', 'DIV2K'])
    parser.add_argument('--testset', type=str, default='DIV2K', choices=['cifar100', 'cifar10', 'DIV2K'])
    parser.add_argument('--train_data_path', default='/home/csudz/Desktop/dsm/Dataset/DIV2K/DIV2K_train_HR_sub_4x', type=str)
    parser.add_argument('--test_data_path', default='/home/csudz/Desktop/dsm/Dataset/DIV2K/DIV2K_valid_HR_sub_4x', type=str)
    parser.add_argument('--input_size', default=32, type=int, help='images input size')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # model
    parser.add_argument('--encoder', type=str, default='enc_base_patch2_embed256', metavar='MODEL', )
    parser.add_argument('--decoder', type=str, default='dec_base_patch2_embed256', metavar='MODEL', )
    parser.add_argument('--finetune',
                        default='/home/csudz/Desktop/dsm/TJSCC/MAE_FS_v2/outputs/pt/Rayleigh/pt_imagenet_MAE-FSv2_Rayleigh_mr-0.125/checkpoint-399.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--scale_factor', type=int, default=4)

    # channel
    parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'])
    parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13', help='random or fixed snr')
    parser.add_argument('--C', type=int, default=64, help='bottleneck dimension')
    parser.add_argument('--selected_nodes_num', type=int, default=16, help='bottleneck dimension')
    parser.add_argument('--window_size', default=4, type=int)
    parser.add_argument('--latent_tokens_num', type=int, default=15, help='bottleneck dimension')

    # Optimizer
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # output
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--log_dir', type=str, default='output')
    return parser


def main(args):
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger = logger_configuration(args, save_log=True)

    data_loader_train, data_loader_val = get_loader(args, args.num_workers)

    encoder = models_sr.__dict__[args.encoder](C=args.C, selected_nodes_num=args.selected_nodes_num,window_size=args.window_size,
                                                multiple_snr=args.multiple_snr, chan_type=args.channel_type,
                                               latent_tokens_num=args.latent_tokens_num)

    if args.finetune and args.training:
        checkpoint = torch.load(args.finetune, weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = encoder.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        keys_to_delete = []
        for key in checkpoint_model.keys():
            if key.startswith('latent_token') or key.startswith('latent_norm') or key.startswith('DownSample') or key.startswith('fc_norm'):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del checkpoint_model[key]

        # interpolate position embedding
        interpolate_pos_embed(encoder, checkpoint_model)
        interpolate_latent_pos_embed(checkpoint_model, args.latent_tokens_num)

        # load pre-trained model
        msg = encoder.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    decoder = models_sr.__dict__[args.decoder](C=args.C, selected_nodes_num=args.selected_nodes_num, scale_factor=args.scale_factor)
    model = Net(encoder=encoder, decoder=decoder,
                multiple_snr=args.multiple_snr,chan_type=args.channel_type)
    model.to(device)

    model_params = [{'params': model.parameters()}]
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256
    # args.lr = args.blr
    cur_lr = args.lr
    optimizer = optim.Adam(model_params, lr=cur_lr)

    load_model(args, model, optimizer)
    if args.training:
        for epoch in tqdm(range(args.start_epoch, args.epochs)):
            train_one_epoch(args, model, data_loader_train, optimizer, epoch, logger)
            if (epoch + 1) % args.save_model_freq == 0 or (epoch + 1) == args.epochs:
                save_model(args, epoch, model, optimizer)
            if (epoch + 1) % args.test_model_freq == 0:
                test(args, model, data_loader_val, logger)
    else:
        test(args, model, data_loader_val, logger)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path('./outputs').mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        args.output_dir = os.path.join('./outputs', args.output_dir)
        args.log_dir = os.path.join('./outputs', args.log_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
