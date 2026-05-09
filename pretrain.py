import argparse
import collections
import datetime
import json
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import util.datasets as ds

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed
import models_mae
import models_global_branch
import models_momentum
from tqdm import tqdm
from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=372, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch2', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--momentum_model', default='Momentum_base_patch2_embed256', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--encoder', type=str, default='enc_base_patch2_embed256', metavar='MODEL', )
    parser.add_argument('--decoder', type=str, default='dec_base_patch2_embed256', metavar='MODEL', )

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.1, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--trainset', type=str, default='imagenet', choices=['cifar100', 'imagenet'])
    parser.add_argument('--train_data_path_imagenet', default='/home/csudz/Desktop/Imagenet/train', type=str,
                        help='dataset path')
    parser.add_argument('--train_data_path_cifar100', default='/home/csudz/Desktop/dsm/TJSCC/dataset/cifar100', type=str)
    parser.add_argument('--output_dir', default='output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # channel parameters
    parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13')
    parser.add_argument('--chan_type', type=str, default='awgn')
    parser.add_argument('--C', type=int, default=64, help='bottleneck dimension')
    parser.add_argument('--selected_nodes_num', type=int, default=16, help='bottleneck dimension')
    parser.add_argument('--window_size', type=int, default=4, help='bottleneck dimension')
    parser.add_argument('--alpha', type=float, default=1, help='bottleneck dimension')
    parser.add_argument('--beta', type=float, default=1, help='bottleneck dimension')
    parser.add_argument('--finetune',
                        default='/home/csudz/Desktop/dsm/TJSCC/MAE_FS_v2/outputs/train_global_branch/checkpoint-49.pth',
                        help='finetune from checkpoint')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    if args.trainset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR100(args.train_data_path_cifar100, train=True, download=True,
                                          transform=transform_train)

    elif args.trainset == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_train_2 = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = ds.ImagenetFolder(args.train_data_path_imagenet, transform=transform_train, transform_=transform_train_2)
        # transform_train = transforms.Compose([
        #         transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # train_dataset = datasets.ImageFolder(args.train_data_path_imagenet, transform=transform_train)

    global_rank = misc.get_rank()
    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, chan_type=args.chan_type,
                                            multiple_snr=args.multiple_snr, C=args.C,
                                            selected_nodes_num=args.selected_nodes_num,
                                            mask_ratio=args.mask_ratio)

    model.to(device)

    model_without_ddp = model

    momentum_model = models_momentum.__dict__[args.momentum_model]()
    momentum_model.to(device)

    target_model = models_global_branch.__dict__[args.encoder](C=64, selected_nodes_num=args.selected_nodes_num,
                                                 multiple_snr=args.multiple_snr, chan_type=args.chan_type)
    checkpoint = torch.load(args.finetune,  weights_only=False)
    checkpoint_model = checkpoint['model']
    checkpoint_model = collections.OrderedDict(
        [(k.replace('encoder.', '', 1), v) if k.split('.')[0] == 'encoder' else (k, v) for k, v in
         checkpoint_model.items()]
    )
    msg =  target_model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    target_model.half().to(device)

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    start_time = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, momentum_model, target_model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and ((epoch+1) % 100 == 0 or epoch + 1 == args.epochs or epoch == 0):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path('./outputs').mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        args.output_dir = os.path.join('./outputs', args.output_dir)
        args.log_dir = os.path.join('./outputs', args.log_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
