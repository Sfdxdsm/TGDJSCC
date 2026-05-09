import argparse
import collections
import datetime
import json
import numpy as np
import os

import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

from util.datasets import SVHNDetectionDataset, detection_collate_fn

assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed, interpolate_latent_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torchvision.datasets as datasets
from engine_finetune_od import train_one_epoch, evaluate_detection_with_channel
import torchvision.transforms as transforms
from tqdm import tqdm
import models_od
from timm.utils import ModelEma


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--encoder', type=str, default='enc_base_patch2_embed256', metavar='MODEL', )
    parser.add_argument('--decoder', type=str, default='dec_base_patch2_embed256', metavar='MODEL', )

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune',
                        default='/home/csudz/Desktop/dsm/TJSCC/MAE_FS_v2/outputs/pt/Rayleigh/pt_imagenet_MAE-FSv2_Rayleigh_mr-0.125/checkpoint-399.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--finetune_from_jscc', action='store_true',
                        help='finetune from jscc model or mae')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--train_data_path', default='/home/csudz/Desktop/dsm/TJSCC/dataset/SVHN_Detection', type=str)
    parser.add_argument('--test_data_path', default='/home/csudz/Desktop/dsm/TJSCC/dataset/SVHN_Detection', type=str)
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
    parser.add_argument('--num_queries', default=20, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/home/csudz/Desktop/dsm/TJSCC/MAE_FS_v2/outputs/multi-task/detection/od_pretrain_MAE-FSv2-cifar100_Rayleigh_mr-0.125_ln-15/checkpoint-99.pth',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
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

    parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13')
    parser.add_argument('--chan_type', type=str, default='awgn')
    parser.add_argument('--C', type=int, default=64, help='bottleneck dimension')
    parser.add_argument('--selected_nodes_num', type=int, default=16, help='bottleneck dimension')
    parser.add_argument('--window_size', default=4, type=int)
    parser.add_argument('--latent_tokens_num', type=int, default=5, help='bottleneck dimension')
    parser.add_argument('--bbox_loss_coef', type=float, default=3)
    parser.add_argument('--giou_loss_coef', type=float, default=2)
    parser.add_argument('--eos_coef', type=float, default=0.07)
    parser.add_argument('--set_cost_class', type=float, default=2.0)
    parser.add_argument('--set_cost_bbox', type=float, default=5.0)
    parser.add_argument('--set_cost_giou', type=float, default=3.0)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = SVHNDetectionDataset(args.train_data_path,split='train')
    dataset_val = SVHNDetectionDataset(args.test_data_path,split='test')

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=detection_collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn = detection_collate_fn
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    encoder = models_od.__dict__[args.encoder](drop_path_rate=args.drop_path,C=args.C, selected_nodes_num=args.selected_nodes_num,window_size=args.window_size,
                                            multiple_snr=args.multiple_snr, chan_type=args.chan_type, latent_tokens_num=args.latent_tokens_num)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location=device,weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = encoder.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        keys_to_delete = []
        for key in checkpoint_model.keys():
            if key.startswith('latent_token') or key.startswith('latent_norm'):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del checkpoint_model[key]

        # interpolate position embedding
        interpolate_pos_embed(encoder, checkpoint_model)
        interpolate_latent_pos_embed(checkpoint_model, args.latent_tokens_num)

        # load pre-trained model
        msg = encoder.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    decoder = models_od.__dict__[args.decoder](C=args.C, selected_nodes_num=args.selected_nodes_num, num_queries=args.num_queries, num_classes=args.nb_classes)
    model = models_od.Net(encoder=encoder, decoder=decoder,bbox_loss_coef=args.bbox_loss_coef,
                          giou_loss_coef=args.giou_loss_coef,set_cost_class=args.set_cost_class,
                          set_cost_box=args.set_cost_bbox,set_cost_giou=args.set_cost_giou,eos_coef=args.eos_coef,
                          multiple_snr=args.multiple_snr,chan_type=args.chan_type)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)


    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_params = [{'params': model.parameters()}]
    optimizer = torch.optim.AdamW(model_params, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        model.eval()
        test_stats = evaluate_detection_with_channel(data_loader_val, model, device, args.multiple_snr)
        print(f"Accuracy of the network on the {len(dataset_val)} test images:")
        print(f"mAp@30: {test_stats['map30']}, mAp@50: {test_stats['map50']}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        # epoch % 10 == 0 or
        if args.output_dir and (epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

    test_stats = evaluate_detection_with_channel(data_loader_val, model, device, multiple_snr=args.multiple_snr)
    print(f"Accuracy of the network on the {len(dataset_val)} test images:")
    print(f"mAp@30: {test_stats['map30']}, mAp@50: {test_stats['map50']}")
    # max_accuracy = float(max(max_accuracy, test_stats['map50']))
    # print(f'Max mAp@50: {max_accuracy:.4f}')

    if log_writer is not None:
        [log_writer.add_scalars('perf/mAp@30', test_stats['map30'][i])for i in range (5)]
        [log_writer.add_scalars('perf/mAp@50', test_stats['map50'][i]) for i in range(5)]

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 **{f'test_{k}': v.tolist() for k, v in test_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters}

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
