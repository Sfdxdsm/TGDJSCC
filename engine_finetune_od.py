# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import numpy as np
import torch

from timm.data import Mixup
from timm.utils import accuracy
from tqdm import tqdm

import util.misc as misc
import util.lr_sched as lr_sched
from util.utils import AverageMeter
from util.distortion import SVHNEvaluator
from util.box_ops import box_cxcywh_to_xyxy_UnNormalize

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        _device_targets = []
        for target in targets:
            _device_target = {
                'boxes': target['boxes'].to(device),
                'labels':target['labels'].to(device)
            }
            _device_targets.append(_device_target)
        targets = _device_targets

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda'):
            loss_dict, CBR, chan_param = model(samples, targets=targets)

        loss = loss_dict['total_loss']
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize(device)

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_detection_with_channel(data_loader, model, device, multiple_snr):
    multiple_snr = multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])

    results = {
        'map30': np.zeros(len(multiple_snr)),
        'map50': np.zeros(len(multiple_snr)),
    }

    evaluator = SVHNEvaluator(iou_thresholds=[0.3,0.5])
    model.eval()

    for snr_idx, snr in enumerate(multiple_snr):
        print(f"\nEvaluating at SNR: {snr}dB")
        evaluator.reset()
        for idx, (images, targets) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict, _, _ = model(images, targets=targets, chan_param=snr)
            pred_logits = loss_dict['pred_logits']
            pred_boxes = loss_dict['pred_boxes']

            probs = torch.softmax(pred_logits, dim=-1)
            scores, labels = probs[...,:-1].max(dim=-1)
            keep_mask = scores > 0.05
            preds = []
            for i in range(len(images)):
                keep = keep_mask[i]
                boxes = box_cxcywh_to_xyxy_UnNormalize(
                    pred_boxes[i][keep],
                    model.encoder.img_size,
                    model.encoder.img_size
                ).cpu().numpy()
                img_labels = labels[i][keep].cpu().numpy()
                img_scores = scores[i][keep].cpu().numpy()

                pred_dict = {
                    'boxes': boxes,
                    'labels': img_labels,
                    'scores': img_scores
                }
                preds.append(pred_dict)

            ground_truths = []
            for target in targets:
                gt_dict = {
                    'boxes': box_cxcywh_to_xyxy_UnNormalize(
                        target['boxes'],
                        model.encoder.img_size,
                        model.encoder.img_size
                    ).cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                }
                ground_truths.append(gt_dict)

            evaluator.evaluate_batch(preds, ground_truths)

        final_results = evaluator.get_detailed_results()

        results['map30'][snr_idx] = final_results['map_0.3']
        results['map50'][snr_idx] = final_results['map_0.5']

        print(f"SNR {snr}dB: mAP@0.3={results['map30'][snr_idx]:.4f}, "
              f"mAP@0.5={results['map50'][snr_idx]:.4f}")

    return results
