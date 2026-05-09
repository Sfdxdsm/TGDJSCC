from torch.nn.utils import clip_grad_norm_

from util.utils import *
from util.distortion import *
from util.lr_sched import adjust_learning_rate

def adjust_sigma(n_epoch_warmup, n_epoch, max_sigma, model, loader, step):
    # Calculate the total number of training steps
    max_steps = int(n_epoch * len(loader))
    # Calculate the number of warmup steps
    warmup_steps = int(n_epoch_warmup * len(loader))

    # If we are in the warmup phase
    if step < warmup_steps:
        # Linear warmup
        sigma = max_sigma * step / warmup_steps
    # If we are in the decay phase
    else:
        # Subtract warmup steps from step and max_steps
        step -= warmup_steps
        max_steps -= warmup_steps

        # Cosine decay
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        # Calculate the end sigma value
        end_sigma = 1e-5
        # Calculate the current sigma value
        sigma = max_sigma * q + end_sigma * (1 - q)
    model.sigma = sigma

def train_one_epoch(args, model, train_loader, optimizer, epoch, logger):
    model.train()
    elapsed, losses, loss_Gs, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(7)]
    metrics = [elapsed, losses, loss_Gs, psnrs, msssims, cbrs, snrs]
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(args.device)
    if args.trainset == 'cifar100' or args.trainset == 'imagenet':
        for batch_idx, (input, label) in enumerate(train_loader):
            cur_lr = adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)
            adjust_sigma(args.warmup_epochs, args.epochs, 0.05, model.encoder.feature_selector, train_loader, batch_idx)
            start_time = time.time()
            args.global_step += 1
            input = input.to(args.device)
            recon_image, mse, loss_G, CBR, SNR = model(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            loss_Gs.update(loss_G.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)

            else:
                psnrs.update(100)
                msssims.update(100)

            if (args.global_step % args.print_step) == 0:
                process = (args.global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{args.global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'loss_G ({loss_Gs.avg:.3f})',
                    f'PSNR ({psnrs.avg:.3f})',
                    f'MSSSIM ({msssims.avg:.3f})',
                    f'Lr {cur_lr:.3e}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
        for i in metrics:
            i.clear()


def test(args, model, test_loader, logger):
    model.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(args.device)
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.testset == 'cifar100':
                for batch_idx, (input,label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.to(args.device)
                    recon_image, mse, loss_G, CBR, SNR = model(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})'
                    ]))
                    logger.info(log)
            else:
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    input = input.to(args.device)
                    recon_image, mse, loss_G, CBR, SNR = model(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})'
                    ]))
                    logger.info(log)
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()

    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")
