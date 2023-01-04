import logging
import datetime
import os
import json
import datasets
import collections.abc
from itertools import repeat
import torch
import warnings
import math
import time
import numpy as np
import skimage
from torchvision.utils import make_grid

def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def get_datasets(args, overwrite_data_root = None):
    data_root = overwrite_data_root if overwrite_data_root is not None else args.data_root
    if "DIV2K" in (args.train_dataset, args.val_dataset):
        if args.train_dataset == "DIV2K":
            train_dataset = datasets.DIV2K("train", data_root, crop=args.patch_size)
        if args.val_dataset == "DIV2K":
            val_dataset = datasets.DIV2K("val", data_root, crop=args.patch_size)
    if "CIFAR10" in (args.train_dataset, args.val_dataset):
        if args.train_dataset == "CIFAR10":
            train_dataset = datasets.CIFAR10("train", data_root, conditional = args.conditional)
        if args.val_dataset == "CIFAR10":
            val_dataset = datasets.CIFAR10("val", data_root, conditional = args.conditional)
        
    return train_dataset, val_dataset

def get_val_dataset(name, data_root, type = "val", idx = None):
    if name.upper() == "DIV2K" :
        return datasets.DIV2K_evaluation(type, data_root, idx=idx)
    if name.upper() =="CIFAR10" :
        return datasets.CIFAR10(type, data_root)
    if name.upper() =="KODAK" :
        return datasets.KODAK_evaluation(data_root, idx=idx)
        

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    """Save model checkpoint

    Args:
        states: model states.
        is_best (bool): whether to save this model as best model so far.
        output_dir (str): output directory to save the checkpoint
        filename (str): checkpoint name
    """
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def train(train_dataloader, model, loss, optim, epoch, writer, logger, device, scaler, args):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_dataloader):
        img = input["img"]
        coords = input["coords"]
        gt_image = target["img"]

        data_time.update(time.time() - end)

        img = img.to(device)
        coords = coords.to(device)
        gt_image = gt_image.to(device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(img, coords)
            # logger.info(output)
            loss_batch = loss(output, gt_image)

        optim.zero_grad()
        
        scaler.scale(loss_batch).backward()
        scaler.step(optim)
        scaler.update()

        losses.update(loss_batch.item(), img.size(0))
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         logger.info(name + " " + str(param.data))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.frequent == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_dataloader), batch_time=batch_time,
                      speed=img.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)
            print(msg)
            # logger.info(output)

            global_steps = i + epoch*len(train_dataloader)
            writer.add_scalar('train_loss', losses.val, global_steps)

def lin2img(tensor, image_resolution=None):
    batch_size, _ , channels = tensor.shape
    
    
    height = image_resolution[0]
    width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

def psnr_ssim(pred_img, gt_img):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()

    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        # p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        # trgt = (trgt / 2.) + 0.5

        ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)
    return np.mean(np.asarray(psnrs)), np.mean(np.asarray(ssims))

def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x

def validate(val_dataloader, model, loss, epoch, writer, logger, device, image_resolution, args):

    batch_time = AverageMeter()
    losses = AverageMeter()
    psnrs = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_dataloader):
            img = input["img"]
            coords = input["coords"]
            gt_image = target["img"]


            img = img.to(device)
            coords = coords.to(device)
            gt_image = gt_image.to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):

                output = model(img, coords)

                loss_batch = loss(output, gt_image)

            num_images = img.size(0)
            losses.update(loss_batch.item(), num_images)
            psnr, ssim = psnr_ssim(lin2img(output, image_resolution), lin2img(gt_image, image_resolution))
            psnrs.update(psnr, num_images)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.frequent == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                        'Psnr {psnr.val:.4f} ({psnr.avg:.4f})'.format(
                          i, len(val_dataloader), batch_time=batch_time,
                          loss=losses, psnr=psnrs)

                logger.info(msg)
                print(msg)


                out_img = lin2img(output, image_resolution)
                gt_img = lin2img(gt_image, image_resolution)

                # out_img = rescale_img((out_img+1)/2, mode='clamp')
                out_img = rescale_img(out_img, mode='clamp')


                # gt_img = rescale_img((gt_img+1) / 2, mode='clamp')
                gt_img = rescale_img(gt_img, mode='clamp')


                output_vs_gt = torch.cat((out_img, gt_img), dim=-1)
                writer.add_image(str(input["idx"].tolist()) + "_pred_vs_gt", make_grid(output_vs_gt), global_step=epoch)
        writer.add_scalar('val_loss', losses.avg, epoch)
        writer.add_scalar("val_psnr", psnrs.avg, epoch)
        logger.info("avg loss " + str(losses.avg))
        print("avg loss " + str(losses.avg))
        logger.info("avg psnr " + str(psnrs.avg))
        print("avg psnr " + str(psnrs.avg))
    
    return psnrs.avg

def get_mape_loss():
    def mape_loss(output, target):
        abs = torch.abs(output - target)
        target_abs = torch.abs(target)
        eps = (torch.ones(target.shape)*torch.finfo(torch.float64).eps).to(target.device)
        denominator = torch.maximum(target_abs, eps)
        return torch.mean( abs / (denominator)).to(target.device)
    return mape_loss

class PrefetchLoader:
    def __init__(self, loader) -> None:
        self.loader = loader
    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = {k: v.cuda(non_blocking=True) for k, v, in next_input.items()}
                next_target = {k: v.cuda(non_blocking=True) for k, v, in next_target.items()}
            if not first:
                yield input, target
            else:
                first = False
            
            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
        
        yield input, target
    
    def __len__(self):
        return len(self.loader)
