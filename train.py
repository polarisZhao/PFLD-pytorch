#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import argparse
import logging
from pathlib import Path
import time

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from dataset import WLFWDatasets
from torch.utils import data
from torch.utils.data import DataLoader

from models import pfld

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)

# def save_checkpoint(state, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     logging.info(f'Save checkpoint to {filename}')

# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected')

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
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    for img, landmark, attribute, euler_angle in train_loader:
        feature , output = model(img)
        loss = criterion(output, landmark)
        print(loss)
        # losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        # if i % args.log_freq == 0:
        #     logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
        #                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')
    

# def validate(val_loader, model, criterion, epoch):
#     model.eval()

#     end = time.time()
#     with torch.no_grad():
#         losses = []
#         for i, (input, target) in enumerate(val_loader):
#             # compute output
#             target.requires_grad = False
#             target = target.cuda(non_blocking=True)
#             output = model(input)

#             loss = criterion(output, target)
#             losses.append(loss.item())

#         elapse = time.time() - end
#         loss = np.mean(losses)
#         logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
#                      f'Loss {loss:.4f}\t'
#                      f'Time {elapse:.3f}')


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    print_args(args)

    # Step 2: model, criterion, optimizer & resume
    model = pfld.PFLDInference()
    criterion = torch.nn.L1Loss(reduction='sum') 
    logging.info("loss log message")

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True) 

    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'==> loading checkpoint {args.resume}')
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            model.load_state_dict(checkpoint)
        else:
            logging.info(f'==> no checkpoint found at {args.resume}')

    # step 3: data 
    # argumetion # ! change data argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    # ! change train dataset and loader
    wlfwdataset = WLFWDatasets(args.dataroot, transform)
    dataloader = DataLoader(wlfwdataset, batch_size=256, shuffle=True, num_workers=0, drop_last=False)

    # for img, landmark, attribute, euler_angle in dataloader:
    #     print("img shape", img.shape)
    #     print("landmark size", landmark.size())
    #     print("attrbute size", attribute.size())
    #     print("euler_angle", euler_angle.size())

    # step 4: run
    # if args.test_initial:
    #     logging.info('Testing from initial')
    #     validate(val_loader, model, criterion, args.start_epoch)

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma) # ! change learning
        train(dataloader, model, criterion, optimizer, epoch)

    #     filename = f'{args.snapshot}_checkpoint_epoch_{epoch}.pth.tar'
    #     save_checkpoint(
    #         {
    #             'epoch': epoch,
    #             'state_dict': model.state_dict(),
    #         },
    #         filename
    #     )
    #     validate(val_loader, model, criterion, epoch)

def parse_args():
    parser = argparse.ArgumentParser(description='Trainging Template')
    # general
    # parser.add_argument('-j', '--workers', default=8, type=int)
    # parser.add_argument('--devices_id', default='0', type=str)
    # parser.add_argument('--test_initial', default='false', type=str2bool)

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.00001, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)

    # -- lr
    parser.add_argument("--lr_step_size", default=20, type=int)
    parser.add_argument("--lr_gamma", default=0.1, type=int)
    # -- resume log and checkpoint 
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    # parser.add_argument('--snapshot', default='./test', type=str, metavar='PATH')
    parser.add_argument('--log_file', default="train.logs", type=str)

    # --dataset
    parser.add_argument('--dataroot', default='data/train_data/list.txt', type=str, metavar='PATH')
    # parser.add_argument('--train_batchsize', default=56, type=int)
    # parser.add_argument('--val_batchsize', default=8, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=100, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)



