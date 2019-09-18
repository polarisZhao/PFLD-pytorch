import argparse
import logging
from pathlib import Path
import time

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

from torch.utils import data
from torch.utils.data import DataLoader

from dataset.datasets import WLFWDatasets
from models.pfld import PFLDInference, AuxiliaryNet
from pfld.loss import PFLDLoss
from pfld.utils import AverageMeter

import cv2

def validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet):
    plfd_backbone.eval()
    auxiliarynet.eval()

    with torch.no_grad():
        losses = []
        losses_ION = []

        for img, landmark_gt, attribute_gt, euler_angle_gt in wlfw_val_dataloader:
            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            attribute_gt.requires_grad = False
            attribute_gt = attribute_gt.cuda(non_blocking=True)

            landmark_gt.requires_grad = False
            landmark_gt = landmark_gt.cuda(non_blocking=True)

            euler_angle_gt.requires_grad = False
            euler_angle_gt = euler_angle_gt.cuda(non_blocking=True)

            plfd_backbone = plfd_backbone.cuda()
            auxiliarynet = auxiliarynet.cuda()

            _, landmarks = plfd_backbone(img)

            loss = torch.mean(
                torch.sqrt(torch.sum((landmark_gt - landmarks)**2, axis=1))
                )

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()
            error_diff = np.sum(np.sqrt(np.sum((landmark_gt - landmarks) ** 2, axis=2)), axis=1)
            interocular_distance = np.sqrt(np.sum((landmarks[:, 60, :] - landmarks[:,72, :]) ** 2, axis=1))
            # interpupil_distance = np.sqrt(np.sum((landmarks[:, 60, :] - landmarks[:, 72, :]) ** 2, axis=1))
            error_norm = np.mean(error_diff / interocular_distance)
        losses.append(loss.cpu().numpy())
        losses_ION.append(error_norm)

        print("NME", np.mean(losses))
        print("ION", np.mean(losses_ION))


def main(args):
    checkpoint = torch.load(args.model_path)

    plfd_backbone = PFLDInference().cuda()
    auxiliarynet = AuxiliaryNet().cuda()

    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    auxiliarynet.load_state_dict(checkpoint['auxiliarynet'])

    transform = transforms.Compose([transforms.ToTensor()])

    wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset, batch_size=8, shuffle=False, num_workers=0)

    validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet)

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="./checkpoint/snapshot/checkpoint.pth.tar", type=str)
    parser.add_argument('--test_dataset', default='./data/test_data/list.txt', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)