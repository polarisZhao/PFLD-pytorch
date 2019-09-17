import argparse
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2

from models.pfld import PFLDInference, AuxiliaryNet
from mtcnn.detector import detect_faces, show_bboxes

def main(args):
    checkpoint = torch.load(args.model_path)
    plfd_backbone = PFLDInference().cuda()
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.cuda()
    transform = transforms.Compose([ transforms.ToTensor()])

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret: break

        bounding_boxes, landmarks = detect_faces(img)

        maxval_index = 0
        if(len(bounding_boxes) > 0):
            maxval_index = np.argmax(bounding_boxes[:,4])

        if(len(bounding_boxes) > maxval_index - 1 and len(bounding_boxes) > 0):
            face = img[int(bounding_boxes[maxval_index][1]):int(bounding_boxes[maxval_index][3]), int(bounding_boxes[maxval_index][0]):int(bounding_boxes[maxval_index][2])]
            if face.size == 0:
                continue
            face_resized = cv2.resize(face, dsize=(112, 112), interpolation=cv2.INTER_LINEAR)
            face_resized= cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_resized = transform(face_resized).unsqueeze(0).cuda()
            _, landmarks = plfd_backbone(face_resized)
            pre_landmark = landmarks.cpu().detach().numpy()[0].reshape(-1, 2) * [int(bounding_boxes[maxval_index][2]) - int(bounding_boxes[maxval_index][0]), int(bounding_boxes[maxval_index][3]) - int(bounding_boxes[maxval_index][1])]

            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(face, (x, y), 1, (255,0,0),-1)

            cv2.imshow("xxxx", face)
            if cv2.waitKey(10) == 27:
                break


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="./checkpoint/snapshot/checkpoint.pth.tar", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)