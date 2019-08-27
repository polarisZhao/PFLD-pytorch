# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil
import sys
debug = False
sys.path.append("/home/zhaozhichao/Documents/github/PFLD-pytorch")
from utils import calculate_pitch_yaw_roll


def rotate_landmark(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.array([[alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
                  [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]])
    align_landmark = np.hstack((landmark, np.ones((landmark.shape[0], 1))))
    return M, np.dot(align_landmark, M.T)


class WFLWData():
    def __init__(self, line, imgdir, image_size=112):
        self.line = line.strip().split()
        # 0   - 195: landmark
        # 196 - 199: bbox
        # 200: pose          0-> normal pose          1->large pose
        # 201: expression    0-> normal expression    1->exaggerate expression
        # 202: illumination  0-> normal illumination  1->extreme illumination
        # 203: make-up       0-> no make-up           1->make-up
        # 204: occlusion     0-> no occlusion         1->occlusion
        # 205: blur          0-> clear                1->blur
        # 206: imgname
        assert (len(self.line) == 207)
        self.landmark = np.asarray(list(map(float, self.line[:196])),
                                   dtype=np.float32).reshape(-1, 2)
        self.box = np.asarray(list(map(int, self.line[196:200])),
                              dtype=np.int32)
        self.pose = map(bool, self.line[200])
        self.expression = map(bool, self.line[201])
        self.illumination = map(bool, self.line[202])
        self.make_up = map(bool, self.line[203])
        self.occlusion = map(bool, self.line[204])
        self.blur = map(bool, self.line[205])
        self.path = os.path.join(imgdir, self.line[206])

        self.image_size = image_size

        self.img = None
        self.imgs = []
        self.landmarks = []
        self.boxes = []

        self.mirror_idx = [
            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44,
            43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52,
            53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63,
            62, 61, 60, 67, 66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84,
            83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96
        ]

    def load_data(self, is_train, repeat, mirror=None):
        img = cv2.imread(self.path)  
        height, width, _ = img.shape

        xy_min = np.min(self.landmark, axis=0).astype(np.int32)
        xy_max = np.max(self.landmark, axis=0).astype(np.int32) 
        wh = xy_max - xy_min + 1 
        center = (xy_min + wh / 2).astype(np.int32)  
        boxsize = int(np.max(wh) * 1.2)  
        xy_min = center - boxsize // 2
        x1, y1 = xy_min
        x2, y2 = xy_min + boxsize

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx,
                                      cv2.BORDER_CONSTANT, 0)

        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.landmark + 0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        if is_train:
            imgT = cv2.resize(
                imgT,
                (self.image_size, self.image_size)) 
        landmark = (self.landmark - xy_min) / boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)
        if is_train:
            while len(self.imgs) < repeat:
                angle = np.random.randint(-20, 20)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate_landmark(angle, (cx, cy), self.landmark)

                imgT = cv2.warpAffine(
                    img, M, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)),
                                         np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size // 2),
                                dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx,
                                              cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:, 0] = 1 - landmark[:, 0]
                    landmark = landmark[self.mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        TRACKED_POINTS = [
            33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16
        ]
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (98, 2)
            save_path = os.path.join(path, prefix + '_' + str(i) + '.png')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)

            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append(lanmark[index])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape(
                (-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(
                euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(
                list(map(str,
                         lanmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(save_path, landmark_str,
                                           attributes_str, euler_angles_str)
            labels.append(label)
        return labels


def get_dataset_list(landmarkDir, outdir, imgDir, is_train):
    """
    """
    with open(landmarkDir, 'r') as f:
        lines = f.readlines()
        labels = []

        save_img = os.path.join(outdir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        for i, line in enumerate(lines):
            Img = WFLWData(line, imgDir)

            img_name = Img.path
            Img.load_data(is_train, 10)
            # print("img_name", img_name)

            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)

            label_txt = Img.save_data(save_img,
                                      str(i) + '_' + filename)

            labels.append(label_txt) 
            if ((i + 1) % 100) == 0: 
                print('file: {}/{}'.format(i + 1, len(lines)))

    with open(os.path.join(outdir, 'list.txt'), 'w') as f:
        for label in labels:
            f.writelines(label)


if __name__ == "__main__":
    landmark_prefix = './data/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_'
    landmark_ext = '.txt'

    data_root = os.path.dirname(os.path.realpath(__file__))
    image_dirs = os.path.join(data_root, 'WFLW_images')

    process_types = ['train', 'test']
    for process_type in process_types:
        print("process {0:}".format(process_type))

        landmark_dir = landmark_prefix + str(process_type) + landmark_ext

        outdir = os.path.join(data_root, str(process_type) + "_data")
        # print(outdir)
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)

        is_train = True if (process_type == 'train') else False
        imgs = get_dataset_list(landmark_dir, outdir, image_dirs, is_train)

        print("process finished: {0:}".format(process_type))
