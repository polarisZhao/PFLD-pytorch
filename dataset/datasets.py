import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader

class WLFWDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.path = None
        self.landmarks = None
        self.attribute = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()
        
    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])
        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)
        self.attribute = np.asarray(self.line[197:203], dtype=np.int32)
        self.euler_angle = np.asarray(self.line[203:206], dtype=np.float32)
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmark, self.attribute, self.euler_angle)

    def __len__(self):
        return len(self.lines)

if __name__ == '__main__':
    file_list = './data/test_data/list.txt'
    wlfwdataset = WLFWDatasets(file_list)
    dataloader = DataLoader(wlfwdataset, batch_size=256, shuffle=True, num_workers=0, drop_last=False)
    for img, landmark, attribute, euler_angle in dataloader:
        print("img shape", img.shape)
        print("landmark size", landmark.size())
        print("attrbute size", attribute)
        print("euler_angle", euler_angle.size())
