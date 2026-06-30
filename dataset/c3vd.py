import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

class C3VD(Dataset):
    def __init__(self, filelist_path, mode, size=(644, 518)):

        self.mode = mode
        self.size = size

        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose([
                                     Resize(
                                         width=net_w,
                                         height=net_h,
                                         resize_target=True,
                                         keep_aspect_ratio=False,
                                         ensure_multiple_of=16,
                                         resize_method='lower_bound',
                                         image_interpolation_method=cv2.INTER_CUBIC,
                                     ),
                                     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                     PrepareForNet(),
                                 ])

    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        depth /= 255
        depth *= 100

        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])

        sample['valid_mask'] = (torch.isnan(sample['depth']) == 0)
        sample['depth'][sample['valid_mask'] == 0] = 0

        sample['image_path'] = self.filelist[item].split(' ')[0]

        return sample

    def __len__(self):
        return len(self.filelist)