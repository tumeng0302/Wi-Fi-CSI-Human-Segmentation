import sys
sys.path.append('/root/Wi-Fi-CSI-Human-Segmentation')
from Dataset import interpolation, get_data_list
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import torch
from PIL import Image
import pyopenpose as op
import cv2
import json

def get_data_list(split: str, data_root: str, mode: str = 'merge'):
    valid_type = ['train', 'train_1', 'train_2', 'train_3', 'val', 'test', 'occu', 'random_1', 'random_2', 'random_3']
    if split not in valid_type:
        raise ValueError(f'Invalid split type: {split}, should be one of {valid_type}')
    
    with open(f'{data_root}/CSI_data_{mode}.json', 'r') as f:
        data_list = json.load(f)
    
    if split == 'train':
        data_list = data_list['train_0'] + data_list['train_1'] + data_list['train_2'] + data_list['train_3']
    elif split in ['train_1', 'train_2', 'train_3']:
        data_list = data_list['train_0'] + data_list[split]
    else:
        data_list = data_list[split]

    return data_list

class PersonInWifiDataset(Dataset):
    def __init__(self, 
                 data_root: str, split: str = 'train_1', 
                 crop_size: tuple = (192, 256), interpolation: float = -1,
                 amp_offset: float = 60000, pha_offset: float = 28000,):
        super().__init__()
        self.data_root = data_root
        self.crop_size = crop_size
        self.data_list = get_data_list(split, data_root, mode='merge')
        self.split = split
        self.data_struct = {}
        for data in self.data_list:
            env = data[0].split('/')[0]
            if env not in self.data_struct:
                self.data_struct[env] = []
            self.data_struct[env].append(data)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                crop_size, interpolation=InterpolationMode.NEAREST),
        ])
        self.interpolation = interpolation
        self.amp_offset = amp_offset
        self.pha_offset = pha_offset

    def normalize(self, x: torch.Tensor, mean: float = 0, std: float = 0.5):
        return ((x - x.mean()) / x.std()) * std + mean  

    def get_data(self, img_path, data_path, npy_path)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mask
        mask = Image.open(f'{self.data_root}/{img_path}_mask.png').convert('L')
        mask = self.transform(mask).float()
        # Skeleton
        img = cv2.imread(f'{self.data_root}/{img_path}.jpg')
        img = cv2.resize(img, self.crop_size[::-1])
        # CSI
        data = np.load(f'{self.data_root}/{data_path}/{npy_path[1]}.npz')
        amp = torch.from_numpy(data['mag'].astype(np.float32)/self.amp_offset)
        pha = torch.from_numpy(data['pha'].astype(np.float32)/self.pha_offset)
        amp, pha = self.normalize(amp), self.normalize(pha)

        return amp, pha, mask, img

    def __getitem__(self, idx):
        data_path = self.data_list[idx][0]
        img_path = data_path.replace('npy', 'img') + '/' + self.data_list[idx][2]
        amp, pha, mask, img = self.get_data(img_path, data_path, self.data_list[idx][1:])    
        
        return amp, pha, mask, img
    
    def __len__(self):
        return len(self.data_list)
    
if __name__ == '__main__':
    dataset = PersonInWifiDataset(data_root='/root/CSI_Dataset')
    print(dataset[0])