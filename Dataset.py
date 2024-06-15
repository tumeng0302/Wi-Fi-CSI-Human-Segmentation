from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import signal
from PIL import Image
import numpy as np
import torch
import json

def get_data_list(split: str, data_root: str):
    valid_type = ['train', 'train_1', 'train_2', 'train_3', 'val', 'test', 'occu']
    if split not in valid_type:
        raise ValueError(f'Invalid split type: {split}, should be one of {valid_type}')
    
    with open(f'{data_root}/CSI_data_split.json', 'r') as f:
        data_list = json.load(f)
    
    if split == 'train':
        data_list = data_list['train_0'] + data_list['train_1'] + data_list['train_2'] + data_list['train_3']
    elif split in ['train_1', 'train_2', 'train_3']:
        data_list = data_list['train_0'] + data_list[split]
    else:
        data_list = data_list[split]

    return data_list

def gen_beta(n, rand):
    beta = (torch.eye(n) * rand)[:, :-1]
    beta_ = torch.cat((torch.zeros(1, n-1), beta[:-1, :]))
    beta_[beta_!=0] = 1 - beta_[beta_!=0]
    beta += beta_
    return beta

def interpolation(amp, pha):
    rand = torch.rand(26, 26)
    beta1 = gen_beta(26, rand)
    beta2 = gen_beta(26, rand)
    for i in range(4):
        amp_front, amp_behind = amp[:26, i, :], amp[25:, i, :]
        pha_front, pha_behind = pha[:26, i, :], pha[25:, i, :]
        amp[:, i, :] = torch.cat((
            torch.matmul(amp_front.T, beta1).T, 
            amp[25, i, :].unsqueeze(0), 
            torch.matmul(amp_behind.T, beta2).T), 0)
        pha[:, i, :] = torch.cat((
            torch.matmul(pha_front.T, beta1).T, 
            pha[25, i, :].unsqueeze(0), 
            torch.matmul(pha_behind.T, beta2).T), 0)
    return amp, pha

class Maks_Dataset(Dataset):
    def __init__(self, data_root, split='train', crop_size=(192, 256)):
        """
        Args:
            data_root: str, path to the dataset
            split: str, should be in: ['train', 'train_1', 'train_2', 'train_3', 'val', 'test', 'occu'], default 'train'
            crop_size: tuple, size of the image. default (192, 256)
        """
        self.data_root = data_root
        self.data_list = get_data_list(split, data_root)
        self.split = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                crop_size, interpolation=InterpolationMode.NEAREST),
        ])
        self.norm = transforms.Normalize(mean=[0.], std=[1.])
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index) -> torch.Tensor:
        data_path = self.data_list[index].replace('npy', 'img')
        mask = Image.open(f'{self.data_root}/{data_path}_mask.png').convert('L')
        mask = self.transform(mask).float()
        return mask
    
class CSI_Dataset(Dataset):
    def __init__(self, 
                 data_root: str, split: str = 'train', 
                 crop_size: tuple = (192, 256), interpolation: float = -1,
                 amp_offset: float = 60000, pha_offset: float = 28000):
        """
        Args:
            data_root: str, path to the dataset
            split: str, should be in: ['train', 'train_1', 'train_2', 'train_3', 'val', 'test', 'occu'], default 'train'
            crop_size: tuple, size of the image. default (192, 256)
            interpolation: 
                float, the probability of use interpolation to augment the data
                set to -1 to disable interpolation. default -1
            amp_offset: int, the offset of amplitude data. default 60000
            pha_offset: int, the offset of phase data. default 28000

        Note:
            Value of amp_offset and pha_offset are refered to the CSI data parsing method.
            Modify amp_offset and pha_offset if you use your own CSI data.
        """
        self.data_root = data_root
        self.data_list = get_data_list(split, data_root)
        self.split = split
        self.b_cha, self.a_cha = signal.butter(5, 100, 'lowpass', fs=2025)
        self.b_tem, self.a_tem = signal.butter(5, 90, 'lowpass', fs=500)
        self.data_struct = {}
        for data in self.data_list:
            env = data.split('/')[0]
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

    def __len__(self):
        return len(self.data_list)
    
    def normalize(self, x: torch.Tensor, mean: float = 0, std: float = 0.5):
        return ((x - x.mean()) / x.std()) * std + mean

    def denoise(self, x: np.ndarray, type: str):
        x_filt = signal.filtfilt(self.b_cha, self.a_cha, x, axis=2)
        x_filt = signal.filtfilt(self.b_tem, self.a_tem, x_filt, axis=0)
        if type == 'amp':
            x_filt[:, :, 1001:1024] = 0
            x_filt[:, :, 1997:] = 0
        elif type == 'pha':
            x_filt[:, :, 1997:] = 0
        return x_filt

    def get_data(self, mask_path, data_path, npy_path)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = Image.open(f'{self.data_root}/{mask_path}_mask.png').convert('L')
        mask = self.transform(mask).float()
        amplist, phalist = [], []
        for npy in enumerate(npy_path):
            data = np.load(f'{self.data_root}/{data_path+npy}.npz')
            amplist.append(data['mag'].astype(np.float32)/self.amp_offset)
            phalist.append(data['pha'].astype(np.float32)/self.pha_offset)
        amp, pha = np.concatenate(amplist, axis=0), np.concatenate(phalist, axis=0)
        amp, pha = torch.from_numpy(self.denoise(amp, 'amp')), torch.from_numpy(self.denoise(pha, 'pha'))
        amp, pha = self.normalize(amp), self.normalize(pha)
        if self.split == 'train' and np.random.random() < self.interpolation:
            amp, pha = interpolation(amp, pha)
        return amp, pha, mask

    def __getitem__(self, idx):
        choose = np.random.random()
        data_path = self.data_list[idx][0]
        env = data_path.split('/')[0]
        mask_path = data_path.replace('npy', 'img') + self.data_list[2]
        amp, pha, mask = self.get_data(mask_path, data_path, self.data_list[1:])
        label = 0

        if self.split == 'train':
            if choose < 0.5:
                data_path2 = np.random.choice(self.data_struct[env])
                mask_path2 = data_path2.replace('npy', 'img')
                amp2, pha2, mask2 = self.get_data(mask_path2, data_path2)
                label = 1

            if choose >= 0.5:
                data_struct = self.data_struct.copy()
                data_struct.pop(env)
                env2 = np.random.choice(list(data_struct.keys()))
                data_path2 = np.random.choice(data_struct[env2])
                mask_path2 = data_path2.replace('npy', 'img')
                amp2, pha2, mask2 = self.get_data(mask_path2, data_path2)
                label = -1
            
            return [[amp, pha, mask], [amp2, pha2, mask2]], torch.tensor(label)
        
        else:
            return amp, pha, mask