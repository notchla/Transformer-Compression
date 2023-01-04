from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop
from torchvision.transforms.functional import crop
import torch
import numpy as np

class DIV2K(Dataset):
    def __init__(self, split, data_root, downsampled=True, random_crop=None, whole_image = False,):
        super().__init__()
        assert split in ['train','val'], "Unknown split"

        self.root =  os.path.join(data_root,'DIV2K')
        self.img_channels = 3
        self.fnames = []
        self.file_type = '.png'
        self.size = (768, 512)
        self.patch_size = random_crop
        self.whole_image = whole_image
        self.random_crop = RandomCrop(random_crop) if random_crop else None
        self.img_resolution = (random_crop, random_crop) if random_crop else self.size

        if split == 'train':
            for i in range(0,800):
                self.fnames.append("DIV2K_train_HR/{:04d}.png".format(i+1))
        elif split == 'val':
            for i in range(800,900):
                self.fnames.append("DIV2K_valid_HR/{:04d}.png".format(i+1))
        self.downsampled = downsampled

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions
            if height > width: img = img.rotate(90, expand=1)
            img.thumbnail(self.size, Image.ANTIALIAS)
        if self.random_crop:
            img = self.random_crop(img)
        return img
    
    def get_fnames(self, indexes):
        return [self.fnames[i] for i in indexes]

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

class DIV2K_evaluation(Dataset):
    def __init__(self, split, data_root, idx = None):
            super().__init__()
            assert split in ['train','val'], "Unknown split"

            self.root =  os.path.join(data_root,'DIV2K')
            self.img_channels = 3
            self.fnames = []
            self.file_type = '.png'
            self.size = (768, 512)
            self.patch_size = 32
            self.img_resolution = self.size

            if split == 'train':
                for i in range(0,800):
                    if idx is not None and idx != i: continue
                    self.fnames.append("DIV2K_train_HR/{:04d}.png".format(i+1))
            elif split == 'val':
                for i in range(800,900):
                    if idx is not None and idx != i - 800: continue
                    self.fnames.append("DIV2K_valid_HR/{:04d}.png".format(i+1))
            
            self.patches = []
            for idx in range(len(self.fnames)):
                path = os.path.join(self.root, self.fnames[idx])
                img = Image.open(path)
                width, height = self.img_resolution
                top, left = (0, 0)
                while True:
                    cropped = crop(img, top, left, 32, 32)
                    self.patches.append(cropped)
                    left += 32
                    if left >= width:
                        left = 0
                        top += 32
                        if top >= height:
                            break

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    def get_fnames(self, indexes):
        return [self.fnames[i//384] for i in indexes]
        
class KODAK_evaluation(Dataset):
    def __init__(self, data_root, idx = None):
            super().__init__()

            self.root =  os.path.join(data_root,'KODAK')
            self.img_channels = 3
            self.fnames = []
            self.file_type = '.png'
            self.size = (768, 512)
            self.patch_size = 32
            self.img_resolution = self.size

            for i in range(0,24):
                    if idx is not None and idx != i: continue
                    self.fnames.append("kodim{:02d}.png".format(i+1))
                    
            
            self.patches = []
            for idx in range(len(self.fnames)):
                path = os.path.join(self.root, self.fnames[idx])
                img = Image.open(path)
                width, height = img.size  # Get dimensions
                if height > width: img = img.rotate(90, expand=1)
                width, height = self.img_resolution
                top, left = (0, 0)
                while True:
                    cropped = crop(img, top, left, 32, 32)
                    self.patches.append(cropped)
                    left += 32
                    if left >= width:
                        left = 0
                        top += 32
                        if top >= height:
                            break

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    def get_fnames(self, indexes):
        return [self.fnames[i//384] for i in indexes]

class CoordDataset(Dataset):
    def __init__(self, dataset, img_resolution):


        self.transform = Compose([
            Resize(img_resolution),
            ToTensor(),
            # Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.dataset = dataset
        self.mgrid = get_mgrid(img_resolution)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx])

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords': self.mgrid, "img" : img}
        gt_dict = {'img': img}

        return in_dict, gt_dict



class CIFAR10(Dataset):
    """
    Download data from: https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
    """
    def __init__(self, split, data_root, downsampled=True, random_crop=None, conditional = None):
        super().__init__()
        assert split in ['train','val'], "Unknown split"

        self.root =  os.path.join(data_root,'cifar10')
        self.img_channels = 3
        self.fnames = []
        self.file_type = '.png'
        self.size = (32, 32)
        self.patch_size = random_crop
        self.random_crop =  None
        self.img_resolution = (random_crop, random_crop) if random_crop else self.size

        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        if conditional is not None:
            for el in conditional:
                assert el in classes, 'Unknown class (avilable: ["airplane", "automobile","bird","cat","deer","dog","frog","horse","ship","truck"])'
            classes = conditional

        if split == 'train':
            for im_class in classes:
                for img in os.listdir(os.path.join(self.root, "train", im_class)):
                    self.fnames.append(f"train/{im_class}/{img}")
        elif split == 'val':
            for im_class in classes:
                for img in os.listdir(os.path.join(self.root, "test", im_class)):
                    self.fnames.append(f"test/{im_class}/{img}")
        self.downsampled = False

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions
            if height > width: img = img.rotate(90, expand=1)
            img.thumbnail(self.size, Image.ANTIALIAS)
        if self.random_crop:
            img = self.random_crop(img)
        return img

    def get_fnames(self, indexes):
        return [self.fnames[i] for i in indexes]
