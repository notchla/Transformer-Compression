from torch.utils.data import Dataset
import os
from PIL import Image

class DIV2K(Dataset):
    def __init__(self, split, data_root, downsampled=True):
        super().__init__()
        assert split in ['train','val'], "Unknown split"

        self.root =  os.path.join(data_root,'DIV2K')
        self.img_channels = 3
        self.fnames = []
        self.file_type = '.png'
        self.size = (768, 512)

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
        return img
