from PIL import Image
from torch.utils.data import Dataset


class WholeSlideImage(Dataset):
    """WholeSlideImage dataset."""

    def __init__(self, slideClass, tileSize, tileMask=None transform=None):
        self.slideClass = slideClass
        self.tileSize = tileSize
        self.transform = transform

    def __len__(self):
        return self.slideClass.getTileCount()

    def __getitem__(self, idx):
        tileAddress = self.slideClass.ind2sub(idx)
        img = Image.fromarray(self.slideClass.getTile(
            tileAddress, writeToNumpy=True)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'tileAddress': tileAddress}

        return sample
