from PIL import Image
from torch.utils.data import Dataset


class WholeSlideImageDataset(Dataset):
    """WholeSlideImage dataset."""

    def __init__(self, slideClass, foregroundOnly=False, transform=None):
        self.slideClass = slideClass
        self.foregroundOnly = foregroundOnly
        self.transform = transform

    def __len__(self):
        return self.slideClass.getTileCount(foregroundOnly=self.foregroundOnly)

    def __getitem__(self, idx):
        tileAddress = self.slideClass.ind2sub(idx, foregroundOnly=self.foregroundOnly)
        img = Image.fromarray(self.slideClass.getTile(
            tileAddress, writeToNumpy=True)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'tileAddress': tileAddress}

        return sample
