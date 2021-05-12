from PIL import Image
from torch.utils.data import Dataset


class WholeSlideImageDataset(Dataset):
    """WholeSlideImage dataset."""

    def __init__(self, slideClass, foregroundOnly=False, tissueLevelThreshold=False, foregroundLevelThreshold=False, transform=None):
        self.slideClass = slideClass
        self.foregroundOnly = foregroundOnly
        self.tissueLevelThreshold = tissueLevelThreshold
        self.foregroundLevelThreshold = foregroundLevelThreshold
        self.transform = transform
        self.suitableTileAddresses = self.slideClass.suitableTileAddresses(tissueLevelThreshold=self.tissueLevelThreshold,
                                                                            foregroundLevelThreshold=self.foregroundLevelThreshold)

        #for tA in self.slideClass.iterateTiles():
        #    if self.tissueLevelThreshold and self.foregroundLevelThreshold:
        #        if (self.slideClass.tileDictionary[tA]['tissueLevel'] >= self.tissueLevelThreshold) and (self.slideClass.tileDictionary[tA]['foregroundLevel'] <= self.foregroundLevelThreshold):
        #            self.suitableTileAddresses.append(tA)
        #    elif self.tissueLevelThreshold and not self.foregroundLevelThreshold:
        #        if (self.slideClass.tileDictionary[tA]['tissueLevel'] >= self.tissueLevelThreshold):
        #            self.suitableTileAddresses.append(tA)
        #    elif self.foregroundLevelThreshold and not self.tissueLevelThreshold:
        #        if (self.slideClass.tileDictionary[tA]['foregroundLevel'] <= self.foregroundLevelThreshold):
        #            self.suitableTileAddresses.append(tA)
        #    else:
        #        self.suitableTileAddresses.append(tA)

    def __len__(self):
        if self.tissueLevelThreshold or self.foregroundLevelThreshold:
            return len(self.suitableTileAddresses)
        else:
            return self.slideClass.getTileCount(foregroundOnly=self.foregroundOnly)

    def __getitem__(self, idx):
        if self.tissueLevelThreshold or self.foregroundLevelThreshold:
            tileAddress = self.suitableTileAddresses[idx]
        else:
            tileAddress = self.slideClass.ind2sub(idx, foregroundOnly=self.foregroundOnly)

        img = Image.fromarray(self.slideClass.getTile(
            tileAddress, writeToNumpy=True)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'tileAddress': tileAddress}

        return sample
