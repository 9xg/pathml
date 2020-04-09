import numpy as np
import pyvips as pv
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.morphology import binary_dilation, remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray, rgb2lab
from tqdm import tqdm
import os
import pickle


##
# TODO: Slide annotations
##
# EXPERIMENTAL
def unwrap_self(arg, **kwarg):
    return Slide.square_int(*arg, **kwarg)
# EXPERIMENTAL END
class Slide:
    __format_to_dtype = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
        'complex': np.complex64,
        'dpcomplex': np.complex128,
    }

    __dtype_to_format = {
        'uint8': 'uchar',
        'int8': 'char',
        'uint16': 'ushort',
        'int16': 'short',
        'uint32': 'uint',
        'int32': 'int',
        'float32': 'float',
        'float64': 'double',
        'complex64': 'complex',
        'complex128': 'dpcomplex',
    }

    __verbosePrefix = '[PathML] '

    def __init__(self, slideFilePath, level=0, verbose=False):
        # pv.cache_set_max(0)
        # pv.leak_set(True)

        self.__verbose = verbose
        self.__slideFilePath = slideFilePath
        try:
            if self.__verbose:
                print(self.__verbosePrefix + "Loading " + self.__slideFilePath)
            self.slide = pv.Image.new_from_file(
                self.__slideFilePath, level=level)
        except:
            raise FileNotFoundError('Whole-slide image could not be loaded')
        else:
            if self.__verbose:
                print(self.__verbosePrefix + "Successfully loaded")

        try:
            self.slideProperties = {x: self.slide.get(
                x) for x in self.slide.get_fields()}
        except:
            raise ImportError(
                'Whole-slide image properties could not be imported')
        else:
            if self.slideProperties['vips-loader'] != 'openslideload':
                raise TypeError(
                    'This image is not compatible. Please refer to the documentation for proper installation of openslide and libvips')
            if self.__verbose:
                print(
                    self.__verbosePrefix + str(len(self.slideProperties)) + " properties were successfully imported")

# EXPERIMENTAL
    def square_int(self, i):
        return self.getTile((0,0),writeToNumpy=True)

    def run(self, num):
        results = []
        results = Parallel(n_jobs= -1, backend="threading")\
            (delayed(unwrap_self)(i) for i in tqdm(zip([self]*len(num), num), total = len(num)))
        print(results)
# EXPERIMENTAL END
    def setTileProperties(self, tileSize, tileOverlap=0, unit='px'):
        # TODO: Implement units for tile size selection
        # TODO: Implement padding of boundary tiles
        self.regionWorker = pv.Region.new(self.slide)
        self.tileOverlap = round(tileOverlap * tileSize)
        self.tileSize = tileSize
        self.tileDictionary = {}
        # Create tile adresses and coordinates
        self.numTilesInX = self.slide.width // (
            self.tileSize - self.tileOverlap)
        if self.numTilesInX * self.tileSize > self.slide.width: self.numTilesInX -= 1
        self.numTilesInY = self.slide.height // (
            self.tileSize - self.tileOverlap)
        if self.numTilesInY * self.tileSize > self.slide.height: self.numTilesInY -= 1

        for y in range(self.numTilesInY):
            for x in range(self.numTilesInX):
                self.tileDictionary[(x, y)] = {'x': x * (self.tileSize - self.tileOverlap),
                                             'y': y * (self.tileSize - self.tileOverlap), 'width': self.tileSize,
                                             'height': self.tileSize}

    def loadTileDictionary(self, dictionaryFilePath):
        pass

    def detectForeground(self, threshold, level=2):
        if not hasattr(self, 'tileDictionary'):
            raise PermissionError(
                'Tile dictionary has to be created before foreground detection')
        # get low-level magnification
        self.lowMagSlide = pv.Image.new_from_file(
            self.__slideFilePath, level=level)
        # smallerImage = self.slide.resize(0.002)
        self.lowMagSlide = np.ndarray(buffer=self.lowMagSlide.write_to_memory(),
                                      dtype=self.__format_to_dtype[self.lowMagSlide.format],
                                      shape=[self.lowMagSlide.height, self.lowMagSlide.width, self.lowMagSlide.bands])
        self.lowMagSlideRGB = self.lowMagSlide
        self.lowMagSlide = rgb2lab(self.lowMagSlide[:, :, 0:3])[:, :, 0]
        downsampleFactor = self.slide.width / self.lowMagSlide.shape[1]

        if threshold is 'otsu':
            thresholdLevel = threshold_otsu(self.lowMagSlide[self.lowMagSlide < 100])  # Ignores all blank areas introduced by certain scanners
        elif threshold is 'triangle':
            thresholdLevel = threshold_triangle(self.lowMagSlide[self.lowMagSlide < 100])  # Ignores all blank areas introduced by certain scanners
        elif isinstance(threshold, int) or isinstance(threshold, float):
            thresholdLevel = threshold
        else:
            raise ValueError('No threshold specified for foreground segmentation')

        self.foregroundTileAddresses = []
        for tileAddress in self.iterateTiles():
            tileXPos = round(self.tileDictionary[tileAddress]['x'] * (1 / downsampleFactor))
            tileYPos = round(self.tileDictionary[tileAddress]['y'] * (1 / downsampleFactor))
            tileWidth = round(self.tileDictionary[tileAddress]['width'] * (1 / downsampleFactor))
            tileHeight = round(self.tileDictionary[tileAddress]['height'] * (1 / downsampleFactor))
            localTmpTile = self.lowMagSlide[tileYPos:tileYPos + tileHeight, tileXPos:tileXPos + tileWidth]
            localTmpTileMean = np.nanmean(localTmpTile)
            self.tileDictionary[tileAddress].update({'foregroundLevel': localTmpTileMean})
            if localTmpTileMean < thresholdLevel:
                self.tileDictionary[tileAddress].update({'foreground': True})
                self.foregroundTileAddresses.append(tileAddress)
            else:
                self.tileDictionary[tileAddress].update({'foreground': False})
        return True

    def getTile(self, tileAddress, writeToNumpy=False, useFetch=False):
        if not hasattr(self, 'tileDictionary'):
            raise PermissionError(
                'setTileProperties must be called before accessing tiles')
        if len(tileAddress) == 2 and isinstance(tileAddress, tuple):
            if self.numTilesInX >= tileAddress[0] and self.numTilesInY >= tileAddress[1]:
                if useFetch:
                    newTmpTile = self.fetchTile(self.tileDictionary[tileAddress]['width'], self.tileDictionary[tileAddress]['height'],
                    self.tileDictionary[tileAddress]['x'], self.tileDictionary[tileAddress]['y'])
                    if writeToNumpy:
                        return np.ndarray(buffer=newTmpTile, dtype=np.uint8, shape=[self.tileDictionary[tileAddress]['width'], self.tileDictionary[tileAddress]['height'],4])
                    else:
                        return newTmpTile
                else:
                    tmpTile = self.slide.extract_area(self.tileDictionary[tileAddress]['x'], self.tileDictionary[tileAddress]
                                                      ['y'], self.tileDictionary[tileAddress]['width'], self.tileDictionary[tileAddress]['height'])
                    if writeToNumpy:
                        # Usingh writeToNumpy = True requires significant memory overhead as the tile is copied to memory
                        return np.ndarray(buffer=tmpTile.write_to_memory(), dtype=self.__format_to_dtype[tmpTile.format], shape=[tmpTile.height, tmpTile.width, tmpTile.bands])
                    else:
                        return tmpTile
            else:
                raise ValueError(
                    'Tile address (' + str(tileAddress[0]) + ', ' + str(tileAddress[1]) + ') is out of bounds')

    def fetchTile(self, patchWidth, patchHeight, patchX, patchY):
        return self.regionWorker.fetch(patchWidth * patchX, patchHeight * patchY, patchWidth, patchHeight)

    def ind2sub(self, tileIndex, foregroundOnly=False):
        if foregroundOnly:
            return self.foregroundTileAddresses[tileIndex]
        else:
            return np.unravel_index(tileIndex, (self.numTilesInX, self.numTilesInY), order='F')

    def saveTile(self, tileAddress, fileName, folder=os.getcwd()):
        if not hasattr(self, 'tileDictionary'):
            raise PermissionError(
                'Tile dictionary has to be created before foreground detection')
        if len(tileAddress) == 2 and isinstance(tileAddress, tuple):
            if self.numTilesInX >= tileAddress[0] and self.numTilesInY >= tileAddress[1]:
                tmpTile = self.slide.extract_area(self.tileDictionary[tileAddress]['x'], self.tileDictionary[tileAddress]
                                                  ['y'], self.tileDictionary[tileAddress]['width'], self.tileDictionary[tileAddress]['height'])

                tmpTile.write_to_file(os.path.join(folder, fileName))
                # return np.ndarray(buffer=tmpTile.write_to_memory(), dtype=self.__format_to_dtype[tmpTile.format], shape=[tmpTile.height, tmpTile.width, tmpTile.bands])
            else:
                raise ValueError(
                    'Tile address (' + str(tileAddress[0]) + ', ' + str(tileAddress[1]) + ') is out of bounds')

    def saveTileDictionary(self, fileName, folder=os.getcwd()):
        pickle.dump(self.tileDictionary, open(os.path.join(folder, fileName)+'.pml', 'wb'))

    def appendTag(self, tileAddress, key, val):
        self.tileDictionary[tileAddress][key] = val

    def thumbnail(self, level):
        self.lowMagSlide = pv.Image.new_from_file(
            self.__slideFilePath, level=level)
        # smallerImage = self.slide.resize(0.002)
        self.lowMagSlide = np.ndarray(buffer=self.lowMagSlide.write_to_memory(),
                                      dtype=self.__format_to_dtype[self.lowMagSlide.format],
                                      shape=[self.lowMagSlide.height, self.lowMagSlide.width, self.lowMagSlide.bands])
        return self.lowMagSlide

# TODO: a check tileaddress function
    def iterateTiles(self, includeImage=False, writeToNumpy=False):
        for key, value in self.tileDictionary.items():
            # if value['foreground']==True: Inplement exclude background
            if includeImage:
                yield key, self.getTile(key,writeToNumpy=writeToNumpy)
            else:
                yield key

    def getTileCount(self, foregroundOnly=False):
        if not hasattr(self, 'tileDictionary'):
            raise PermissionError(
                'setTileProperties must be called before tile counting')
        if foregroundOnly:
            return len(self.foregroundTileAddresses)
        else:
            return len(self.tileDictionary)
