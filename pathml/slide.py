import numpy as np
import pyvips as pv
# import multiprocessing as mp
from skimage.transform import downscale_local_mean
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.morphology import binary_dilation, remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray, rgb2lab
from tqdm import tqdm


##
# TODO: Slide annotations
##

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
            if self.__verbose:
                print(
                    self.__verbosePrefix + "This slide was digitized on a scanner by " + self.slideProperties[
                        'openslide.vendor'].capitalize() + " at an objective power of " + self.slideProperties[
                        'openslide.objective-power'])

    def integrityCheck(self):
        pass

    def setTileProperties(self, tileSize, tileOverlap=0, padEdgeTiles=False, unit='px'):
        # TODO: Implement units for tile size selection
        # TODO: Implement padding of boundary tiles
        self.padEdgeTiles = padEdgeTiles
        self.tileOverlap = round(tileOverlap * tileSize)
        self.tileSize = tileSize
        self.tileMetadata = {}
        # Create tile adresses and coordinates
        self.numTilesInX = self.slide.width // (
            self.tileSize - self.tileOverlap)
        if self.numTilesInX * self.tileSize > self.slide.width: self.numTilesInX -= 1
        self.numTilesInY = self.slide.height // (
            self.tileSize - self.tileOverlap)
        if self.numTilesInY * self.tileSize > self.slide.height: self.numTilesInY -= 1

        for y in range(self.numTilesInY):
            for x in range(self.numTilesInX):
                self.tileMetadata[(x, y)] = {'x': x * (self.tileSize - self.tileOverlap),
                                             'y': y * (self.tileSize - self.tileOverlap), 'width': self.tileSize,
                                             'height': self.tileSize}

    def foregroundMask():
        pass

    def detectForeground(self, threshold=False, level=2):
        if not hasattr(self, 'tileMetadata'):
            raise PermissionError(
                'setTileProperties must be called before foreground detection')
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

        self.foregroundTileAddresses = []
        for tileAddress in self.iterateTiles():
            tileXPos = round(self.tileMetadata[tileAddress]['x'] * (1 / downsampleFactor))
            tileYPos = round(self.tileMetadata[tileAddress]['y'] * (1 / downsampleFactor))
            tileWidth = round(self.tileMetadata[tileAddress]['width'] * (1 / downsampleFactor))
            tileHeight = round(self.tileMetadata[tileAddress]['height'] * (1 / downsampleFactor))
            localTmpTile = self.lowMagSlide[tileYPos:tileYPos + tileHeight, tileXPos:tileXPos + tileWidth]
            localTmpTileMean = np.nanmean(localTmpTile)
            if localTmpTileMean < threshold:
                self.tileMetadata[tileAddress].update({'foreground': True})
                self.foregroundTileAddresses.append(tileAddress)
            else:
                self.tileMetadata[tileAddress].update({'foreground': False})
        return True

    def getTile(self, tileAddress, writeToNumpy=False):
        if not hasattr(self, 'tileMetadata'):
            raise PermissionError(
                'setTileProperties must be called before accessing tiles')
        if len(tileAddress) == 2 and isinstance(tileAddress, tuple):
            if self.numTilesInX >= tileAddress[0] and self.numTilesInY >= tileAddress[1]:
                tmpTile = self.slide.extract_area(self.tileMetadata[tileAddress]['x'], self.tileMetadata[tileAddress]
                                                  ['y'], self.tileMetadata[tileAddress]['width'], self.tileMetadata[tileAddress]['height'])
                if writeToNumpy:
                    # Usingh writeToNumpy = True requires significant memory overhead as the tile is copied to memory
                    return np.ndarray(buffer=tmpTile.write_to_memory(), dtype=self.__format_to_dtype[tmpTile.format], shape=[tmpTile.height, tmpTile.width, tmpTile.bands])
                else:
                    return tmpTile
            else:
                raise ValueError(
                    'Tile address (' + str(tileAddress[0]) + ', ' + str(tileAddress[1]) + ') is out of bounds')

    def ind2sub(self, tileIndex, foregroundOnly=False):
        if foregroundOnly:
            return self.foregroundTileAddresses[tileIndex]
        else:
            return np.unravel_index(tileIndex, (self.numTilesInX, self.numTilesInY), order='F')

    def saveTile(self, tileAddress, fileName, folder):
        if not hasattr(self, 'tileMetadata'):
            raise PermissionError(
                'setTileProperties must be called before accessing tiles')
        if len(tileAddress) == 2 and isinstance(tileAddress, tuple):
            if self.numTilesInX >= tileAddress[0] and self.numTilesInY >= tileAddress[1]:
                tmpTile = self.slide.extract_area(self.tileMetadata[tileAddress]['x'], self.tileMetadata[tileAddress]
                                                  ['y'], self.tileMetadata[tileAddress]['width'], self.tileMetadata[tileAddress]['height'])

                tmpTile.write_to_file(folder + fileName)
                # return np.ndarray(buffer=tmpTile.write_to_memory(), dtype=self.__format_to_dtype[tmpTile.format], shape=[tmpTile.height, tmpTile.width, tmpTile.bands])
            else:
                raise ValueError(
                    'Tile address (' + str(tileAddress[0]) + ', ' + str(tileAddress[1]) + ') is out of bounds')

    def appendTag(self, tileAddress, key, val):
        self.tileMetadata[tileAddress][key] = val

    def thumbnail(self):
        self.lowMagSlide = pv.Image.new_from_file(
            self.__slideFilePath, level=2)
        # smallerImage = self.slide.resize(0.002)
        self.lowMagSlide = np.ndarray(buffer=self.lowMagSlide.write_to_memory(),
                                      dtype=self.__format_to_dtype[self.lowMagSlide.format],
                                      shape=[self.lowMagSlide.height, self.lowMagSlide.width, self.lowMagSlide.bands])
        return self.lowMagSlide

# TODO: a check tileaddress function
    def iterateTiles(self, includeImage=False, writeToNumpy=False):
        for key, value in self.tileMetadata.items():
            # if value['foreground']==True: Inplement exclude background
            if includeImage:
                yield key, self.getTile(key,writeToNumpy=writeToNumpy)
            else:
                yield key

    def segmentForegroundTiles(self, segmentationData=False):
        if type(segmentationData).__name__ == "SlideAnnotation":
            pass
        elif 1 == 2:
            pass
        else:
            pass
        pass

    def getTileCount(self, foregroundOnly=False):
        if not hasattr(self, 'tileMetadata'):
            raise PermissionError(
                'setTileProperties must be called before tile counting')
        if foregroundOnly:
            return len(self.foregroundTileAddresses)
        else:
            return len(self.tileMetadata)
