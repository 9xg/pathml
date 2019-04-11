import numpy as np
import pyvips as pv
from skimage.transform import downscale_local_mean
from skimage.filters import threshold_triangle
from skimage.morphology import binary_dilation, remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray, rgb2lab


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

    def __init__(self, slideFilePath, verbose=False):
        # pv.cache_set_max(0)
        # pv.leak_set(True)

        self.__verbose = verbose
        self.__slideFilePath = slideFilePath
        if self.__verbose: print(self.__verbosePrefix + "Attempting to load " + self.__slideFilePath)
        try:
            self.slide = pv.Image.new_from_file(self.__slideFilePath, level=0)
        except:
            raise FileNotFoundError('Whole-slide image could not be loaded')
        else:
            if self.__verbose: print(self.__verbosePrefix + "Successfully loaded")

        try:
            self.slideProperties = {x: self.slide.get(x) for x in self.slide.get_fields()}
        except:
            raise ImportError('Whole-slide image properties could not be imported')
        else:
            if self.slideProperties['vips-loader'] != 'openslideload': raise TypeError(
                'This image is not compatible. Please refer to the documentation for proper installation of openslide and libvips')
            if self.__verbose: print(
                self.__verbosePrefix + str(len(self.slideProperties)) + " properties were successfully imported")
            if self.__verbose: print(
                self.__verbosePrefix + "This slide was digitized on a scanner by " + self.slideProperties[
                    'openslide.vendor'].capitalize() + " at an objective power of " + self.slideProperties[
                    'openslide.objective-power'])

    def setTileProperties(self, tileSize, tileOverlap=0, padEdgeTiles=False, unit='px'):
        # TODO: Implement units for tile size selection
        # TODO: Implement padding of boundary tiles
        # TODO: Tile overlap not supported yey
        self.padEdgeTiles = padEdgeTiles
        self.tileOverlap = 0
        self.tileSize = tileSize
        self.tileMetadata = {}
        # Create tile adresses and coordinates
        self.numTilesInX = self.slide.width // (self.tileSize - self.tileOverlap)
        self.numTilesInY = self.slide.height // (self.tileSize - self.tileOverlap)
        for y in range(self.numTilesInY):
            for x in range(self.numTilesInX):
                self.tileMetadata[(x, y)] = {'x': x * (self.tileSize - self.tileOverlap),
                                             'y': y * (self.tileSize - self.tileOverlap), 'width': self.tileSize,
                                             'height': self.tileSize}

    def detectForeground(self, foregroundThreshold=False, hardSegmentation=False, level=2):
        if not hasattr(self, 'tileMetadata'): raise PermissionError(
            'setTileProperties must be called before foreground detection')
        # get low-level magnification
        self.lowMagSlide = pv.Image.new_from_file(self.__slideFilePath, level=level)
        # smallerImage = self.slide.resize(0.002)
        self.lowMagSlide = np.ndarray(buffer=self.lowMagSlide.write_to_memory(),
                                      dtype=self.__format_to_dtype[self.lowMagSlide.format],
                                      shape=[self.lowMagSlide.height, self.lowMagSlide.width, self.lowMagSlide.bands])
        self.lowMagSlide = rgb2lab(self.lowMagSlide[:,:,0:3])[:,:,0]

        downsampleFactor = round(float(self.slideProperties['openslide.level[' + str(level) + '].downsample']))
        localTileSize = round(self.tileSize / downsampleFactor)
        #localTileOverlap = round(self.tileOverlap / downsampleFactor)
        self.lowMagSlide = self.lowMagSlide[0:(self.slide.height // (self.tileSize - self.tileOverlap) * localTileSize),
                           0:(self.slide.width // (self.tileSize - self.tileOverlap) * localTileSize)]
        self.lowMagSlide = downscale_local_mean(rgb2gray(self.lowMagSlide), (localTileSize, localTileSize), cval=1)

        # TODO: This has to be cleaned!!!
        binarizationTh = threshold_triangle(self.lowMagSlide)
        binarizationTh = foregroundThreshold if foregroundThreshold else binarizationTh
        lowMagSlideBin = self.lowMagSlide < binarizationTh

        minBinSize = 8 if hardSegmentation else 2
        lowMagSlideBin = binary_fill_holes(remove_small_objects(lowMagSlideBin, minBinSize))
        lowMagSlideBin = binary_dilation(lowMagSlideBin) if hardSegmentation else lowMagSlideBin

        self.lowMagSlideBin = lowMagSlideBin

        tmpForegroundCount = 0
        for key, value in self.tileMetadata.items():
            lookupTile = lowMagSlideBin[int(key[1]), int(key[0])]
            self.tileMetadata[key].update({'foreground': lookupTile})
            if lookupTile: tmpForegroundCount = tmpForegroundCount + 1

        self.foregroundTileCount = tmpForegroundCount
        ## INTEGRITY CHECK
        # integrityCheck = np.ones(lowMagSlideBin.shape, dtype=bool)
        # for key, value in self.tileMetadata.items():
        #    integrityCheck[int(key[1]),int(key[0])] = value['foreground']
        # print(np.array_equal(lowMagSlideBin, integrityCheck))
        return True

    def getTile(self, tileAddress):
        if not hasattr(self, 'tileMetadata'): raise PermissionError(
            'setTileProperties must be called before accessing tiles')
        if len(tileAddress)==2 and isinstance(tileAddress, tuple):
            if self.numTilesInX >= tileAddress[0] and self.numTilesInY >= tileAddress[1]:
                tmpTile = self.slide.extract_area(self.tileMetadata[tileAddress]['x'],self.tileMetadata[tileAddress]['y'],self.tileMetadata[tileAddress]['width'],self.tileMetadata[tileAddress]['height'])
                return np.ndarray(buffer=tmpTile.write_to_memory(), dtype=self.__format_to_dtype[tmpTile.format], shape=[tmpTile.height, tmpTile.width, tmpTile.bands])
            else:
                raise ValueError('Tile address ('+str(tileAddress[0])+', '+str(tileAddress[1])+') is out of bounds')

    def saveTile(self, tileAddress, fileName):
        if not hasattr(self, 'tileMetadata'): raise PermissionError(
            'setTileProperties must be called before accessing tiles')
        if len(tileAddress)==2 and isinstance(tileAddress, tuple):
            if self.numTilesInX >= tileAddress[0] and self.numTilesInY >= tileAddress[1]:
                tmpTile = self.slide.extract_area(self.tileMetadata[tileAddress]['x'],self.tileMetadata[tileAddress]['y'],self.tileMetadata[tileAddress]['width'],self.tileMetadata[tileAddress]['height'])
                tmpTile.write_to_file(fileName)
                #return np.ndarray(buffer=tmpTile.write_to_memory(), dtype=self.__format_to_dtype[tmpTile.format], shape=[tmpTile.height, tmpTile.width, tmpTile.bands])
            else:
                raise ValueError('Tile address ('+str(tileAddress[0])+', '+str(tileAddress[1])+') is out of bounds')

    def appendTag(self, tileAddress, key, val):
        self.tileMetadata[tileAddress][key] = val

# TODO: a check tileaddress function
    def iterateTiles(self, excludeBackground = True, includeImage = False):
        for key, value in self.tileMetadata.items():
            if value['foreground']==True:
                if includeImage:
                    yield key,self.getTile(key)
                else:
                    yield key

    def segmentForegroundTiles(self, segmentationData=False):
        if type(segmentationData).__name__ == "SlideAnnotation":
            pass
        elif 1==2:
            pass
        else:
            pass
        pass

    def getTileCount(self):
        if not hasattr(self, 'tileMetadata'): raise PermissionError(
            'setTileProperties must be called before tile counting')
        return len(self.tileMetadata)
