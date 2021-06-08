# This is the experimental version of slide with annotation and tile extraction
# capabilities

import torch
from torchvision import transforms
import numpy as np
import pyvips as pv
from PIL import Image, ImageDraw
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from skimage.transform import downscale_local_mean
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.morphology import binary_dilation, remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray, rgb2lab
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from pathml.processor import Processor
from pathml.models.tissuedetector import tissueDetector
from pathml.utils.torch.WholeSlideImageDataset import WholeSlideImageDataset
from pathml.utils.torch.dice_loss import dice_coeff
import xml.etree.ElementTree as ET
from shapely import geometry
from shapely.ops import unary_union
from tqdm import tqdm
import random
import json
import os
import sys
import pickle
pv.cache_set_max(0)


# EXPERIMENTAL
def unwrap_self(arg, **kwarg):
    return Slide.square_int(*arg, **kwarg)
# EXPERIMENTAL END
class Slide:
    """The main class of PathML; a representation of whole-slide image containing
    dictionary of tiles, and upon which further analyses are added, including
    but not limited to tissue detection and annotation, and from which tiles
    from whole-slide images can be extracted.

    Args:
        slideFilePath (string): path to a WSI (to make from scratch) or to a .pml file (to reload a saved Slide object, see saveSelf())
        newSlideFilePath (string, optional): if loading a .pml file and the location of the WSI has changed, the new path to WSI can be inputted here
        level (int, optional): the level of the WSI pyramid at which to operate on; 0 is the highest resolution and default and how many levels are present above that depends on the WSI
        verbose (Boolean, optional): whether to output a verbose output. Default is false.
    """

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

    # If slideFilePath can be a path to a WSI (to make from scratch),
    # or a path to a .pml file (to make from a pre-existing pathml Slide saved with saveSelf())
    def __init__(self, slideFilePath, newSlideFilePath=False, level=0, verbose=False):

        self.__verbose = verbose

        if slideFilePath[-4:] == '.pml': # initing from .pml file
            contents = pickle.load(open(slideFilePath, 'rb'))
            if newSlideFilePath:
                self.slideFilePath = newSlideFilePath
            else:
                self.slideFilePath = contents['slideFilePath']
            self.tileDictionary = contents['tileDictionary']
            if "rawTissueDetectionMap" in contents:
                self.rawTissueDetectionMap = contents['rawTissueDetectionMap']
            if 'annotationClassMultiPolygons' in contents:
                self.annotationClassMultiPolygons = contents['annotationClassMultiPolygons']
        else: # initing from WSI file (from scratch)
            self.slideFilePath = slideFilePath
        self.slideFileName = Path(self.slideFilePath).stem

        try:
            if self.__verbose:
                print(self.__verbosePrefix + "Loading " + self.slideFilePath)
            self.slide = pv.Image.new_from_file(
                self.slideFilePath, level=level)
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
            if 'vips-loader' not in self.slideProperties:
                print("Warning: 'vips-loader' not present in slide properties; verify independently that slide has loaded properly.")
            else:
                if self.slideProperties['vips-loader'] != 'openslideload':
                    raise TypeError(
                        'This image is not compatible. Please refer to the documentation for proper installation of openslide and libvips')
            if self.__verbose:
                print(
                    self.__verbosePrefix + str(len(self.slideProperties)) + " properties were successfully imported")

        if slideFilePath[-4:] == '.pml': # setTileProperties if starting from .pml file
            self.regionWorker = pv.Region.new(self.slide)
            self.tileSize = self.tileDictionary[list(self.tileDictionary.keys())[0]]['height']
            xs = []
            ys = []
            for adrs in self.iterateTiles():
                xs.append(adrs[0])
                ys.append(adrs[1])
            self.numTilesInX = len(list(set(xs)))
            self.numTilesInY = len(list(set(ys)))
            #self.setTilePropertiesFromPml(self.tileDictionary)

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
        """A function to set the properties of the tile dictionary in a Slide object.
        Should be the first function called on a newly created Slide object.

        Args:
            tileSize (int): the edge length of each square tile in the requested unit
            tileOverlap (float, optional): the fraction of a tile's edge length that overlaps the left, right, above, and below tiles. Default is 0.
            unit (string, optional): the unit to measure tileSize by. Default is 'px' for pixels and no other units are current supported

        Example:
            pathml_slide.setTileProperties(400)
        """

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
        return self

    # internal function to do some housekeeping if creating Slide from a .pml file
    #def setTilePropertiesFromPml(self):
        #self.tileDictionary = tileDictionary
    #    self.regionWorker = pv.Region.new(self.slide)
    #    self.tileSize = self.tileDictionary[list(self.tileDictionary.keys())[0]]['height']
    #    xs = []
    #    ys = []
    #    for adrs in self.iterateTiles():
    #        xs.append(adrs[0])
    #        ys.append(adrs[1])
    #    self.numTilesInX = len(list(set(xs)))
    #    self.numTilesInY = len(list(set(ys)))

    #def loadTileDictionary(self, dictionaryFilePath):
    #    pass

    def getNonoverlappingSegmentationInferenceArray(self, aggregationMethod='average', padWithZeros='True'):

        if padWithZeros:
            inference_array = np.zeros((self.slide.height, self.slide.width))
        else:
            inference_array = np.empty((self.slide.height, self.slide.width))
            inference_array[:] = np.nan

        point_tuples = []
        for x in range(self.slide.width):
            for y in range(self.slide.height):
                point_tuples.append((x,y))

        pixels = pd.DataFrame({'xy_tuple': point_tuples,
                                'x': [point_tuple[0] for point_tuple in point_tuples],
                                'y': [point_tuple[1] for point_tuple in point_tuples]})
        pixels = geopandas.GeoDataFrame(pixels, geometry=geopandas.points_from_xy(pixels.x, pixels.y))

        tiles =

        for y in range(self.slide.height):
            for x in range(self.slide.width):
                inference_array[y,x] = self._inferencePixelValue(y, x, aggregationMethod=aggregationMethod)

        return inference_array

    def _inferencePixelValue(self, y, x, aggregationMethod='average'):

        # find which tiles in tile dictionary overlap pixel at height y and width x



    def detectForeground(self, level=4, overwriteExistingForegroundDetection=False, threshold=None):
        """A function to implement traditional foreground filtering methods on the
        tile dictionary to exclude background tiles from subsequent operations.

        Args:
            level (int, optional): the level of the WSI pyramid to detect foreground on. Default is 4. Not all WSIs will have a 4th level, so alter if necessary. If memory runs out, increase the level to detect foreground with a less high resolution image.
            overwriteExistingForegroundDetection (Boolean, optional): whether to old foreground detection if it is present in the tile dictionary already. Default is False.
            threshold (string or int): Legacy argument, avoid using. Default is to put the results of all tissue detection methods (Otsu, triangle, simple thresholding) in the tile dictionary. Can be set to 'otsu', 'triangle' or an int to do simple darkness thresholding at that int value (tiles with a 0-100 foregroundLevel value less or equal to than the set value are considered foreground, where 0 is a pure black tile, 100 is a pure white tile)

        Example:
            pathml_slide.detectForeground()
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before foreground detection')
        if not overwriteExistingForegroundDetection and 'foregroundLevel' in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
            raise Warning('Foreground already detected. Use overwriteExistingForegroundDetection to write over old detections.')
        # get low-level magnification
        self.lowMagSlide = pv.Image.new_from_file(
            self.slideFilePath, level=level)
        # smallerImage = self.slide.resize(0.002)
        self.lowMagSlide = np.ndarray(buffer=self.lowMagSlide.write_to_memory(),
                                      dtype=self.__format_to_dtype[self.lowMagSlide.format],
                                      shape=[self.lowMagSlide.height, self.lowMagSlide.width, self.lowMagSlide.bands])
        self.lowMagSlideRGB = self.lowMagSlide
        self.lowMagSlide = rgb2lab(self.lowMagSlide[:, :, 0:3])[:, :, 0]
        downsampleFactor = self.slide.width / self.lowMagSlide.shape[1]

        #if threshold is 'otsu':
        thresholdLevelOtsu = threshold_otsu(self.lowMagSlide[self.lowMagSlide < 100])  # Ignores all blank areas introduced by certain scanners
        #elif threshold is 'triangle':
        thresholdLevelTriangle = threshold_triangle(self.lowMagSlide[self.lowMagSlide < 100])  # Ignores all blank areas introduced by certain scanners
        #elif isinstance(threshold, int) or isinstance(threshold, float):
        #    thresholdLevel = threshold
        #else:
        #    raise ValueError('No threshold specified for foreground segmentation')

        if type(threshold) in [int, float]:
            thresholdIsNumeric = True
        else:
            thresholdIsNumeric = False
        self.foregroundTileAddresses = []
        for tileAddress in self.iterateTiles():
            tileXPos = round(self.tileDictionary[tileAddress]['x'] * (1 / downsampleFactor))
            tileYPos = round(self.tileDictionary[tileAddress]['y'] * (1 / downsampleFactor))
            tileWidth = round(self.tileDictionary[tileAddress]['width'] * (1 / downsampleFactor))
            tileHeight = round(self.tileDictionary[tileAddress]['height'] * (1 / downsampleFactor))
            localTmpTile = self.lowMagSlide[tileYPos:tileYPos + tileHeight, tileXPos:tileXPos + tileWidth]
            localTmpTileMean = np.nanmean(localTmpTile)

            self.tileDictionary[tileAddress].update({'foregroundLevel': localTmpTileMean})
            if threshold and thresholdIsNumeric:
                if localTmpTileMean <= threshold:
                    self.tileDictionary[tileAddress].update({'foreground': True})
                    self.foregroundTileAddresses.append(tileAddress)
                else:
                    self.tileDictionary[tileAddress].update({'foreground': False})

            if localTmpTileMean <= thresholdLevelOtsu:
                self.tileDictionary[tileAddress].update({'foregroundOtsu': True})
                #self.otsuForegroundTileAddresses.append(tileAddress)
                if threshold and threshold == 'otsu':
                    self.tileDictionary[tileAddress].update({'foreground': True})
                    self.foregroundTileAddresses.append(tileAddress)
            else:
                self.tileDictionary[tileAddress].update({'foregroundOtsu': False})
                if threshold and threshold == 'otsu':
                    self.tileDictionary[tileAddress].update({'foreground': False})

            if localTmpTileMean <= thresholdLevelTriangle:
                self.tileDictionary[tileAddress].update({'foregroundTriangle': True})
                #self.triangleForegroundTileAddresses.append(tileAddress)
                if threshold and threshold == 'triangle':
                    self.tileDictionary[tileAddress].update({'foreground': True})
                    self.foregroundTileAddresses.append(tileAddress)
            else:
                self.tileDictionary[tileAddress].update({'foregroundTriangle': False})
                if threshold and threshold == 'triangle':
                    self.tileDictionary[tileAddress].update({'foreground': False})

        return True

    def getTile(self, tileAddress, writeToNumpy=False, useFetch=False):
        """A function to return a desired tile in the tile dictionary in the
        form of a pyvips Image.

        Args:
            tileAddress (tuple of ints): the (x, y) coordinate touple of the desired tile to extract.
            writeToNumpy (Boolean, optional): whether to return a numpy array of the tile (otherwise a pyvips Image object will be returbed). Default is False.
            useFetch (Boolean, optional): whether to use pyvip's fetchTile() function to extract the tile, which is purported to be faster than extractArea(). Default is False.

        Example:
            pathml_slide.getTile((15,20))
        """

        if not self.hasTileDictionary():
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
        """A function to save a specific tile image to an image file.

        Args:
            tileAddress (tuple of ints): the (x, y) coordinate touple of the desired tile to save.
            fileName (string): the name of the image file including an image extension.
            folder (string, optional): the path to the directory where the tile image will be saved. Default is the current working directory.

        Example:
            pathml_slide.saveTile((15,20), "tile_15x_20y.jpg" folder="/path/to/tiles_directory")
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before saving tile')
        if len(tileAddress) == 2 and isinstance(tileAddress, tuple):
            if self.numTilesInX >= tileAddress[0] and self.numTilesInY >= tileAddress[1]:
                tmpTile = self.slide.extract_area(self.tileDictionary[tileAddress]['x'], self.tileDictionary[tileAddress]
                                                  ['y'], self.tileDictionary[tileAddress]['width'], self.tileDictionary[tileAddress]['height'])

                tmpTile.write_to_file(os.path.join(folder, fileName))
                # return np.ndarray(buffer=tmpTile.write_to_memory(), dtype=self.__format_to_dtype[tmpTile.format], shape=[tmpTile.height, tmpTile.width, tmpTile.bands])
            else:
                raise ValueError(
                    'Tile address (' + str(tileAddress[0]) + ', ' + str(tileAddress[1]) + ') is out of bounds')


    def saveTileDictionary(self, fileName=False, folder=os.getcwd()):
        """A function to save just the tileDictionary attribute of a Slide
        object into a pickled file. Note that these pickled files cannot be used
        as an input when initializing a Slide object; please use saveSelf()
        instead.

        Args:
            fileName (string, optional): the name of the file where the pickled tile dictionary (a dict) will be stored, excluding an extension. Default is the slideFileName attribute.
            folder (string, optional): the path to the directory where the pickled tile dictionary will be saved. Default is the current working directory.

        Example:
            pathml_slide.saveTileDictionary(folder="path/to/pathml_tile_dictionaries")
        """

        # get case ID
        if fileName:
            if type(fileName) == str:
                id = fileName
            else:
                raise ValueError('fileName must be a string')
        else:
            id = self.slideFileName

        pickle.dump(self.tileDictionary, open(os.path.join(folder, id)+'.pml', 'wb'))

    def saveSelf(self, fileName=False, folder=os.getcwd()):
        """A function to save a pickled PathML Slide object to a .pml file for re-use later
        (re-loading is performed by providing the path to the .pml file when initializing a Slide object).
        This function should be re-run after each major step in an analysis on a Slide.

        Args:
            fileName (string, optional): the name of the file where the pickled Slide will be stored, excluding an extension. Default is the slideFileName attribute.
            folder (string, optional): the path to the directory where the pickled Slide will be saved. Default is the current working directory.

        Example:
            pathml_slide.saveSelf("pathml_slide" folder="/path/to/pathml_slides")
        """

        if not self.hasTileDictionary():
            raise PermissionError('setTileProperties must be called before saving self')

        # get case ID
        if fileName:
            if type(fileName) == str:
                id = fileName
            else:
                raise ValueError('fileName must be a string')
        else:
            id = self.slideFileName

        if hasattr(self, 'rawTissueDetectionMap'):
            if self.hasAnnotations():
                outputDict = {'slideFilePath': self.slideFilePath, 'tileDictionary': self.tileDictionary, 'rawTissueDetectionMap': self.rawTissueDetectionMap, 'annotationClassMultiPolygons': self.annotationClassMultiPolygons}
            else:
                outputDict = {'slideFilePath': self.slideFilePath, 'tileDictionary': self.tileDictionary, 'rawTissueDetectionMap': self.rawTissueDetectionMap}
        else:
            if self.hasAnnotations():
                outputDict = {'slideFilePath': self.slideFilePath, 'tileDictionary': self.tileDictionary, 'annotationClassMultiPolygons': self.annotationClassMultiPolygons}
            else:
                outputDict = {'slideFilePath': self.slideFilePath, 'tileDictionary': self.tileDictionary}

        pickle.dump(outputDict, open(os.path.join(folder, id)+'.pml', 'wb'))

    def appendTag(self, tileAddress, key, val):
        """A function to add key-value pair of data to a certain tile in the tile dictionary.

        Args:
            tileAddress (tuple of ints): the (x, y) coordinate touple of the desired tile to save.
            key (string): the key to store at the tile address.
            value: the value to store at the key at the tile address.

        Example:
            pathml_slide.appendTag((15,20), "brightness_level", 0.7)
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before appending tag')
        self.tileDictionary[tileAddress][key] = val

    def thumbnail(self, level):
        self.lowMagSlide = pv.Image.new_from_file(
            self.slideFilePath, level=level)
        # smallerImage = self.slide.resize(0.002)
        self.lowMagSlide = np.ndarray(buffer=self.lowMagSlide.write_to_memory(),
                                      dtype=self.__format_to_dtype[self.lowMagSlide.format],
                                      shape=[self.lowMagSlide.height, self.lowMagSlide.width, self.lowMagSlide.bands])
        return self.lowMagSlide

    def hasTileDictionary(self):
        """A function that returns a Boolean of whether the Slide object has a
        tile dictionary by setTileProperties() yet.

        Example:
            pathml_slide.hasTileDictionary()
        """

        return hasattr(self, 'tileDictionary')

    def hasAnnotations(self):
        """A function that returns a Boolean of whether annotations have been
        added to the tile dictionary by addAnnotations() yet.

        Example:
            pathml_slide.hasAnnotations()
        """

        return hasattr(self, 'annotationClassMultiPolygons')

    def hasTissueDetection(self):
        """A function that returns a Boolean of whether deep tissue detections
        have been added to the tile dictionary by detectTissue() yet.

        Example:
            pathml_slide.hasTissueDetection()
        """

        return hasattr(self, 'rawTissueDetectionMap')

    #def hasInferredClassifications(self):
    #    return hasattr(self, 'rawTissueDetectionMap')

# TODO: a check tileaddress function
    def iterateTiles(self, tileDictionary=False, includeImage=False, writeToNumpy=False):
        """A generator function to iterate over all tiles in the tile dictionary,
        returning the tile address or the tile address and the tile image if
        specified with includeImage.

        Args:
            tileDictionary (dict, optional): the tile dictionary to iterate over. Default is the Slide's own tile dictionary.
            includeImage (Boolean, optional): whether to return a numpy array of the tile alongside its address. Default is False.
            writeToNumpy (Boolean, optional): whether to return a numpy array of the tile (if not, a pyvips Image object will be returned) if includeImage is set to True.

        Example:
            for address in pathml_slide.iterateTiles():
                print(address)
        """

        tileDictionaryIterable = self.tileDictionary if not tileDictionary else tileDictionary
        for key, value in tileDictionaryIterable.items():
            # if value['foreground']==True: Inplement exclude background
            if includeImage:
                yield key, self.getTile(key,writeToNumpy=writeToNumpy)
            else:
                yield key

    def getTileCount(self, foregroundLevelThreshold=False, tissueLevelThreshold=False, foregroundOnly=False):
        """A function that returns the number of tiles in the tile dictionary.
        Arguments can be used to find the number of tiles with desired
        characteristics in the tile dictionary.

        Args:
            foregroundLevelThreshold (string or int, optional): returns the number of tiles considered foreground if 'otsu' or 'triangle' is used, or the number of tiles at or above the minimum threshold specified if simple average darkness intensity foreground filtering was used (0 is a pure black tile, 100 is a pure white tile). Default is not to filter the tile count this way. detectForeground() must be run first.
            tissueLevelThreshold (float, optional): returns the number of tiles at or above the deep tissue detector tissue probability specified. Default is not to filter the tile count this way. detectTissue() must be run first.
            foregroundOnly (Boolean, optional): Legacy argument, avoid using. Whether to return the count of only the number of foreground tiles found with detectForeground(). Only available if threshold argument was used when detectForeground() was called.

        Example:
            pathml_slide.getTileCount()
        """

        if foregroundLevelThreshold or tissueLevelThreshold:
            return len(self.suitableTileAddresses(foregroundLevelThreshold=foregroundLevelThreshold, tissueLevelThreshold=tissueLevelThreshold))
        elif foregroundOnly:
            if hasattr(self, 'foregroundTileAddresses'):
                return len(self.foregroundTileAddresses)
            else:
                raise PermissionError('foregroundOnly cannot be defined unless the threshold argument was used when detectForeground() was called. Note that the use of foregroundOnly is a legacy feature and not recommended.')
        else:
            return len(self.suitableTileAddresses(foregroundLevelThreshold=foregroundLevelThreshold, tissueLevelThreshold=tissueLevelThreshold))
            #raise PermissionError('At least one of arguments foregroundLevelThreshold, tissueLevelThreshold, foregroundOnly must be defined.')

    # ADAM EXPERIMENTAL
    def addAnnotations(self, annotationFilePath, classesToAdd=False, negativeClass=False, level=0,
        overwriteExistingAnnotations=False, mergeOverlappingAnnotationsOfSameClass=True, acceptMultiPolygonAnnotations=False):
        """A function that adds the overlap between all (desired) classes present in an annotation file and each tile in the tile dictionary
        Annotations within groups in ASAP are taken to be within one class, where the name of the ASAP group is the name of the class; similarly,
        annotations within classes in QuPath are taken to be within one class, where the name of the QuPath class is the name of the class.
        (except the negativeClass if one is specified). Acceptable ASAP annotation tools to make annotations for this function include the
        RectangleAnnotation, PolyAnnotation, and SplineAnnotation tools; in QuPath, acceptable tools include the Rectangle, Ellipse, Polygon,
        and Brush tools. Annotations should be polygons, i.e. closed regions that do not self-overlap at any point. Annotations of different
        classes are expected to never overlap, and annotations of the same class can only overlap (and will be merged into one polygon) if
        mergeOverlappingAnnotationsOfSameClass is set to True.

        Args:
            annotationFilePath (string): the path to the file containing the annotation. The file must be either an xml file from the ASAP software or a GeoJSON file from the QuPath software.
            classesToAdd (list of strings, optional): a list of classes to add from the annotation file. Default is that all annotation classes will be used.
            negativeClass (string, optional): the name of the class of negative annotations (donut holes) to subtract from the other annotations. Default is not to consider any class to be a negative space class.
            level (int, optional): the level of the WSI pyramid to make use of. Default is 0.
            overwriteExistingAnnotations (Boolean, optional): whether to overwrite any preexisting annotations in the tile dictionary. Default is False.
            mergeOverlappingAnnotationsOfSameClass (Boolean, optional): whether to automatically merge annotations of the same class that overlap into one polygon. Default is True.
            acceptMultiPolygonAnnotations (Boolean, optional): whether or not to accept annotations that parse into MultiPolygons using Shapely. Default is False and users are strongly discouraged from setting it to True.

        Example:
            pathml_slide.addAnnotations("/path/to/annotations.xml", negativeClass="negative")
        """

        #if fileType != ['asap', 'Asap', 'ASAP', 'qupath', 'Qupath', 'QuPath', 'QUPATH']:
        #    raise ValueError('fileType must be ASAP or QuPath')
        #if fileType in ['qupath', 'Qupath', 'QuPath', 'QUPATH']: # REMOVE ONCE FIXED
        #    raise ValueError('QuPath annotation files not currently supported')
        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before adding annotations')

        foundOverlap = False
        for k in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
            if 'Overlap' in k:
                foundOverlap = True
                if not overwriteExistingAnnotations:
                    raise Warning('Annotations have already been added to the tile dictionary. Use overwriteExistingAnnotations if you wish to write over them')
        if foundOverlap:
            for address in self.iterateTiles():
                for key in self.tileDictionary[address].copy():
                    if 'Overlap' in key:
                        del self.tileDictionary[address][key]

        if (not type(level) == int) or (level < 0):
            raise ValueError('level must be an integer 0 or greater')
        if 'openslide.level['+str(level)+'].downsample' not in self.slideProperties:
            raise ValueError('level not present in slide')
        if (classesToAdd) and (not isinstance(classesToAdd, list)):
            raise ValueError("classestoAdd must a list")
        if not os.path.isfile(annotationFilePath):
            raise FileNotFoundError('Annotation file could not be loaded')

        # Check if annotationFilePath points to an ASAP xml file or a QuPath GeoJSON file
        with open(annotationFilePath) as unknownFile:
            c = unknownFile.read(1)
            if c == '<':
                fileType = 'asap_xml'
            else:
                fileType = 'qupath_geojson'

        slideHeight = int(self.slideProperties['height'])
        annotationScalingFactor = float(self.slideProperties['openslide.level[0].downsample'])/float(self.slideProperties['openslide.level['+str(level)+'].downsample'])
        print("Scale: "+str(annotationScalingFactor))

        classMultiPolys = {}
        classPolys = {}
        negativePolys = []

        # Turn ASAP xml annotations into a dict of multipolygons, with one multipolygon for each class
        if fileType == 'asap_xml':
            try:
                tree = ET.parse(annotationFilePath)
            except:
                raise ImportError('Annotation file is not an xml file')
            root = tree.getroot() # Get root of .xml tree
            if not root.tag == "ASAP_Annotations": # Check whether we actually deal with an ASAP .xml file
                raise ImportError('Annotation file is not an ASAP xml file')

            allAnnotations = root.find('Annotations') # Find all annotations for this slide
            print('xml file valid - ' + str(len(allAnnotations)) + ' annotation(s) found.') # Display number of found annotations

            # Iterate over all annotations to collect annotations in the same class
            #classPolys = {}
            #negativePolys = []
            for annotation in allAnnotations:

                annotationClass = annotation.attrib['PartOfGroup']

                if (classesToAdd) and (annotationClass not in classesToAdd):
                    if (negativeClass):
                        if (annotationClass != negativeClass):
                            print("Skipping an annotation which doesn't appear in classesToAdd or in negativeClass")
                            continue
                    else:
                        print("Skipping an annotation which doesn't appear in classesToAdd")
                        continue

                if annotationClass not in classPolys:
                    if (negativeClass) and (annotationClass != negativeClass):
                        classPolys[annotationClass] = []

                annotationTree = annotation.find('Coordinates')
                polygon = []
                for coordinate in annotationTree:
                    info = coordinate.attrib
                    polygon.append((float(info['X'])*annotationScalingFactor, float(info['Y'])*annotationScalingFactor))
                polygonNp = np.asarray(polygon)
                polygonNp[:,1] = slideHeight-polygonNp[:,1]
                try:
                    poly = geometry.Polygon(polygonNp).buffer(0)
                except:
                    raise ValueError('Annotation cannot be made into polygon(s)')

                # Make sure the annotation produced a polygon
                if poly.geom_type != 'Polygon':
                    if poly.geom_type == 'MultiPolygon':
                        if not acceptMultiPolygonAnnotations:
                            for i, compPoly in enumerate(list(poly)):
                                print('Component polygon '+str(i+1)+' centroid / area: ('+str(compPoly.centroid.x)+', '+str(slideHeight-compPoly.centroid.y)+') / '+str(compPoly.area))
                            raise ValueError('Annotation with centroid ('+str(poly.centroid.x)+', '+str(slideHeight-poly.centroid.y)+
                                ') produces a Shapely '+poly.geom_type+' instead of a polygon; check to see if it self-intersects. See above for the centroids of the component Polygons of the MultiPolygon.')
                    else:
                        raise ValueError('Annotation with centroid ('+str(poly.centroid.x)+', '+str(slideHeight-poly.centroid.y)+
                            ') produces a Shapely '+poly.geom_type+' instead of a polygon; check to see if it self-intersects.')

                if (negativeClass) and (annotationClass == negativeClass):
                    negativePolys.append(poly)
                else:
                    if poly.geom_type == 'MultiPolygon':
                        for componentPoly in list(poly):
                            classPolys[annotationClass].append(componentPoly)
                    else:
                        classPolys[annotationClass].append(poly)

        # Turn QuPath GeoJSON annotations into a dict of multipolygons, with one multipolygon for each class
        else:
            with open(annotationFilePath) as f:
                allAnnotations = json.load(f)
            if not isinstance(allAnnotations, list):
                raise Warning('GeoJSON file does not have an outer list structure')
            else:
                print('JSON file valid - ' + str(len(allAnnotations)) + ' annotation(s) found.')

            for annotation in allAnnotations:
                if annotation['geometry']['type'] != 'Polygon':
                    raise ValueError('Found annotation that was not a polygon in JSON file')

                try:
                    annotationClass = annotation['properties']['classification']['name']
                except:
                    raise ValueError('Found QuPath annotation without a class; all annotations must be assigned to a class')

                if (classesToAdd) and (annotationClass not in classesToAdd):
                    if (negativeClass):
                        if (annotationClass != negativeClass):
                            print("Skipping an annotation which doesn't appear in classesToAdd or in negativeClass")
                            continue
                    else:
                        print("Skipping an annotation which doesn't appear in classesToAdd")
                        continue

                if annotationClass not in classPolys:
                    if (negativeClass) and (annotationClass != negativeClass):
                        classPolys[annotationClass] = []

                if len(annotation['geometry']['coordinates']) > 1:
                    raise ValueError('Multiple sets of coordinates found for an annotation')
                annotationCoordinates = annotation['geometry']['coordinates'][0]
                polygon = []
                for coordinate in annotationCoordinates:
                    x_coord = coordinate[0]
                    y_coord = coordinate[1]
                    polygon.append((float(x_coord)*annotationScalingFactor, float(y_coord)*annotationScalingFactor))
                polygonNp = np.asarray(polygon)
                polygonNp[:,1] = slideHeight-polygonNp[:,1]
                try:
                    poly = geometry.Polygon(polygonNp).buffer(0)
                except:
                    raise ValueError('Annotation cannot be made into a polygon')

                # Make sure the annotation produced a polygon
                if poly.geom_type != 'Polygon':
                    if poly.geom_type == 'MultiPolygon':
                        if not acceptMultiPolygonAnnotations:
                            for i, compPoly in enumerate(list(poly)):
                                print('Component polygon '+str(i+1)+' centroid / area: ('+str(compPoly.centroid.x)+', '+str(slideHeight-compPoly.centroid.y)+') / '+str(compPoly.area))
                            raise ValueError('Annotation with centroid ('+str(poly.centroid.x)+', '+str(slideHeight-poly.centroid.y)+
                                ') produces a Shapely '+poly.geom_type+' instead of a polygon; check to see if it self-intersects. See above for the centroids of the component Polygons of the MultiPolygon.')
                    else:
                        raise ValueError('Annotation with centroid ('+str(poly.centroid.x)+', '+str(slideHeight-poly.centroid.y)+
                            ') produces a Shapely '+poly.geom_type+' instead of a polygon; check to see if it self-intersects.')

                if (negativeClass) and (annotationClass == negativeClass):
                    negativePolys.append(poly)
                else:
                    if poly.geom_type == 'MultiPolygon':
                        for componentPoly in list(poly):
                            classPolys[annotationClass].append(componentPoly)
                    else:
                        classPolys[annotationClass].append(poly)


        # Make a Shapely MultiPolygon for each class
        #classMultiPolys = {ancl:geometry.MultiPolygon(ply) for (ancl,ply) in classPolys.items()}
        if mergeOverlappingAnnotationsOfSameClass:
            classMultiPolys = {ancl:unary_union(ply_list) for (ancl,ply_list) in classPolys.items()}
        else:
            classMultiPolys = {ancl:geometry.MultiPolygon(ply) for (ancl,ply) in classPolys.items()}

        # If desired, merge any polygons of the same class that overlap
        #if mergeOverlappingAnnotationsOfSameClass:
        #    for ancl, mply in classMultiPolys.items():
        #        if mpl

        if (negativeClass) and (len(negativePolys) == 0):
            print('Warning: 0 '+negativeClass+' annotations found, but negativeClass assigned a value')
        elif (negativeClass):
            print(str(len(negativePolys))+' '+negativeClass+' annotation(s) found')

        # Remove negativeClass polygons from each class multipolygon
        for ancl in classMultiPolys:
            for nply in negativePolys:
                if classMultiPolys[ancl].intersects(nply):
                    classMultiPolys[ancl] = classMultiPolys[ancl].difference(nply)

        # Check for overlapping annotations from different classes
        for ancl1, ancl1multipoly in classMultiPolys.items():
            for ancl2, ancl2multipoly in classMultiPolys.items():
                if ancl1 != ancl2:
                    if ancl1multipoly.overlaps(ancl2multipoly):
                        clsIntersection = ancl1multipoly.intersection(ancl2multipoly)
                        raise ValueError('Annotation classes '+ancl1multipoly+' and '+ancl2multipoly+' overlap near ('+str(clsIntersection.centroid.x)+', '+str(slideHeight-clsIntersection.centroid.y)+')')
                else:
                    continue


        # Iterate over all tiles in tile dictionary, marking the overlap of each class MultiPolygon with each tile
        for address in self.iterateTiles():
            x = self.tileDictionary[address]['x']
            y = self.tileDictionary[address]['y']
            height = self.tileDictionary[address]['height']
            tile = geometry.box(x, (slideHeight-y)-height, x+height, slideHeight-y)

            for class_name, class_multipoly in classMultiPolys.items():
                tile_class_overlap = tile.intersection(class_multipoly).area/(height**2)
                self.tileDictionary[address].update({class_name+'Overlap': tile_class_overlap})

        self.annotationClassMultiPolygons = classMultiPolys


    # SEGMENTATION
    # ADAM EXPERIMENTAL
    def getAnnotationTileMask(self, tileAddress, maskClass, writeToNumpy=False, verbose=False, acceptTilesWithoutClass=False):
        """A function that returns the PIL Image of the binary mask of a
        tile-annotation class overlap. Note that the output values are 0 (white)
        to 255 (black).

        Args:
            tileAddress (tuple of ints): the (x, y) coordinate touple of the desired tile to get the annotation mask for.
            maskClass (string): the class to extract a segmentation mask for.
            writeToNumpy (Boolean, optional): whether to return the annotation tile mask in the form of a numpy array instead of a PIL Image. Default is False.
            acceptTilesWithoutClass (Boolean, optional): whether to allow the input of tiles that lack either annotations or annotations with maskClass present. Default is False. If set to True, in cases where tiles lack either annotations or annotations with maskClass present, a blank mask will be returned.
            verbose (Boolean, optional): whether to output verbose messages. Default is False.

        Example:
            pathml_slide.getAnnotationTileMask((15,20), "metastasis")
        """

        height = self.tileDictionary[tileAddress]['height']

        if not self.hasTileDictionary():
            raise PermissionError('setTileProperties must be called before extracting tiles')
        if tileAddress not in self.tileDictionary:
            raise ValueError('tileAddress must be in tileDictionary')
        if not self.hasAnnotations():
            #print("Warning: no annotations found in Slide. All ground truth tile masks in Slide will be be totally absent of "+classToThreshold+" pixels. Returning blank tile mask. Run addAnnotations() if there should be annotations in this Slide.")
            if acceptTilesWithoutClass:
                if writeToNumpy:
                    return np.zeros((height, height))
                    #return np.array(mask.transpose(Image.FLIP_TOP_BOTTOM))
                else:
                    blank_mask = Image.new('1', (height, height), 0)
                    return blank_mask
            else:
                raise PermissionError('addAnnotations must be called before extracting tiles')
        else:
            if maskClass not in self.annotationClassMultiPolygons:
                if acceptTilesWithoutClass:
                    if writeToNumpy:
                        return np.zeros((height, height))
                        #return np.array(mask.transpose(Image.FLIP_TOP_BOTTOM))
                    else:
                        blank_mask = Image.new('1', (height, height), 0)
                        return blank_mask
                else:
                    raise ValueError(maskClass+' not in annotationClassMultiPolygons')

        #print('PERFORMING ACTUAL COMPARISONS')

        slideHeight = int(self.slideProperties['height'])
        x = self.tileDictionary[tileAddress]['x']
        y = self.tileDictionary[tileAddress]['y']
        tileBox = geometry.box(x, (slideHeight-y)-height, x+height, slideHeight-y)

        mask = Image.new('1', (height, height), 0)

        if verbose: print(self.tileDictionary[tileAddress]['x'], self.tileDictionary[tileAddress]['y'])

        if self.annotationClassMultiPolygons[maskClass].geom_type == 'Polygon':
            mask = self._getTileMask(tileBox, self.annotationClassMultiPolygons[maskClass], mask)
        elif self.annotationClassMultiPolygons[maskClass].geom_type == 'MultiPolygon':
            plygns = list(self.annotationClassMultiPolygons[maskClass])
            for plygn in plygns:
                mask = self._getTileMask(tileBox, plygn, mask)
        else:
            raise Warning('The value at key '+maskClass+' in annotationClassMultiPolygons must be Shapely Polygon or MultiPolygon')

        if writeToNumpy:
            return np.array(mask.transpose(Image.FLIP_TOP_BOTTOM))
        else:
            return mask.transpose(Image.FLIP_TOP_BOTTOM)

    # Adds overlap from one annotation to existing mask
    def _getTileMask(self, tile_box, single_annotation, mask, fillPixelValue=1, verbose=False):

        if verbose: print("- - - - - - - Checking overlap with annotation", list(single_annotation.exterior.coords), "- - - - - - - -")
        box_coords = list(tile_box.exterior.coords)
        bottom_left_point_of_tile = box_coords[3] # this point is always the bottom left corners

        intersection = tile_box.intersection(single_annotation)

        if intersection.area > 0:
            if verbose: print("intersection area: ", intersection.area)

            # Single polygon intersection
            if intersection.geom_type == 'Polygon':
                if verbose: print("number of polygons comprising intersection: 1")
                mask_polygon = []
                for point in list(intersection.exterior.coords):
                    mask_polygon.append((point[0]-bottom_left_point_of_tile[0], point[1]-bottom_left_point_of_tile[1]))
                if verbose: print("mask polygon: ", mask_polygon)
                ImageDraw.Draw(mask).polygon(mask_polygon, outline=fillPixelValue, fill=fillPixelValue)

            # Multi-polygon intersection
            elif intersection.geom_type in ['MultiPolygon', 'GeometryCollection']:
                intersecs = list(intersection)
                if verbose: print("number of geometry elements comprising intersection: ", len(intersecs))
                for intersec in intersecs:
                    if intersec.geom_type == 'Polygon':
                        mask_polygon = []
                        for point in list(intersec.exterior.coords):
                            mask_polygon.append((point[0]-bottom_left_point_of_tile[0], point[1]-bottom_left_point_of_tile[1]))
                        if verbose: print("mask polygon: ", mask_polygon)
                        ImageDraw.Draw(mask).polygon(mask_polygon, outline=fillPixelValue, fill=fillPixelValue)
                    else:
                        if verbose:
                            print(intersec.geom_type+' intersection found. Skipping it...')
                        else:
                            pass


            # Non-polygonal intersection (should never happen)
            #elif intersection.geom_type == 'GeometryCollection':
            #    intersecs = list(intersection)
            #    if verbose: print("number of geometry elements comprising intersection: ", len(intersecs))
            #    for intersec in intersecs:

            else:
                raise Warning('Intersection type unknown: '+intersection.geom_type)

        if verbose: print("\n")
        return mask

    #def numberOfSuitableTiles(self, className, tileAnnotationOverlapThreshold=0.5, foregroundLevelThreshold=False, tissueLevelThreshold=False):
    #    if not self.hasTileDictionary():
    #        raise PermissionError(
    #            'setTileProperties must be called before counting suitable tiles')
    #    if not self.hasAnnotations():
    #        raise PermissionError(
    #            'addAnnotations must be called before counting suitable tiles')
    #    if className+'Overlap' not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
    #        raise ValueError(className+' not found in tile dictionary')
    #    print('MADE IT')
    #    #print(list(self.tileDictionary.keys()))
    #    kys = list(self.tileDictionary.keys())
    #    #print(self.tileDictionary)
    #    tileCounter = 0
    #    for address in self.iterateTiles():
    #        print('address', address)
    #        #if self.tileDictionary[address][className+'Overlap'] >= tileAnnotationOverlapThreshold:
    #        print(self.tileDictionary[address][className+'Overlap'])
    #        if (tissueLevelThreshold) and (foregroundLevelThreshold):
    #            if 'tissueLevel' not in self.tileDictionary[address]:
    #                raise PermissionError('Deep tissue detection must be performed with detectTissue() before tissueLevelThreshold can be defined')
    #            if 'foregroundLevel' not in self.tileDictionary[address]:
    #                raise PermissionError('Foreground detection must be performed with detectForeground() before foregroundLevelThreshold can be defined')
    #            if (self.tileDictionary[address]['tissueLevel'] >= tissueLevelThreshold) and (self.tileDictionary[address]['foregroundLevel'] <= foregroundLevelThreshold):
    #                if self.tileDictionary[address][className+'Overlap'] >= tileAnnotationOverlapThreshold:
    #                    tileCounter = tileCounter + 1
#
#            elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
#                if 'tissueLevel' not in self.tileDictionary[address]:
#                    raise PermissionError('Deep tissue detection must be performed with detectTissue() before tissueLevelThreshold can be defined')
#                if self.tileDictionary[address]['tissueLevel'] >= tissueLevelThreshold: # do not extract background and artifact tiles
#                    if self.tileDictionary[address][className+'Overlap'] >= tileAnnotationOverlapThreshold:
#                        tileCounter = tileCounter + 1
#
#            elif (foregroundLevelThreshold) and (not tissueLevelThreshold):
#                if 'foregroundLevel' not in self.tileDictionary[address]:
#                    raise PermissionError('Foreground detection must be performed with detectForeground() before foregroundLevelThreshold can be defined')
#                if self.tileDictionary[address]['foregroundLevel'] <= foregroundLevelThreshold:
#                    if self.tileDictionary[address][className+'Overlap'] >= tileAnnotationOverlapThreshold:
#                        tileCounter = tileCounter + 1
#
#            else:
#                    if self.tileDictionary[address][className+'Overlap'] >= tileAnnotationOverlapThreshold:
#                        tileCounter = tileCounter + 1
#
#            return tileCounter



    # ADAM EXPERIMENTAL
    def extractAnnotationTiles(self, outputDir, tileDirName=False, numTilesToExtractPerClass='all', classesToExtract=False, otherClassNames=False,
        extractSegmentationMasks=False, tileAnnotationOverlapThreshold=0.5, foregroundLevelThreshold=False, tissueLevelThreshold=False,
        returnTileStats=True, returnOnlyNumTilesFromThisClass=False, seed=False):
        """A function to extract tiles that overlap with annotations into
        directory structure amenable to torch.utils.data.ConcatDataset.

        Args:
            outputDir (string): the path to the directory where the tile directory will be stored
            tileDirName (string, optional): what to call the hightest level tile directory that will be created. Default is 'tiles'
            numTilesToExtractPerClass (dict or int or 'all', optional): expected to be positive integer, a dictionary with class names as keys and positive integers as values, or 'all' to extract all suitable tiles for each class. Default is 'all'.
            classesToExtract (string or list of strings, optional): defaults to extracting all classes found in the annotations, but if defined, must be a string or a list of strings of class names.
            otherClassNames (string or list of strings, optional): if defined, creates an empty class directory alongside the unannotated class directory for each class name in the list (or string) for torch ImageFolder purposes
            extractSegmentationMasks (Boolean, optional): whether to extract a 'masks' directory that is exactly parallel to the 'tiles' directory, and contains binary segmentation mask tiles for each class desired. Default is False.
            tileAnnotationOverlapThreshold (float, optional): a number greater than 0 and less than or equal to 1, or a dictionary of such values, with a key for each class to extract. The numbers specify the minimum fraction of a tile's area that overlaps a given class's annotations for it to be extracted. Default is 0.5.
            foregroundLevelThreshold (string or int or float, optional): if defined as an int, only extracts tiles with a 0-100 foregroundLevel value less or equal to than the set value (0 is a black tile, 100 is a white tile). Only includes Otsu's method-passing tiles if set to 'otsu', or triangle algorithm-passing tiles if set to 'triangle'. Default is not to filter on foreground at all.
            tissueLevelThreshold (Boolean, optional): if defined, only extracts tiles with a 0 to 1 tissueLevel probability greater than or equal to the set value. Default is False.
            returnTileStats (Boolean, optional): whether to return the 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation. Default is True.
            returnOnlyNumTilesFromThisClass (string, optional): causes only the number of suitable tiles for the specified class in the slide; no tile images are created if a string is provided. Default is False.
            seed (int, optional): the random seed to use for reproducible anayses. Default is not to use a seed when randomly selecting tiles.

        Example:
            channel_data = pathml_slide.extractAnnotationTiles("/path/to/directory", numTilesToExtractPerClass=200, tissueLevelThreshold=0.995)
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before extracting tiles')
        if not self.hasAnnotations():
            raise PermissionError(
                'addAnnotations must be called before extracting tiles')
        if seed:
            if type(seed) != int:
                raise ValueError('Seed must be an integer')
            random.seed(seed)
        # get case ID
        if tileDirName:
            if type(tileDirName) != str:
                raise ValueError("tileDirName must be a string")
            else:
                id = tileDirName
        else:
            id = self.slideFileName

        # get classes to extract
        extractionClasses = []
        if not classesToExtract:
            for key, value in self.tileDictionary[list(self.tileDictionary.keys())[0]].items():
                if 'Overlap' in key:
                    extractionClasses.append(key)
        elif type(classesToExtract) == list:
            extractionClasses = [classToExtract+'Overlap' for classToExtract in classesToExtract]
            for extractionClass in extractionClasses:
                if extractionClass not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                    raise ValueError(extractionClass+' not found in tile dictionary')
        elif type(classesToExtract) == str:
            extractionClasses = [classesToExtract+'Overlap']
            if extractionClasses[0] not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                raise ValueError(extractionClasses[0]+' not found in tile dictionary')

        else:
            raise ValueError("classesToExtract must be a string or list of strings")
        extractionClasses = [extractionClass.split('Overlap')[0] for extractionClass in extractionClasses]
        print('Found '+str(len(extractionClasses))+' class(es) to extract:', extractionClasses)

        # Convert annotationOverlapThreshold into a dictionary (if necessary)
        annotationOverlapThresholdDict = {}
        if (type(tileAnnotationOverlapThreshold) == int) or (type(tileAnnotationOverlapThreshold) == float):
            if (tileAnnotationOverlapThreshold <= 0) or (tileAnnotationOverlapThreshold > 1):
                raise ValueError('tileAnnotationOverlapThreshold must be greater than 0 and less than or equal to 1')
            for extractionClass in extractionClasses:
                annotationOverlapThresholdDict[extractionClass] = tileAnnotationOverlapThreshold
        elif type(tileAnnotationOverlapThreshold) == dict:
            for ec, taot in tileAnnotationOverlapThreshold.items():
                if ec not in extractionClasses:
                    raise ValueError('Class '+str(ec)+' present as a key in tileAnnotationOverlapThreshold but absent from the tileDictionary')
                if ((type(taot) != int) and (type(taot) != float)) or ((taot <= 0) or (taot > 1)):
                    raise ValueError('Tile annotation overlap threshold of class '+str(ec)+' must be a number greater than zero and less than or equal to 1')
            for extractionClass in extractionClasses:
                if extractionClass not in tileAnnotationOverlapThreshold:
                    raise ValueError('Class '+str(extractionClass)+' present in the tileDictionary but not present as a key in tileAnnotationOverlapThreshold')
            annotationOverlapThresholdDict = tileAnnotationOverlapThreshold
        else:
            raise ValueError('tileAnnotationOverlapThreshold must be a dictionary or number greater than 0 and less than or equal to 1')

        if ((type(tissueLevelThreshold) != int) and (type(tissueLevelThreshold) != float)) or ((tissueLevelThreshold <= 0) or (tissueLevelThreshold > 1)):
            raise ValueError('tissueLevelThreshold must be a number greater than zero and less than or equal to 1')

        # Get tiles to extract
        annotatedTileAddresses = {extractionClass: [] for extractionClass in extractionClasses}

        suitable_tile_addresses = self.suitableTileAddresses(tissueLevelThreshold=tissueLevelThreshold, foregroundLevelThreshold=foregroundLevelThreshold)
        for address in suitable_tile_addresses:
            for extractionClass in extractionClasses:
                if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
                    annotatedTileAddresses[extractionClass].append(address)

        #for address in self.iterateTiles():
        #    #print('address', address)
        #    if (tissueLevelThreshold) and (foregroundLevelThreshold):
        #        if 'tissueLevel' not in self.tileDictionary[address]:
        #            raise PermissionError('Deep tissue detection must be performed with detectTissue() before tissueLevelThreshold can be defined')
        #        if 'foregroundLevel' not in self.tileDictionary[address]:
        #            raise PermissionError('Foreground detection must be performed with detectForeground() before foregroundLevelThreshold can be defined')
        #        if (self.tileDictionary[address]['tissueLevel'] >= tissueLevelThreshold) and (self.tileDictionary[address]['foregroundLevel'] <= foregroundLevelThreshold):
        #            for extractionClass in extractionClasses:
        #                if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
        #                    annotatedTileAddresses[extractionClass].append(address)
#
#            elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
#                if 'tissueLevel' not in self.tileDictionary[address]:
#                    raise PermissionError('Deep tissue detection must be performed with detectTissue() before tissueLevelThreshold can be defined')
#                if self.tileDictionary[address]['tissueLevel'] >= tissueLevelThreshold: # do not extract background and artifact tiles
#                    for extractionClass in extractionClasses:
#                        if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
#                            annotatedTileAddresses[extractionClass].append(address)
#
#            elif (foregroundLevelThreshold) and (not tissueLevelThreshold):
#                if 'foregroundLevel' not in self.tileDictionary[address]:
#                    raise PermissionError('Foreground detection must be performed with detectForeground() before foregroundLevelThreshold can be defined')
#                if self.tileDictionary[address]['foregroundLevel'] <= foregroundLevelThreshold:
#                    for extractionClass in extractionClasses:
#                        if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
#                            annotatedTileAddresses[extractionClass].append(address)
#
#            else:
#                for extractionClass in extractionClasses:
#                    if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
#                        annotatedTileAddresses[extractionClass].append(address)

        annotatedTilesToExtract = {} #{extractionClass: [] for extractionClass in extractionClasses}
        if type(numTilesToExtractPerClass) == int:
            if numTilesToExtractPerClass <= 0:
                raise ValueError('If numTilesToExtractPerClass is an integer, it must be greater than 0')
            for extractionClass in extractionClasses:
                if len(annotatedTileAddresses[extractionClass]) == 0:
                    print('Warning: 0 suitable '+extractionClass+' tiles found')
                if len(annotatedTileAddresses[extractionClass]) < numTilesToExtractPerClass:
                    print('Warning: '+str(len(annotatedTileAddresses[extractionClass]))+' suitable '+extractionClass+' tiles found but requested '+str(numTilesToExtractPerClass)+' tiles to extract. Extracting all suitable tiles...')
                    annotatedTilesToExtract[extractionClass] = annotatedTileAddresses[extractionClass]
                else:
                    annotatedTilesToExtract[extractionClass] = random.sample(annotatedTileAddresses[extractionClass], numTilesToExtractPerClass)
            #annotatedTilesToExtract = {extractionClass: random.sample(annotatedTileAddresses[extractionClass], numTilesToExtractPerClass) for extractionClass in extractionClasses}
        elif numTilesToExtractPerClass == 'all':
            for extractionClass in extractionClasses:
                if len(annotatedTileAddresses[extractionClass]) == 0:
                    print('Warning: 0 suitable '+extractionClass+' tiles found')
                if len(annotatedTileAddresses[extractionClass]) > 500:
                    print('Warning: '+str(len(annotatedTileAddresses[extractionClass]))+' suitable '+extractionClass+' tiles found')
                annotatedTilesToExtract[extractionClass] = annotatedTileAddresses[extractionClass]

        elif type(numTilesToExtractPerClass) == dict:
            for ec,tc in numTilesToExtractPerClass.items():
                if ec not in extractionClasses:
                    raise Warning('Class '+ec+' present as a key in numTilesToExtractPerClass dictionary but absent from the tileDictionary')
            for extractionClass in extractionClasses:
                if len(annotatedTileAddresses[extractionClass]) == 0:
                    print('Warning: 0 suitable '+extractionClass+' tiles found')
                if extractionClass not in numTilesToExtractPerClass:
                    raise ValueError(extractionClass+' not present in the numTilesToExtractPerClass dictionary')
                numTiles = numTilesToExtractPerClass[extractionClass]
                if (type(numTiles) != int) or (numTiles <= 0):
                    raise ValueError(extractionClass+' does not have a positive integer set as its value in the numTilesToExtractPerClass dictionary')
                if len(annotatedTileAddresses[extractionClass]) < numTiles:
                    print('Warning: '+str(len(annotatedTileAddresses[extractionClass]))+' suitable '+extractionClass+' tiles found but requested '+str(numTiles)+' tiles to extract. Extracting all suitable tiles...')
                    annotatedTilesToExtract[extractionClass] = annotatedTileAddresses[extractionClass]
                else:
                    annotatedTilesToExtract[extractionClass] = random.sample(annotatedTileAddresses[extractionClass], numTiles)

        else:
            raise ValueError("numTilesToExtractPerClass must be a positive integer, a dictionary, or 'all'")

        # Create empty class tile directories for extractionClasses with at least one suitable tile
        if not returnOnlyNumTilesFromThisClass:
            for extractionClass,tte in annotatedTilesToExtract.items():
                if len(tte) > 0:
                    try:
                        os.makedirs(os.path.join(outputDir, 'tiles', id, extractionClass), exist_ok=True)
                    except:
                        raise ValueError(os.path.join(outputDir, 'tiles', id, extractionClass)+' is not a valid path')
                    if otherClassNames:
                        if type(otherClassNames) == str:
                            try:
                                os.makedirs(os.path.join(outputDir, 'tiles', id, otherClassNames), exist_ok=True)
                            except:
                                raise ValueError(os.path.join(outputDir, 'tiles', id, otherClassNames)+' is not a valid path')
                        elif type(otherClassNames) == list:
                            for otherClassName in otherClassNames:
                                if type(otherClassName) != str:
                                    raise ValueError('If otherClassNames is a list, all elements of list must be strings')
                                try:
                                    os.makedirs(os.path.join(outputDir, 'tiles', id, otherClassName), exist_ok=True)
                                except:
                                    raise ValueError(os.path.join(outputDir, 'tiles', id, otherClassName)+' is not a valid path')
                        else:
                            raise ValueError('otherClassNames must be a string or list of strings')

                    # Create empty class mask directory (if desired)
                    if extractSegmentationMasks:
                        try:
                            os.makedirs(os.path.join(outputDir, 'masks', id, extractionClass), exist_ok=True)
                        except:
                            raise ValueError(os.path.join(outputDir, 'masks', id, extractionClass)+' is not a valid path')
                        if otherClassNames:
                            if type(otherClassNames) == str:
                                try:
                                    os.makedirs(os.path.join(outputDir, 'masks', id, otherClassNames), exist_ok=True)
                                except:
                                    raise ValueError(os.path.join(outputDir, 'masks', id, otherClassNames)+' is not a valid path')
                            elif type(otherClassNames) == list:
                                for otherClassName in otherClassNames:
                                    if type(otherClassName) != str:
                                        raise ValueError('If otherClassNames is a list, all elements of list must be strings')
                                    try:
                                        os.makedirs(os.path.join(outputDir, 'masks', id, otherClassName), exist_ok=True)
                                    except:
                                        raise ValueError(os.path.join(outputDir, 'masks', id, otherClassName)+' is not a valid path')
                            else:
                                raise ValueError('otherClassNames must be a string or list of strings')

        channel_sums = np.zeros(3)
        channel_squared_sums = np.zeros(3)
        tileCounter = 0
        normalize_to_1max = transforms.Compose([transforms.ToTensor()])

        # Extract tiles
        for ec,tte in annotatedTilesToExtract.items():
            if returnOnlyNumTilesFromThisClass and ec == returnOnlyNumTilesFromThisClass:
                return(len(annotatedTileAddresses[ec]))

            if len(tte) > 0:
                if extractSegmentationMasks:
                    print("Extracting "+str(len(tte))+" of "+str(len(annotatedTileAddresses[ec]))+" "+ec+" tiles and segmentation masks...")
                else:
                    print("Extracting "+str(len(tte))+" of "+str(len(annotatedTileAddresses[ec]))+" "+ec+" tiles...")

            for tl in tte:
                area = self.getTile(tl)
                if (tissueLevelThreshold) and (foregroundLevelThreshold):
                    area.write_to_file(os.path.join(outputDir, 'tiles', id, ec,
                        id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
                elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
                    area.write_to_file(os.path.join(outputDir, 'tiles', id, ec,
                        id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel.jpg'), Q=100)
                elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
                    area.write_to_file(os.path.join(outputDir, 'tiles', id, ec,
                        id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
                else:
                    area.write_to_file(os.path.join(outputDir, 'tiles', id, ec,
                        id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize.jpg'), Q=100)

                tileCounter = tileCounter + 1
                if returnTileStats:
                    nparea = self.getTile(tl, writeToNumpy=True)[...,:3] # remove transparency channel
                    nparea = normalize_to_1max(nparea).numpy() # normalize values from 0-255 to 0-1
                    local_channel_sums = np.sum(nparea, axis=(1,2))
                    local_channel_squared_sums = np.sum(np.square(nparea), axis=(1,2))
                    channel_sums = np.add(channel_sums, local_channel_sums)
                    channel_squared_sums = np.add(channel_squared_sums, local_channel_squared_sums)

                # Extract segmentation masks
                if extractSegmentationMasks:
                    mask = self.getAnnotationTileMask(tl, ec)
                    if (tissueLevelThreshold) and (foregroundLevelThreshold):
                        mask.save(os.path.join(outputDir, 'masks', id, ec,
                            id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'))
                    elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
                        mask.save(os.path.join(outputDir, 'masks', id, ec,
                            id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_mask.gif'))
                    elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
                        mask.save(os.path.join(outputDir, 'masks', id, ec,
                            id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'))
                    else:
                        mask.save(os.path.join(outputDir, 'masks', id, ec,
                            id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_mask.gif'))

        if returnOnlyNumTilesFromThisClass:
            raise Warning(returnOnlyNumTilesFromThisClass+' not found in tile dictionary')

        if tileCounter == 0:
            print('Warning: 0 suitable aannotated tiles found across all classes; making no tile directories and returning zeroes')

        if returnTileStats:
            if tileDirName:
                return {'slide': tileDirName,
                        'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                        'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                        'num_tiles': tileCounter}
            else:
                return {'slide': self.slideFileName,
                        'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                        'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                        'num_tiles': tileCounter}
        else:
            return True
            #channel_means_across_tiles = np.array(channel_means_across_tiles)
            #channel_stds_across_tiles = np.array(channel_stds_across_tiles)
            #return {'channel_means': np.mean(channel_means_across_tiles, axis=0),
            #        'channel_stds': np.mean(channel_stds_across_tiles, axis=0),
            #        'num_tiles': tileCounter}



            # support per class number of tiles to extract
            # raise warning if fewer than number found in a slide's class
            # raise warning if no slides found in a slide's class
            # print how many of each tile are being extracted for each class
            # numTilesToExtractPerClass (allow dictionary?), can be 'all'
            # automatically detect whether it is an asap xml or qupath geojson (and if not throw error)

    # ADAM EXPERIMENTAL
    def extractRandomUnannotatedTiles(self, outputDir, tileDirName=False, numTilesToExtract=100, unannotatedClassName='unannotated', otherClassNames=False,
        extractSegmentationMasks=False, foregroundLevelThreshold=False, tissueLevelThreshold=False, returnTileStats=True, seed=False):
        """A function to extract randomly selected tiles that don't overlap any
        annotations into directory structure amenable to torch.utils.data.ConcatDataset

        Args:
            outputDir (string): the path to the directory where the tile directory will be stored
            tileDirName (string, optional): what to call the hightest level tile directory that will be created. Default is 'tiles'
            numTilesToExtract (int, optional): the number of random unannotated tiles to extract. Default is 50.
            unannotatedClassName (string, optional): the name that the unannotated "class" directory should be called. Default is "unannotated".
            otherClassNames (string or list of strings, optional): if defined, creates an empty class directory alongside the unannotated class directory for each class name in the list (or string) for torch ImageFolder purposes
            extractSegmentationMasks (Boolean, optional): whether to extract a 'masks' directory that is exactly parallel to the 'tiles' directory, and contains binary segmentation mask tiles for each class desired (these tiles will of course all be entirely black). Default is False.
            foregroundLevelThreshold (string or int or float, optional): if defined as an int, only extracts tiles with a 0-100 foregroundLevel value less or equal to than the set value (0 is a black tile, 100 is a white tile). Only includes Otsu's method-passing tiles if set to 'otsu', or triangle algorithm-passing tiles if set to 'triangle'. Default is not to filter on foreground at all.
            tissueLevelThreshold (Boolean, optional): if defined, only extracts tiles with a 0 to 1 tissueLevel probability greater than or equal to the set value. Default is False.
            returnTileStats (Boolean, optional): whether to return the 0-1 normalized sum of channel values, the sum of the squares of channel values, and the number of tiles extracted for use in global mean and variance computation. Default is True.
            seed (int, optional): the random seed to use for reproducible anayses. Default is not to use a seed when randomly selecting tiles.

        Example:
            channel_data = pathml_slide.extractRandomUnannotatedTiles("/path/to/directory", numTilesToExtract=200, unannotatedClassName="non_metastasis", tissueLevelThreshold=0.995)
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before extracting tiles')

        if seed:
            if type(seed) != int:
                raise ValueError('Seed must be an integer')
            random.seed(seed)

        # get case ID
        if tileDirName:
            if type(tileDirName) != str:
                raise ValueError("tileDirName must be a string")
            else:
                id = tileDirName
        else:
            id = self.slideFileName

        if ((type(tissueLevelThreshold) != int) and (type(tissueLevelThreshold) != float)) or ((tissueLevelThreshold <= 0) or (tissueLevelThreshold > 1)):
            raise ValueError('tissueLevelThreshold must be a number greater than zero and less than or equal to 1')
        if (type(numTilesToExtract) != int) or (numTilesToExtract <= 0):
            raise ValueError('numTilesToExtract must be a integer greater than 0')

        # get classes to NOT extract
        annotationClasses = []
        for key, value in self.tileDictionary[list(self.tileDictionary.keys())[0]].items():
            if 'Overlap' in key:
                annotationClasses.append(key)
        if len(annotationClasses) == 0:
            print('No annotations found in tile dictionary; sampling randomly from all suitable tiles')
            #raise Warning('No annotations currently added to Slide tile dictionary; annotations can be added with addAnnotations()')

        # Collect all unannotated tiles
        unannotatedTileAddresses = []

        suitable_tile_addresses = self.suitableTileAddresses(tissueLevelThreshold=tissueLevelThreshold, foregroundLevelThreshold=foregroundLevelThreshold)
        for address in suitable_tile_addresses:
            overlapsAnnotation = False
            for annotationClass in annotationClasses:
                if self.tileDictionary[address][annotationClass] > 0:
                    overlapsAnnotation = True
                    break
            if not overlapsAnnotation:
                unannotatedTileAddresses.append(address)

        #for address in self.iterateTiles():
#
#            if tissueLevelThreshold and foregroundLevelThreshold:
#                if 'tissueLevel' not in self.tileDictionary[address]:
#                    raise PermissionError('Deep tissue detection must be performed with detectTissue() before tissueLevelThreshold can be defined')
#                if 'foregroundLevel' not in self.tileDictionary[address]:
#                    raise PermissionError('Foreground detection must be performed with detectForeground() before tissueLevelThreshold can be defined')
#                #if foregroundLevelThreshold:
#                if (self.tileDictionary[address]['tissueLevel'] >= tissueLevelThreshold) and (self.tileDictionary[address]['foregroundLevel'] <= foregroundLevelThreshold):
#                    overlapsAnnotation = False
#                    for annotationClass in annotationClasses:
#                        if self.tileDictionary[address][annotationClass] > 0:
#                            overlapsAnnotation = True
#                            break
#                    if not overlapsAnnotation:
#                        unannotatedTileAddresses.append(address)
#
#            elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
#                if 'tissueLevel' not in self.tileDictionary[address]:
#                    raise PermissionError('Deep tissue detection must be performed with detectTissue() before tissueLevelThreshold can be defined')
#                if self.tileDictionary[address]['tissueLevel'] >= tissueLevelThreshold: # do not extract background and artifact tiles
#                    overlapsAnnotation = False
#                    for annotationClass in annotationClasses:
#                        if self.tileDictionary[address][annotationClass] > 0:
#                            overlapsAnnotation = True
#                            break
#                    if not overlapsAnnotation:
#                        unannotatedTileAddresses.append(address)
#
#            elif (foregroundLevelThreshold) and (not tissueLevelThreshold):
#                if 'foregroundLevel' not in self.tileDictionary[address]:
#                    raise PermissionError('Foreground detection must be performed with detectForeground() before foregroundLevelThreshold can be defined')
#                if self.tileDictionary[address]['foregroundLevel'] <= foregroundLevelThreshold:
#                    overlapsAnnotation = False
#                    for annotationClass in annotationClasses:
#                        if self.tileDictionary[address][annotationClass] > 0:
#                            overlapsAnnotation = True
#                            break
#                    if not overlapsAnnotation:
#                        unannotatedTileAddresses.append(address)
#
#            else:
#                overlapsAnnotation = False
#                for annotationClass in annotationClasses:
#                    if self.tileDictionary[address][annotationClass] > 0:
#                        overlapsAnnotation = True
#                        break
#                if not overlapsAnnotation:
#                    unannotatedTileAddresses.append(address)

        if len(unannotatedTileAddresses) == 0:
            print('Warning: 0 unannotated tiles found; making no tile directories and returning zeroes')
        if len(unannotatedTileAddresses) < numTilesToExtract:
            print('Warning: '+str(len(unannotatedTileAddresses))+' unannotated tiles found but requested '+str(numTilesToExtract)+' tiles to extract. Extracting all suitable tiles...')
            unannotatedTilesToExtract = unannotatedTileAddresses
        else:
            unannotatedTilesToExtract = random.sample(unannotatedTileAddresses, numTilesToExtract)
            #print(str(len(unannotatedTileAddresses))+' unannotated tiles found in total')

        # Create empty class tile directories
        if len(unannotatedTileAddresses) > 0:
            try:
                os.makedirs(os.path.join(outputDir, 'tiles', id, unannotatedClassName), exist_ok=True)
            except:
                raise ValueError(os.path.join(outputDir, 'tiles', id, unannotatedClassName)+' is not a valid path')
            if otherClassNames:
                if type(otherClassNames) == str:
                    try:
                        os.makedirs(os.path.join(outputDir, 'tiles', id, otherClassNames), exist_ok=True)
                    except:
                        raise ValueError(os.path.join(outputDir, 'tiles', id, otherClassNames)+' is not a valid path')
                elif type(otherClassNames) == list:
                    for otherClassName in otherClassNames:
                        if type(otherClassName) != str:
                            raise ValueError('If otherClassNames is a list, all elements of list must be strings')
                        try:
                            os.makedirs(os.path.join(outputDir, 'tiles', id, otherClassName), exist_ok=True)
                        except:
                            raise ValueError(os.path.join(outputDir, 'tiles', id, otherClassName)+' is not a valid path')
                else:
                    raise ValueError('otherClassNames must be a string or list of strings')

            # Create empty class mask directory (if desired)
            if extractSegmentationMasks:
                try:
                    os.makedirs(os.path.join(outputDir, 'masks', id, unannotatedClassName), exist_ok=True)
                except:
                    raise ValueError(os.path.join(outputDir, 'masks', id, unannotatedClassName)+' is not a valid path')
                if otherClassNames:
                    if type(otherClassNames) == str:
                        try:
                            os.makedirs(os.path.join(outputDir, 'masks', id, otherClassNames), exist_ok=True)
                        except:
                            raise ValueError(os.path.join(outputDir, 'masks', id, otherClassNames)+' is not a valid path')
                    elif type(otherClassNames) == list:
                        for otherClassName in otherClassNames:
                            if type(otherClassName) != str:
                                raise ValueError('If otherClassNames is a list, all elements of list must be strings')
                            try:
                                os.makedirs(os.path.join(outputDir, 'masks', id, otherClassName), exist_ok=True)
                            except:
                                raise ValueError(os.path.join(outputDir, 'masks', id, otherClassName)+' is not a valid path')
                    else:
                        raise ValueError('otherClassNames must be a string or list of strings')

            if extractSegmentationMasks:
                print("Extracting "+str(len(unannotatedTilesToExtract))+" of "+str(len(unannotatedTileAddresses))+" "+unannotatedClassName+" tiles and segmentation masks...")
            else:
                print("Extracting "+str(len(unannotatedTilesToExtract))+" of "+str(len(unannotatedTileAddresses))+" "+unannotatedClassName+" tiles...")

        # Extract the desired number of unannotated tiles
        channel_sums = np.zeros(3)
        channel_squared_sums = np.zeros(3)
        tileCounter = 0
        normalize_to_1max = transforms.Compose([transforms.ToTensor()])

        #print("Tiles to extract:", unannotatedTilesToExtract)
        for tl in unannotatedTilesToExtract:
            #print("On tile:", tl)
            area = self.getTile(tl)
            if (tissueLevelThreshold) and (foregroundLevelThreshold):
                area.write_to_file(os.path.join(outputDir, 'tiles', id, unannotatedClassName,
                    id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
            elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
                area.write_to_file(os.path.join(outputDir, 'tiles', id, unannotatedClassName,
                    id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel.jpg'), Q=100)
            elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
                area.write_to_file(os.path.join(outputDir, 'tiles', id, unannotatedClassName,
                    id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel.jpg'), Q=100)
            else:
                area.write_to_file(os.path.join(outputDir, 'tiles', id, unannotatedClassName,
                    id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize.jpg'), Q=100)

            tileCounter = tileCounter + 1
            if returnTileStats:
                nparea = self.getTile(tl, writeToNumpy=True)[...,:3] # remove transparency channel
                #print(nparea[...,0])
                #print(nparea[...,1])
                #print(nparea[...,2])
                #print(nparea[...,3])
                nparea = normalize_to_1max(nparea).numpy() # normalize values from 0-255 to 0-1
                local_channel_sums = np.sum(nparea, axis=(1,2))
                local_channel_squared_sums = np.sum(np.square(nparea), axis=(1,2))

                channel_sums = np.add(channel_sums, local_channel_sums)
                channel_squared_sums = np.add(channel_squared_sums, local_channel_squared_sums)
                #print(channel_sums)
                #print("---")
                #print(channel_squared_sums)
                #channel
                #print(nparea.shape)
                #channel_means = np.mean(nparea, axis=(1,2))
                #channel_stds = np.std(nparea, axis=(1,2))
                #print(channel_means)
                #print(channel_stds)
                #quit()
                #channel_means_across_tiles.append(channel_means)
                #channel_stds_across_tiles.append(channel_stds)
            if extractSegmentationMasks:
                height = self.tileDictionary[tl]['height']
                mask = Image.new('1', (height, height), 0) # blank mask

                if (tissueLevelThreshold) and (foregroundLevelThreshold):
                    mask.save(os.path.join(outputDir, 'masks', id, unannotatedClassName,
                        id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'))
                elif (tissueLevelThreshold) and (not foregroundLevelThreshold):
                    mask.save(os.path.join(outputDir, 'masks', id, unannotatedClassName,
                        id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['tissueLevel']*1000)))+'tissueLevel_mask.gif'))
                elif (not tissueLevelThreshold) and (foregroundLevelThreshold):
                    mask.save(os.path.join(outputDir, 'masks', id, unannotatedClassName,
                        id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_'+str(int(round(self.tileDictionary[tl]['foregroundLevel'])))+'foregroundLevel_mask.gif'))
                else:
                    mask.save(os.path.join(outputDir, 'masks', id, unannotatedClassName,
                        id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_mask.jpg'))



        if returnTileStats:
            #channel_means_across_tiles = np.array(channel_means_across_tiles)
            #channel_stds_across_tiles = np.array(channel_stds_across_tiles)
            if tileDirName:
                return {'slide': tileDirName,
                        'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                        'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                        'num_tiles': tileCounter}
            else:
                return {'slide': self.slideFileName,
                        'channel_sums': channel_sums,#np.mean(channel_means_across_tiles, axis=0).tolist(),
                        'channel_squared_sums': channel_squared_sums,#np.mean(channel_stds_across_tiles, axis=0).tolist(),
                        'num_tiles': tileCounter}
        else:
            return True



    # ADAM EXPERIMENTAL
    def detectTissue(self, tissueDetectionLevel=1, tissueDetectionTileSize=512, tissueDetectionTileOverlap=0, tissueDetectionUpsampleFactor=4, batchSize=20, overwriteExistingTissueDetection=False, modelStateDictPath='../pathml/pathml/models/deep-tissue-detector_densenet_state-dict.pt', architecture='densenet'):
        """A function to apply PathML's built-in deep tissue detector to assign
        artifact, background, and tissue probabilities that sum to one to each tile
        in the tile dictionary. The raw tissue detection map for a WSI is saved into
        a Slide attribute called rawTissueDetectionMap in the Slide which can be loaded
        into a new Slide object to save inference time with detectTissueFromRawTissueDetectionMap().
        For this reason calling saveSelf() after detectTissue() finishes is recommended.

        Args:
            tissueDetectionLevel (int, optional): the level of the WSI pyramid at which to perform the tissue detection. Default is 1.
            tissueDetectionTileSize (int, optional): the edge length in pixels of the tiles that the deep tissue detector will be inferred on. Default is 512.
            tissueDetectionTileOverlap (float, optional): the fraction of a tile's edge length that overlaps the left, right, above, and below tiles. Default is 0.
            tissueDetectionUpsampleFactor (int, optional): the factor why which the WSI should be upsampled when performing tissue detection. Default is 4.
            batchSize (int, optional): the number of tiles per minibatch when inferring on the deep tissue detector. Default is 20.
            overwriteExistingTissueDetection (Boolean, optional): whether to overwrite any existing deep tissue detector predictions if they are already present in the tile dictionary. Default is False.
            modelStateDictPath (string, optional): the path to the state dictionary of the deep tissue detector; it must be a 3-class classifier, with the class order as follows: background, artifact, tissue. Default is the path to the state dict of the deep tissue detector build into PathML.
            architecture (string, optional): the name of the architecture that the state dict belongs to. Currently supported architectures include resnet18, inceptionv3, vgg16, vgg16_bn, vgg19, vgg19_bn, densenet, alexnet, and squeezenet. Default is "densenet", which is the architecture of PathML's built-in deep tissue detector.

        Example:
            pathml_slide.detectTissue()
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before applying tissue detector')
        if hasattr(self, 'rawTissueDetectionMap') and (not overwriteExistingTissueDetection):
            raise Warning('Tissue detection has already been performed. Use overwriteExistingTissueDetection if you wish to write over it')

        if torch.cuda.is_available():
            print("Inferring tissue detection model using GPU")
        else:
            print("Inferring tissue detection model using CPU")

        print("Detecting tissue of "+self.slideFilePath)
        tissueForegroundSlide = Slide(self.slideFilePath, level=tissueDetectionLevel).setTileProperties(tileSize=tissueDetectionTileSize, tileOverlap=tissueDetectionTileOverlap) # tile size and overlap for tissue detector, not final tiles
        tmpProcessor = Processor(tissueForegroundSlide)
        tissueForegroundTmp = tmpProcessor.applyModel(tissueDetector(modelStateDictPath=modelStateDictPath, architecture=architecture), batch_size=batchSize, predictionKey='tissue_detector').adoptKeyFromTileDictionary(upsampleFactor=tissueDetectionUpsampleFactor)

        predictionMap = np.zeros([tissueForegroundTmp.numTilesInY, tissueForegroundTmp.numTilesInX,3])
        for address in tissueForegroundTmp.iterateTiles():
            if 'tissue_detector' in tissueForegroundTmp.tileDictionary[address]:
                predictionMap[address[1], address[0], :] = tissueForegroundTmp.tileDictionary[address]['tissue_detector']

        predictionMap2 = np.zeros([self.numTilesInY, self.numTilesInX])
        predictionMap1res = resize(predictionMap, predictionMap2.shape, order=0, anti_aliasing=False)

        self.rawTissueDetectionMap = predictionMap
        self.resizedTissueDetectionMap = predictionMap1res

        for address in self.iterateTiles():
            self.tileDictionary[address].update({'artifactLevel': predictionMap1res[address[1], address[0]][0]})
            self.tileDictionary[address].update({'backgroundLevel': predictionMap1res[address[1], address[0]][1]})
            self.tileDictionary[address].update({'tissueLevel': predictionMap1res[address[1], address[0]][2]})

    def detectTissueFromRawTissueDetectionMap(self, rawTissueDetectionMap, overwriteExistingTissueDetection=False):
        """Function to load a raw tissue detection map from a previous application
        of detectTissue() to a slide.

        Args:
            rawTissueDetectionMap (numpy array): the raw tissue detection map numpy array saved in a Slide object's rawTissueDetectionMap attribute.
            overwriteExistingTissueDetection (Boolean, optional): whether to overwrite any existing deep tissue detection predictions in the tile dictionary if they are present. Default is False.

        Example:
            pathml_slide.detectTissueFromRawTissueDetectionMap(Slide("/path/to/old_pathml_slide.pml")).rawTissueDetectionMap)
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before applying tissue detector')
        if hasattr(self, 'rawTissueDetectionMap') and (not overwriteExistingTissueDetection):
            raise Warning('Tissue detection has already been performed. Use overwriteExistingTissueDetection if you wish to write over it')

        predictionMap2 = np.zeros([self.numTilesInY, self.numTilesInX])
        predictionMap1res = resize(rawTissueDetectionMap, predictionMap2.shape, order=0, anti_aliasing=False)

        self.rawTissueDetectionMap = rawTissueDetectionMap
        self.resizedTissueDetectionMap = predictionMap1res

        for address in self.iterateTiles():
            self.tileDictionary[address].update({'artifactLevel': predictionMap1res[address[1], address[0]][0]})
            self.tileDictionary[address].update({'backgroundLevel': predictionMap1res[address[1], address[0]][1]})
            self.tileDictionary[address].update({'tissueLevel': predictionMap1res[address[1], address[0]][2]})


    # ADAM EXPERIMENTAL
    def visualizeTissueDetection(self, fileName=False, folder=os.getcwd()):
        """A function to generate a 3-color tissue detection map showing where
        on a WSI the deep tissue detector applied with detectTissue() artifact
        was found (red), where background was found (green), and where tissue
        was found (blue).

        Args:
            fileName (string, optional): the name of the file where the deep tissue detection inference map image will be saved, excluding an extension
            folder (string, optional): the path to the directory where the deep tissue detection inference map image will be saved. Default is the current working directory.

        Example:
            pathml_slide.visualizeTissueDetection(folder="/path/where/tissue_detection_map_will_be_saved")
        """

        if not self.hasTileDictionary():
            raise PermissionError('setTileProperties must be called before saving self')
        if not hasattr(self, 'rawTissueDetectionMap'):
            raise PermissionError('detectTissue must be called before getting resized tissue detection map')

        # get case ID
        if fileName:
            if type(fileName) == str:
                id = fileName
            else:
                raise ValueError('fileName must be a string')
        else:
            id = self.slideFileName

        map = resize(self.rawTissueDetectionMap, np.zeros([self.numTilesInY, self.numTilesInX]).shape, order=0, anti_aliasing=False)
        #plt.imsave(os.path.join(folder, id+'.png'), map)

        plt.figure()
        #plt.imshow(resize(ourNewImg, classMask.shape))
        plt.imshow(map, cmap=mpl.colors.ListedColormap(['blue', 'yellow', 'red']))
        #plt.imshow(foregroundMask, cmap='plasma', alpha=0.3, vmin=0, vmax=1.0)
        #plt.colorbar()
        plt.title(id+"\n"+'deep tissue detection')
        if folder:
            #os.makedirs(os.path.join(folder, self.slideFileName), exist_okay=True)
            plt.savefig(os.path.join(folder, id+"_tissuedetection.png"))
        else:
            plt.show(block=False)

    # ADAM EXPERIMENTAL
    def inferClassifier(self, trainedModel, classNames, dataTransforms=None, batchSize=30, numWorkers=16, foregroundLevelThreshold=False, tissueLevelThreshold=False, overwriteExistingClassifications=False):
        """A function to infer a trained classifier on a Slide object using
        PyTorch.

        Args:
            trainedModel (torchvision.models): A PyTorch torchvision model that has been trained for the classification task desired for inference.
            classNames (list of strings): an alphabetized list of class names.
            dataTransforms (torchvision.transforms.Compose): a PyTorch torchvision.Compose object with the desired data transformations.
            batchSize (int, optional): the number of tiles to use in each inference minibatch.
            numWorkers (int, optional): the number of workers to use when inferring the model on the WSI
            foregroundLevelThreshold (string or int or float, optional): if defined as an int, only infers trainedModel on tiles with a 0-100 foregroundLevel value less or equal to than the set value (0 is a black tile, 100 is a white tile). Only infers on Otsu's method-passing tiles if set to 'otsu', or triangle algorithm-passing tiles if set to 'triangle'. Default is not to filter on foreground at all.
            tissueLevelThreshold (Boolean, optional): if defined, only infers trainedModel on tiles with a 0 to 1 tissueLevel probability greater than or equal to the set value. Default is False.
            overwriteExistingClassifications (Boolean, optional): whether to overwrite any existing classification inferences if they are already present in the tile dictionary. Default is False.

        Example:
            import torchvision
            class_names = ['metastasis', 'non_metastasis']
            trained_model = torchvision.models.vgg19_bn(pretrained=False)
            num_ftrs = trainedModel.classifier[6].in_features
            trained_model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
            trained_model.load_state_dict(torch.load(path_to_model_state_dict))
            data_transforms = torchvision.transforms.Compose([
                transforms.Resize(patch_size),
                transforms.ToTensor(),
                transforms.Normalize(global_channel_means.tolist(), global_channel_stds.tolist())
            ])
            pathml_slide.inferClassifier(trained_model, classNames=class_names,
                                            dataTransforms=data_transforms, tissueLevelThreshold=0.995)
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before inferring a classifier')
        if tissueLevelThreshold:
            if not self.hasTissueDetection():
                raise PermissionError('Deep tissue detection must be performed with detectTissue() before tissueLevelThreshold can be defined')
        if foregroundLevelThreshold:
            if 'foregroundLevel' not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                raise PermissionError('Foreground detection must be performed with detectForeground() before tissueLevelThreshold can be defined')
        if type(classNames) != list:
            raise ValueError('classes must be a list if defined')

        if torch.cuda.is_available():
            print("Inferring model on GPU")
        else:
            print("Inferring model on CPU")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainedModel.to(device)
        trainedModel.eval()

        pathSlideDataset = WholeSlideImageDataset(self, tissueLevelThreshold=tissueLevelThreshold,
            foregroundLevelThreshold=foregroundLevelThreshold, transform=dataTransforms)

        if 'classifierInferencePrediction' in self.tileDictionary[pathSlideDataset.suitableTileAddresses[0]]:
            if not overwriteExistingClassifications:
                raise PermissionError('Classification predictions are already present in the tile dictionary. Set overwriteExistingClassifications to True to overwrite them.')

        pathSlideDataloader = torch.utils.data.DataLoader(pathSlideDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
        classifierPredictionTileAddresses = []
        for inputs in tqdm(pathSlideDataloader):
            inputTile = inputs['image'].to(device)
            output = trainedModel(inputTile)
            output = output.to(device)

            batch_prediction = torch.nn.functional.softmax(
                output, dim=1).cpu().data.numpy()

            # Reshape it is a Todo - instead of for looping
            for index in range(len(inputTile)):
                tileAddress = (inputs['tileAddress'][0][index].item(),
                               inputs['tileAddress'][1][index].item())
                preds = batch_prediction[index, ...].tolist()
                if len(preds) != len(classNames):
                    raise ValueError('Model has '+str(len(preds))+' classes but only '+str(len(classNames))+' class names were provided in the classes argument')
                prediction = {}
                for i, pred in enumerate(preds):
                    prediction[classNames[i]] = pred
                self.appendTag(tileAddress, 'classifierInferencePrediction', prediction)
                classifierPredictionTileAddresses.append(tileAddress)
        if len(classifierPredictionTileAddresses) > 0:
            self.classifierPredictionTileAddresses = classifierPredictionTileAddresses
        else:
            raise Warning('No suitable tiles found at current tissueLevelThreshold and foregroundLevelThreshold')



##########################################################

    def inferSegmenter(self, trainedModel, classNames, dataTransforms=None, batchSize=1, numWorkers=16, foregroundLevelThreshold=False, tissueLevelThreshold=False, overwriteExistingSegmentations=False):#, saveInChunksAtFolder=False):
        """A function to infer a trained segmentation model on a Slide object using
        PyTorch.

        Args:
            trainedModel (torchvision.models): A PyTorch segmentation model that has been trained for the segmentation task desired for inference.
            classNames (list of strings): a list of class names. The first class name is expected to correspond with the first channel of the output mask image, the second with the second, and so on.
            dataTransforms (torchvision.transforms.Compose): a PyTorch torchvision.Compose object with the desired data transformations.
            batchSize (int, optional): the number of tiles to use in each inference minibatch.
            numWorkers (int, optional): the number of workers to use when inferring the model on the WSI
            foregroundLevelThreshold (string or int or float, optional): if defined as an int, only infers trainedModel on tiles with a 0-100 foregroundLevel value less or equal to than the set value (0 is a black tile, 100 is a white tile). Only infers on Otsu's method-passing tiles if set to 'otsu', or triangle algorithm-passing tiles if set to 'triangle'. Default is not to filter on foreground at all.
            tissueLevelThreshold (Boolean, optional): if defined, only infers trainedModel on tiles with a 0 to 1 tissueLevel probability greater than or equal to the set value. Default is False.
            overwriteExistingSegmentations (Boolean, optional): whether to overwrite any existing segmentation inferences if they are already present in the tile dictionary. Default is False.

        Example:
            import torch
            import torchvision
            class_names = ['metastasis']
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            trained_model = UNet(n_channels=3, n_classes=len(class_names))
            trained_model.load_state_dict(torch.load(path_to_model_state_dict, map_location=device))
            pathml_slide.inferSegmenter(trained_model, classNames=class_names, batchSize=6, tissueLevelThreshold=0.995)
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before inferring a classifier')
        if tissueLevelThreshold:
            if not self.hasTissueDetection():
                raise PermissionError('Deep tissue detection must be performed with detectTissue() before tissueLevelThreshold can be defined')
        if foregroundLevelThreshold:
            if 'foregroundLevel' not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                raise PermissionError('Foreground detection must be performed with detectForeground() before tissueLevelThreshold can be defined')
        if type(classNames) != list:
            raise ValueError('classes must be a list if defined')

        if torch.cuda.is_available():
            print("Inferring model on GPU")
        else:
            print("Inferring model on CPU")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainedModel.to(device)
        trainedModel.eval()

        if trainedModel.n_classes != len(classNames):
            raise ValueError('Model has '+str(trainedModel.n_classes)+' classes but only '+str(len(classNames))+' class names were provided in the classes argument')

        pathSlideDataset = WholeSlideImageDataset(self, tissueLevelThreshold=tissueLevelThreshold,
            foregroundLevelThreshold=foregroundLevelThreshold, transform=dataTransforms, segmenting=True)

        if 'segmenterInferencePrediction' in self.tileDictionary[pathSlideDataset.suitableTileAddresses[0]]:
            if not overwriteExistingSegmentations:
                raise PermissionError('Segmentation predictions are already present in the tile dictionary. Set overwriteExistingSegmentations to True to overwrite them.')

        pathSlideDataloader = torch.utils.data.DataLoader(pathSlideDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
        segmenterPredictionTileAddresses = []
        counter = 0
        for inputs in tqdm(pathSlideDataloader):

            inputTile = inputs['image'].to(device)

            # input into net
            output = trainedModel(inputTile)
            output = output.to(device)

            if trainedModel.n_classes > 1:
                batch_probs = F.softmax(output, dim=1)
            else:
                batch_probs = torch.sigmoid(output)

            for index in range(len(inputTile)):
                tileAddress = (inputs['tileAddress'][0][index].item(),
                               inputs['tileAddress'][1][index].item())

                tile_probs = batch_probs[index, ...]

                tile_probs = tile_probs.squeeze(0)

                tf = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(pathSlideDataset.tileSize),
                    #transforms.Resize(full_img.size[1]),
                    transforms.ToTensor() # converts from HWC to CWH and divides by 255
                ])

                tile_probs = tf(tile_probs.cpu())
                full_mask = tile_probs.squeeze().cpu().numpy()
                #full_mask = np.float16(full_mask)

                #print("Output mask data type:", full_mask.dtype)

                #print("full mask shape:", full_mask.shape)
                #print("full mask shape length:", len(full_mask.shape))

                if (len(full_mask.shape) == 2 and len(classNames) == 1):
                    pass
                elif (len(full_mask.shape) == 2 and len(classNames) != 1):
                    raise ValueError('Model has 1 output class but '+str(len(classNames))+' class names were provided in the classes argument')
                elif full_mask.shape[0] != len(classNames):
                    raise ValueError('Model has '+str(full_mask.shape[-1])+' output classes but only '+str(len(classNames))+' class names were provided in the classes argument')

                segmentation_masks = {}

                if len(full_mask.shape) == 2: # only one output class
                    segmentation_masks[classNames[0]] = full_mask
                else: # multiple output classes
                    for class_index in len(classNames):
                        segmentation_masks[classNames[class_index]] = full_mask[class_index,...]

                #if saveInChunksAtFolder:
                #    if not 'segmenterInferencePrediction' in self.tileDictionary[tileAddress]:
                #        self.appendTag(tileAddress, 'segmenterInferencePrediction', segmentation_masks)
                #else:
                self.appendTag(tileAddress, 'segmenterInferencePrediction', segmentation_masks)
                segmenterPredictionTileAddresses.append(tileAddress)

                #counter = counter + 1
                #if saveInChunksAtFolder:
                #    if len(counter) > 1000
                #    self.saveSelf(folder=saveInChunksAtFolder)
                #    counter = 0

        if len(segmenterPredictionTileAddresses) > 0:
            self.segmenterPredictionTileAddresses = segmenterPredictionTileAddresses
        else:
            raise Warning('No suitable tiles found at current tissueLevelThreshold and foregroundLevelThreshold')

####################################################



    def getTileDiceScore(self, tileAddress, className, pixelBinarizationThreshold=0.5):
        """A function that returns the Dice coefficient by comparing the tile's
        ground truth segmentation mask from addAnnotations() with the tile's
        inference segmentation mask output by a trained model via inferSegmenter().

        Args:
            tileAddress (tuple): the tile dictionary address of the tile to compute the Dice score for.
            className (string): the name of the class to compute the Dice score for. This class name must be present in the tile dictionary in both the annotations (added to the tile dictionry via addAnnotations()) as well in the segmentation inference output (added to the tile dictionary via inferSegmenter()).
            pixelBinarizationThreshold (float, optional): the 0-1 threshold above which a pixel in the segmentation probability mask output by the trained model is considered a member of the class. Default is 0.5.

        Example:
            tile_dice_score = getTileDiceScore(pathml_slide.suitableTileAddresses[0], 'metastasis')
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before getting a tile Dice score.')
        #if not self.hasAnnotations():
        #    print("Warning: no annotations found in Slide. All ground truth tile masks in Slide will be be totally absent of "+classToThreshold+" pixels. Run addAnnotations() if there should be annotations in this Slide.")
        if 'segmenterInferencePrediction' not in self.tileDictionary[tileAddress]:
            raise PermissionError('No segmentation prediction currently exists at the specified tile address. Please run inferSegmenter() and make sure that the specified tile is within the suitable tile set requested with tissueLevelThreshold and foregroundLevelThreshold.')
        if className not in self.tileDictionary[tileAddress]['segmenterInferencePrediction']:
            raise PermissionError(className+' is not a class present in segmenterInferencePrediction.')
        #if className not in self.annotationClassMultiPolygons:
        #    raise PermissionError(className+' is not a class present in annotations.')

        binarized_prediction_mask = (torch.from_numpy(self.tileDictionary[tileAddress]['segmenterInferencePrediction'][className]) > pixelBinarizationThreshold).float()
        # we set acceptTilesWithoutClass to True so that we will get blank masks for negative slides
        ground_truth_mask = torch.from_numpy(self.getAnnotationTileMask(tileAddress, className, writeToNumpy=True, acceptTilesWithoutClass=True)).float()
        ground_truth_mask = torch.div(ground_truth_mask, 255) # getAnnotationTileMask outputs 0-255 arrays, so we need to normalize to be 0-1 range
        #print("mean of ground truth mask:", torch.mean(ground_truth_mask))
        return dice_coeff(binarized_prediction_mask, ground_truth_mask).item()




    def suitableTileAddresses(self, tissueLevelThreshold=False, foregroundLevelThreshold=False):
        """A function that returns a list of the tile address tuples that meet
        set tissue and foreground thresholds. All addresses will be returned if
        neither tissueLevelThreshold nor foregroundLevelThreshold is defined.

        Args:
            foregroundLevelThreshold (string or int or float, optional): if defined as an int, only includes the tile address of tiles with a 0-100 foregroundLevel value less or equal to than the set value (0 is a black tile, 100 is a white tile). Only includes Otsu's method-passing tiles if set to 'otsu', or triangle algorithm-passing tiles if set to 'triangle'. Default is not to filter on foreground at all.
            tissueLevelThreshold (int or float, optional): if defined, only includes the tile addresses of tiles with a 0 to 1 tissueLevel probability greater than or equal to the set value. Default is False.

        Example:
            suitable_tile_addresses = pathml_slide.suitableTileAddresses(tissueLevelThreshold=0.995, foregroundLevelThreshold=88)
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before tile counting')
        if foregroundLevelThreshold:
            if 'foregroundLevel' not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                raise PermissionError('Foreground detection must be performed with detectForeground() before foregroundLevelThreshold can be defined.')
            if (foregroundLevelThreshold not in ['otsu', 'triangle']) and (type(foregroundLevelThreshold) not in [int, float]):
                raise ValueError("foregroundLevelThreshold must be an int, a float, 'otsu', or 'triangle'")
        if tissueLevelThreshold:
            if 'tissueLevel' not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                raise PermissionError('Tissue detection must be performed with detectTissue() before tissueLevelThreshold can be defined.')
            if type(foregroundLevelThreshold) not in [int, float]:
                raise ValueError("tissueLevelThreshold must be an int or float")

        suitableTileAddresses = []
        for tA in self.iterateTiles():
            if tissueLevelThreshold and foregroundLevelThreshold:
                if foregroundLevelThreshold == 'otsu':
                    if (self.tileDictionary[tA]['tissueLevel'] >= tissueLevelThreshold) and self.tileDictionary[tA]['foregroundOtsu']:
                        suitableTileAddresses.append(tA)
                elif foregroundLevelThreshold == 'triangle':
                    if (self.tileDictionary[tA]['tissueLevel'] >= tissueLevelThreshold) and self.tileDictionary[tA]['foregroundTriangle']:
                        suitableTileAddresses.append(tA)
                else:
                    if (self.tileDictionary[tA]['tissueLevel'] >= tissueLevelThreshold) and (self.tileDictionary[tA]['foregroundLevel'] <= foregroundLevelThreshold):
                        suitableTileAddresses.append(tA)
            elif tissueLevelThreshold and not foregroundLevelThreshold:
                if (self.tileDictionary[tA]['tissueLevel'] >= tissueLevelThreshold):
                    suitableTileAddresses.append(tA)
            elif foregroundLevelThreshold and not tissueLevelThreshold:
                if foregroundLevelThreshold == 'otsu':
                    if self.tileDictionary[tA]['foregroundOtsu']:
                        suitableTileAddresses.append(tA)
                elif foregroundLevelThreshold == 'triangle':
                    if self.tileDictionary[tA]['foregroundTriangle']:
                        suitableTileAddresses.append(tA)
                else:
                    if (self.tileDictionary[tA]['foregroundLevel'] <= foregroundLevelThreshold):
                        suitableTileAddresses.append(tA)
            else:
                suitableTileAddresses.append(tA)
        return suitableTileAddresses

    def visualizeClassifierInference(self, classToVisualize, folder=False, level=4):
        """A function to create an inference map image of a Slide after running
        inferClassifier() on it.

        Args:
            classToVisualize (string): the class to make an inference map image for. This class must be present in the tile dictionary from inferClassifier().
            folder (string, optional): the path to the directory where the map will be saved; if it is not defined, then the map will only be shown and not saved.
            level (int, optional): the level of the WSI pyramid to make the inference map image from.

        Example:
            pathml_slide.visualizeClassifierInference("metastasis", folder="path/to/folder")
        """

        ourNewImg = self.thumbnail(level=level)
        classMask = np.zeros((self.numTilesInX, self.numTilesInY)[::-1])

        #if not hasattr(self, 'segmenterPredictionTileAddresses'):
        foundPrediction = False
        for tileAddress, tileEntry in self.tileDictionary.items():
            if 'classifierInferencePrediction' in tileEntry:
                if classToVisualize in tileEntry['classifierInferencePrediction']:
                    classMask[tileAddress[1],tileAddress[0]] = tileEntry['classifierInferencePrediction'][classToVisualize]
                    foundPrediction = True
                else:
                    raise ValueError(classToVisualize+' not in classifierInferencePrediction')
            else:
                classMask[tileAddress[1],tileAddress[0]] = 0
        if not foundPrediction:
            raise ValueError('No predictions found in slide. Use inferClassifier() to generate them.')

        plt.figure()
        plt.imshow(resize(ourNewImg, classMask.shape))
        plt.imshow(classMask, cmap='plasma', alpha=0.3, vmin=0, vmax=1.0)
        plt.colorbar()
        plt.title(self.slideFileName+"\n"+classToVisualize)
        if folder:
            os.makedirs(os.path.join(folder, self.slideFileName), exist_okay=True)
            plt.savefig(os.path.join(folder, self.slideFileName, self.slideFileName+"_"+classToVisualize+".png"))
        else:
            plt.show(block=False)

    def visualizeThumbnail(self, folder=False, level=4):
        """A function to create a low-resolution image of the WSI stored in a
        Slide.

        Args:
            folder (string, optional): the path to the directory where the thumbnail image will be saved; if it is not defined, then the thumbnail image will only be shown with matplotlib.pyplot and not saved.
            level (int, optional): the level of the WSI pyramid to make the thumbnail image from. Higher numbers will result in a lower resolution thumbnail. Default is 4.

        Example:
            pathml_slide.visualizeThumbnail(folder="path/to/folder")
        """

        ourNewImg = self.thumbnail(level=level)
        classMask = np.zeros((self.numTilesInX, self.numTilesInY)[::-1])

        plt.figure()
        plt.imshow(resize(ourNewImg, classMask.shape))
        plt.title(self.slideFileName)
        if folder:
            os.makedirs(os.path.join(folder, self.slideFileName), exist_okay=True)
            plt.savefig(os.path.join(folder, self.slideFileName, self.slideFileName+"_thumbnail.png"))
        else:
            plt.show(block=False)

    def visualizeForeground(self, foregroundLevelThreshold, folder=False, colors=['#04F900', '#0000FE']):
        """A function to create a map image of a Slide after running
        detectForeground() on it.

        Args:
            foregroundLevelThreshold (string or int, optional): applies Otsu's method to find the threshold if set to 'otsu', the triangle algorithm to find the threshold if set to 'triangle', or simply uses the tiles at or above the minimum darkness intensity threshold specified if set as an int (0 is a pure black tile, 100 is a pure white tile). Default is not to filter the tile count this way. detectForeground() must be run first.
            folder (string, optional): the path to the directory where the map will be saved; if it is not defined, then the map will only be shown and not saved.
            colors (list, optional): a list of length two containing the color for the background followed by the color for the foreground in the map image. Colors must be defined for use in matplotlib.imshow's cmap argument. Default is a light green (#04F900) for background and a dark blue (#0000FE) for foreground.

        Example:
            pathml_slide.visualizeForeground("otsu", folder="path/to/folder")
        """

        if not self.hasTileDictionary():
            raise PermissionError(
                'setTileProperties must be called before inferring a classifier')
        if 'foregroundLevel' not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
            raise PermissionError('Foreground detection must be performed with detectForeground() before tissueLevelThreshold can be defined')

        foregroundMask = np.zeros((self.numTilesInX, self.numTilesInY)[::-1])

        for tileAddress, tileEntry in self.tileDictionary.items():
            if foregroundLevelThreshold == 'otsu':
                foregroundMask[tileAddress[1],tileAddress[0]] = tileEntry['foregroundOtsu']
            elif foregroundLevelThreshold == 'triangle':
                foregroundMask[tileAddress[1],tileAddress[0]] = tileEntry['foregroundTriangle']
            else:
                if tileEntry['foregroundLevel'] <= foregroundLevelThreshold:
                    foregroundMask[tileAddress[1],tileAddress[0]] = True
                else:
                    foregroundMask[tileAddress[1],tileAddress[0]] = False
            #foregroundMask[tileAddress[1],tileAddress[0]] = tileEntry['foreground']

        plt.figure()
        #plt.imshow(resize(ourNewImg, classMask.shape))
        plt.imshow(foregroundMask, cmap=mpl.colors.ListedColormap(colors))
        #plt.imshow(foregroundMask, cmap='plasma', alpha=0.3, vmin=0, vmax=1.0)
        #plt.colorbar()
        plt.title(self.slideFileName+"\n"+str(foregroundLevelThreshold)+" thresholded")
        #else:
        #    plt.title(self.slideFileName+"\n"+'Foreground')
        if folder:
            os.makedirs(os.path.join(folder, id), exist_okay=True)
            #if method:
            plt.savefig(os.path.join(folder, id, id+str(foregroundLevelThreshold)+"_thresholded_foregrounddetection.png"))
            #else:
            #    plt.savefig(os.path.join(folder, id, id+"_foregrounddetection.png"))
        else:
            plt.show(block=False)

    def numTilesAboveClassPredictionThreshold(self, classToThreshold, probabilityThresholds):
        """A function to return the number of tiles at or above one or a list of
        probability thresholds for a classification class added to each tile in
        the tile dictionary by inferClassifier().

        Args:
            classToThreshold (string): the class to threshold the tiles by. The class must be already present in the tile dictionary from inferClassifier().
            probabilityThresholds (float or list of floats): the probability threshold or list of probability thresholds (in the range 0 to 1) to check. If a float is provided, just that probability threshold will be used, and an int of the number of tiles at or above that threshold will be returned. If a list of floats is provided, a list of ints of the number of tiles at or above each of those thresholds in respective order to the inputted threshold list will be returned.
        """

        if not hasattr(self, 'classifierPredictionTileAddresses'):
            foundPrediction = False
            classifierPredictionTileAddresses = []
            for tileAddress, tileEntry in self.tileDictionary.items():
                if 'classifierInferencePrediction' in tileEntry:
                    classifierPredictionTileAddresses.append(tileAddress)
                    foundPrediction = True
            if foundPrediction:
                self.classifierPredictionTileAddresses = classifierPredictionTileAddresses
            else:
                raise ValueError('No predictions found in slide. Use inferClassifier() to generate them.')

        if type(probabilityThresholds) in [float, int]:
            pT = [probabilityThresholds]
        elif type(probabilityThresholds) == list:
            pT = probabilityThresholds
        else:
            raise ValueError('probabilityThresholds must be an int, float, or list of ints or floats')

        numTilesAboveProbThreshList = []
        for probabilityThreshold in pT:
            numTilesAboveProbThresh = 0
            for addr in self.classifierPredictionTileAddresses:
                if classToThreshold not in self.tileDictionary[addr]['classifierInferencePrediction']:
                    raise ValueError(classToVisualize+' not in classifierInferencePrediction at tile '+str(addr))
                if self.tileDictionary[addr]['classifierInferencePrediction'][classToThreshold] >= probabilityThreshold:
                    numTilesAboveProbThresh = numTilesAboveProbThresh + 1
            numTilesAboveProbThreshList.append(numTilesAboveProbThresh)

        if len(numTilesAboveProbThreshList) > 1:
            return numTilesAboveProbThreshList
        else:
            return numTilesAboveProbThreshList[0]

    def classifierMetricAtThreshold(self, classToThreshold, probabilityThresholds, tileAnnotationOverlapThreshold=0.5, metric="accuracy", assignZeroToTilesWithoutAnnotationOverlap=True):
        """A function to return the tile-level metric of a class probability
        threshold (or list of thresholds) compared to the ground truth, where a
        tile with ground truth annotation overlap greater than or equal to
        tileAnnotationOverlapThreshold is considered to be ground truth positive
        for that class. Ground truth annotations are expected to have been added
        to each tile in the tile dictionary by addAnnotations(). Class probability
        labels are expected to have been added to each tile in the tile dictionary
        by inferClassifier(). Metrics include 'accuracy', 'balanced_accuracy',
        'f1', 'precision', or 'recall'.

        Args:
            classToThreshold (string): the class to threshold the tiles by. The class must be already present in the tile dictionary from inferClassifier().
            probabilityThresholds (float or list of floats): the probability threshold or list of probability thresholds (in the range 0 to 1) to check. If a float is provided, just that probability threshold will be used, and a float of the accuracy of the classifier using that threshold as when the model considered a tile positive for the class will be returned. If a list of floats is provided, a list of floats of accuracies for those thresholds will be returned in respective order the inputted threshold list will be returned.
            tileAnnotationOverlapThreshold (float, optional): the class annotation overlap threshold at or above which a tile is considered ground truth positive for that class. Default is 0.5.
            metric (string, optional): which metric to compute. Options are 'accuracy', 'balanced_accuracy', 'f1', 'precision', or 'recall'. Default is 'accuracy'.
            assignZeroToTilesWithoutAnnotationOverlap (Boolean, optional): whether to assign a ground truth metric value of 0 (or else throw an error) for tiles that lack an overlap with classToThreshold in the ground truth annotations. Default is True.
        """

        if not hasattr(self, 'classifierPredictionTileAddresses'):
            foundPrediction = False
            classifierPredictionTileAddresses = []
            for tileAddress, tileEntry in self.tileDictionary.items():
                if 'classifierInferencePrediction' in tileEntry:
                    classifierPredictionTileAddresses.append(tileAddress)
                    foundPrediction = True
            if foundPrediction:
                self.classifierPredictionTileAddresses = classifierPredictionTileAddresses
            else:
                raise ValueError('No classification predictions found in Slide. Use inferClassifier() to generate them.')

        if not self.hasAnnotations():
            print("Warning: no annotations found in Slide. All tiles in Slide will be assumed to be negative for "+classToThreshold+". Run addAnnotations() if there should be annotations in this Slide.")
        #elif classToThreshold+'Overlap' not in self.tileDictionary[self.classifierPredictionTileAddresses[0]]:
        #    print("Warning: no annotations found in Slide. All tiles in Slide will be assumed to be negative for "+classToThreshold+". Run addAnnotations() if there should be annotations in this Slide.")
            #raise PermissionError(
            #    'addAnnotations must be called before extracting tiles')

        if type(probabilityThresholds) in [float, int]:
            pT = [probabilityThresholds]
        elif type(probabilityThresholds) == list:
            pT = probabilityThresholds
        else:
            raise ValueError('probabilityThresholds must be an int, float, or list of ints or floats')

        metrics = []

        for probabilityThreshold in pT:
            ground_truths = []
            predictions = []
            for predictionTileAddress in self.classifierPredictionTileAddresses:

                if classToThreshold+'Overlap' not in self.tileDictionary[predictionTileAddress]:
                    if assignZeroToTilesWithoutMetric:
                        ground_truths.append(0)
                    else:
                        raise ValueError(classToThreshold+' not found at tile '+str(predictionTileAddress))
                elif self.tileDictionary[predictionTileAddress][classToThreshold+'Overlap'] >= tileAnnotationOverlapThreshold:
                    ground_truths.append(1)
                else:
                    ground_truths.append(0)

                if classToThreshold not in self.tileDictionary[predictionTileAddress]['classifierInferencePrediction']:
                    raise ValueError(classToVisualize+' not in classifierInferencePrediction at tile '+str(predictionTileAddress))
                if self.tileDictionary[predictionTileAddress]['classifierInferencePrediction'][classToThreshold] >= probabilityThreshold:
                    predictions.append(1)
                else:
                    predictions.append(0)

            if metric == "accuracy":
                metrics.append(accuracy_score(ground_truths, predictions))
            elif metric == "balanced_accuracy":
                metrics.append(balanced_accuracy_score(ground_truths, predictions))
            elif metric == "f1":
                metrics.append(f1_score(ground_truths, predictions))
            elif metric == "precision":
                metrics.append(precision_score(ground_truths, predictions))
            elif metric == "recall":
                metrics.append(recall_score(ground_truths, predictions))
            else:
                raise ValueError("metric must be one of: 'accuracy', 'balanced_accuracy', 'f1', 'precision', or 'recall'")

        if len(metrics) > 1:
            return metrics
        else:
            return metrics[0]

########################################################################################

    def segmenterMetricAtThreshold(self, classToThreshold, probabilityThresholds, metric="dice_coeff"):
        """A function to return the pixel-level metric of a class probability
        threshold (or list of thresholds) compared to the ground truth, where a
        pixel with ground truth annotation overlap greater than or equal to
        probabilityThresholds is considered to be ground truth positive
        for that class. Ground truth annotations are expected to have been added
        to each tile in the tile dictionary by addAnnotations(). Class probability
        labels are expected to have been added to each tile in the tile dictionary
        by inferSegmenter(). The only metric currently available is the Dice
        coefficient. The metric will be applied to all tiles with predictions
        added by inferSegmenter() and the average of that metric across those
        tiles will be returned to give one metric per slide per threshold in
        probabilityThresholds.

        Args:
            classToThreshold (string): the class to threshold the pixels by. The class must be already present in the tile dictionary from inferSegmenter().
            probabilityThresholds (float or list of floats): the probability threshold or list of probability thresholds (in the range 0 to 1) to check. If a float is provided, just that probability threshold will be used, and a float of the accuracy of the segmenter using that threshold as when the model considered a pixel positive for the class will be returned. If a list of floats is provided, a list of floats of accuracies for those thresholds will be returned in respective order the inputted threshold list will be returned.
            metric (string, optional): which metric to compute. The only option currently available is 'dice_coeff'. Default is 'dice_coeff'.
        """

        if not hasattr(self, 'segmenterPredictionTileAddresses'):
            foundPrediction = False
            segmenterPredictionTileAddresses = []
            for tileAddress, tileEntry in self.tileDictionary.items():
                if 'segmenterInferencePrediction' in tileEntry:
                    segmenterPredictionTileAddresses.append(tileAddress)
                    foundPrediction = True
            if foundPrediction:
                self.segmenterPredictionTileAddresses = segmenterPredictionTileAddresses
            else:
                raise ValueError('No segmentation predictions found in Slide. Use inferSegmenter() to generate them.')

        if not self.hasAnnotations():
            print("Warning: no annotations found in Slide. All tiles in Slide will be assumed to be negative for "+classToThreshold+". Run addAnnotations() if there should be annotations in this Slide.")

        if type(probabilityThresholds) in [float, int]:
            pT = [probabilityThresholds]
        elif type(probabilityThresholds) == list:
            pT = probabilityThresholds
        else:
            raise ValueError('probabilityThresholds must be an int, float, or list of ints or floats')

        metrics = []
        ground_truth_masks_np = []

        if metric == 'dice_coeff':
            for predictionTileAddress in self.segmenterPredictionTileAddresses:
                ground_truth_masks_np.append(self.getAnnotationTileMask(predictionTileAddress, classToThreshold, writeToNumpy=True, acceptTilesWithoutClass=True))
                #ground_truth_mask = torch.from_numpy(self.getAnnotationTileMask(predictionTileAddress, classToThreshold, writeToNumpy=True, acceptTilesWithoutClass=True)).float()
                #ground_truth_mask = torch.div(ground_truth_mask, 255) # getAnnotationTileMask outputs 0-255 arrays, so we need to normalize to be 0-1 range
                #ground_truth_masks.append(ground_truth_mask)
            print("finished making ground truth masks")

            for probabilityThreshold in pT:
                #binarized_prediction_masks = []
                dice_scores_at_threshold = []
                for i, predictionTileAddress in enumerate(self.segmenterPredictionTileAddresses):
                    binarized_prediction_mask = (torch.from_numpy(self.tileDictionary[predictionTileAddress]['segmenterInferencePrediction'][classToThreshold]) > probabilityThreshold).float()
                    ground_truth_mask_torch = torch.div(torch.from_numpy(ground_truth_masks_np[i]).float(), 255)
                    dice_scores_at_threshold.append(dice_coeff(binarized_prediction_mask, ground_truth_mask_torch).item())

                print("Mean Dice score at threshold "+str(probabilityThreshold)+":", np.mean(dice_scores_at_threshold))
                metrics.append(np.mean(dice_scores_at_threshold))



                #mean_dice_at_threshold = dice_coeff(torch.stack(binarized_prediction_masks), torch.stack(ground_truth_masks)).item()
                #print("Mean Dice score at threshold "+str(probabilityThreshold)+":", mean_dice_at_threshold)
                #metrics.append(mean_dice_at_threshold)








            #for probabilityThreshold in pT:
            #    tds = []
            #    for predictionTileAddress in self.segmenterPredictionTileAddresses:
                    #print(predictionTileAddress)
            #        tds.append(self.getTileDiceScore(predictionTileAddress, classToThreshold, pixelBinarizationThreshold=probabilityThreshold))
                    #print("Dice score:", tds)
            #    metrics.append(np.mean(tds))
            #    print("Average Dice score thresholding at "+str(probabilityThreshold)+":", np.mean(tds))
        else:
            raise ValueError("metric must be one of: 'dice_coeff'")

        if len(metrics) > 1:
            return metrics
        else:
            return metrics[0]
