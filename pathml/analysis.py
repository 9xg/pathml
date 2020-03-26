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

class Analysis:

    __verbosePrefix = '[PathML] '

    def __init__(self, tileDictionaryFilePath, verbose=False):
        # pv.cache_set_max(0)
        # pv.leak_set(True)

        self.__verbose = verbose
        self.__tileDictionaryFilePath = tileDictionaryFilePath
        try:
            if self.__verbose:
                print(self.__verbosePrefix + "Loading " + self.__tileDictionaryFilePath)
            self.tileDictionary = pickle.load( open( tileDictionaryFilePath, "rb" ) )
        except:
            raise FileNotFoundError('Tile dictionary could not be loaded')
        else:
            if self.__verbose:
                print(self.__verbosePrefix + "Successfully loaded")
            self.numTilesInX = max([key[0] for key in self.tileDictionary.keys()])+1
            self.numTilesInY = max([key[1] for key in self.tileDictionary.keys()])+1

    def iterateTiles(self):
        for key, value in self.tileDictionary.items():
            yield key

    def generateInferenceMap(self, predictionSelector):
        predictionMap = np.zeros([self.numTilesInY, self.numTilesInX])
        for address in self.iterateTiles():
            if 'prediction' in self.tileDictionary[address]:
                predictionMap[address[1], address[0]] = float(self.tileDictionary[address]['prediction'][predictionSelector])
        if not np.any(predictionMap):
            raise ValueError('Generated inference map is empty. No predictions were found for the provided prediction selector. Please check the presence of relevant tags in the tile dictionary.')
        return predictionMap

    def generateForegroundMap(self):
        foregroundBinaryMask = np.zeros([self.numTilesInY, self.numTilesInX])
        for address in self.iterateTiles():
            #print(address)
            foregroundBinaryMask[address[1], address[0]] = int(self.tileDictionary[address]['foreground'] is True)
        return foregroundBinaryMask
