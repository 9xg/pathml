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
pv.cache_set_max(0)


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
        return self

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
    def iterateTiles(self, tileDictionary=False, includeImage=False, writeToNumpy=False):
        tileDictionaryIterable = self.tileDictionary if not tileDictionary else tileDictionary
        for key, value in tileDictionaryIterable.items():
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

    # ADAM EXPERIMENTAL
    # Adds the overlap between all (desired) classes present in an annotation file and each tile in the tile dictionary
    def addAnnotations(self, annotationFilePath, fileType, classesToAdd='all', magnificationLevel=0):
        if fileType != ['asap', 'Asap', 'ASAP', 'qupath', 'Qupath', 'QuPath', 'QUPATH']:
            raise ValueError('fileType must be ASAP or QuPath')
        if fileType in ['qupath', 'Qupath', 'QuPath', 'QUPATH']: # REMOVE ONCE FIXED
            raise ValueError('QuPath annotation files not currently supported')
        if (not type(magnificationLevel) == int) or (magnificationLevel < 0):
            raise ValueError('magnificationLevel must be an integer 0 or greater')
        if 'openslide.level['+str(magnificationLevel)+'].downsample' not in self.slideProperties:
            raise ValueError('magnificationLevel not present in slide')
        if (classesToAdd != 'all') and (not isinstance(classesToAdd, list)):
            raise ValueError("classestoAdd must be 'all' or a list")
        if not os.path.isfile(annotationFilePath):
            raise FileNotFoundError('Annotation file could not be loaded')
        try:
            tree = ET.parse(wsi_annotations[i])
        except:
            raise ImportError('Annotation file is not an xml file')

        # ADD QUPATH GEOJSON SUPPORT

        root = tree.getroot() # Get root of .xml tree
        if not (root.tag == "ASAP_Annotations"_: # Check whether we actually deal with an ASAP .xml file
            raise ImportError('Annotation file is not an ASAP xml file')

        allAnnotations = root.find('Annotations') # Find all annotations for this slide
        print('xml file valid - ' + str(len(allAnnotations)) + ' annotations found.') # Display number of found annotations

        slideHeight = int(self.slideProperties['height'])
        annotationScalingFactor = float(self.slideProperties['openslide.level[0].downsample'])/float(self.slideProperties['openslide.level['+str(magnificationLevel)+'].downsample'])
        print("Scale: "+str(annotationScalingFactor))

        # Iterate over all annotations to collect annotations in the same class
        class_polys = {}
        for annotation in allAnnotations:
            annotation_class = annotation.attrib['PartOfGroup']
            if (classesToAdd != 'all') and (annotation_class not in classesToAdd):
                continue # skip annotations from classes not in the classesToAdd list

            if annotation_class not in class_polys:
                class_polys[annotation_class] = []

            annotationTree = annotation.find('Coordinates')
            polygon = []
            for coordinate in annotationTree:
                info = coordinate.attrib
                polygon.append((float(info['X'])*annotationScalingFactor, float(info['Y'])*annotationScalingFactor))
            polygonNp = np.asarray(polygon)
            polygonNp[:,1] = slideHeight-polygonNp[:,1]
            poly = geometry.Polygon(polygonNp).buffer(0)
            class_polys[annotation_class].append(poly)

        # Make a Shapely MultiPolygon for each class
        class_multipolys = {ancl:geometry.MultiPolygon(ply) for (ancl,ply) in class_polys.items()}

        # Iterate over all tiles in tile dictionary, marking the overlap of each class MultiPolygon with each tile
        for address in self.iterateTiles():
            x = self.tileDictionary[address]['x']
            y = self.tileDictionary[address]['y']
            height = self.tileDictionary[address]['height']
            tile = geometry.box(x, (slideHeight-y)-height, x+height, slideHeight-y)

            for class_name, class_multipoly in class_multipolys.items():
                tile_class_overlap = tile.intersection(class_multipoly).area/(height**2)
                self.tileDictionary[address].update({class_name+'Overlap': tile_class_overlap})

    # ADAM EXPERIMENTAL
    # Extract tiles into directory structure amenable to torch.utils.data.ConcatDataset
    # extractionDirectory is expected to be of the form: '/path/to/tiles'
    # classesToExtract is expected to be 'all' or a list
    # tileAnnotationOverlapThreshold is expected to be a number greater than 0 and less than or equal to 1,
    #   or a dictionary of such values, with a key for each class to extract
    def extractAnnotationTiles(self, extractionDirectory, caseId='getFromSlideFilePath', classesToExtract='all', tileAnnotationOverlapThreshold=0.5, extractForegroundTilesOnly=False, extractTissueTilesOnly=False, tissueLevelThreshold=0.5):

        # get case ID
        if caseId == 'getFromSlideFilePath':
            id = os.path.basename(self.__slideFilePath).split('.')[0]
        elif type(caseId) == str:
            id = caseId
        else:
            raise ValueError("caseId must be 'all' or a string")

        # get classes to extract
        extractionClasses = []
        if classesToExtract == 'all':
            for key, value in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                if 'Overlap' in key:
                    extractionClasses.append(key)
        elif type(classesToExtract) == list:
            extractionClasses = [classToExtract+'Overlap' for classToExtract in classesToExtract]
            for extractionClass in extractionClasses:
                if extractionClass not in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
                    raise ValueError(extractionClass+' not found in tile dictionary')
        else:
            raise ValueError("classesToExtract must be 'all' or a list of class names")
        extractionClasses = [extractionClass.split('Overlap')[0] for extractionClass in extractionClasses]
        print('Found '+str(len(extractionClasses))+' classes to extract:', extractionClasses)

        # Convert annotationOverlapThreshold into a dictionary (if necessary)
        annotationOverlapThresholdDict = {}
        if (type(tileAnnotationOverlapThreshold) == int) or (type(tileAnnotationOverlapThreshold) == float):
            if (tileAnnotationOverlapThreshold <= 0) or (tileAnnotationOverlapThreshold > 1):
                raise ValueErrorr('tileAnnotationOverlapThreshold must be greater than 0 and less than or equal to 1')
            for extractionClass in extractionClasses:
                annotationOverlapThresholdDict[extractionClass] = tileAnnotationOverlapThreshold
        elif type(tileAnnotationOverlapThreshold) == dict:
            for ec, taot in tileAnnotationOverlapThreshold.items():
                if ec not in extractionClasses:
                    raise ValueError('Class '+str(ec)+' present as a key in tileAnnotationOverlapThreshold but not present in classes to extract')
                if ((type(taot) != int) and (type(taot) != float)) or ((taot <= 0) or (taot > 1)):
                    raise ValueError('Tile annotation overlap threshold of class '+str(ec)+' must be a number greater than zero and less than or equal to 1')
            for extractionClass in extractionClasses:
                if extractionClass not in tileAnnotationOverlapThreshold:
                    raise ValueError('Class '+str(extractionClass)+' present in classes to extract but not present as a key in tileAnnotationOverlapThreshold')
            annotationOverlapThresholdDict = tileAnnotationOverlapThreshold
        else:
            raise ValueError('tileAnnotationOverlapThreshold must be a dictionary or number greater than 0 and less than or equal to 1')

        # Create empty class tile directories
        for extractionClass in extractionClasses:
            os.path.makdirs(os.path.join(extractionDirectory, id, extractionClass), exist_ok=True)

        if ((type(tissueLevelThreshold) != int) and (type(tissueLevelThreshold) != float)) or ((tissueLevelThreshold <= 0) or (tissueLevelThreshold > 1)):
            raise ValueError('tissueLevelThreshold must be a number greater than zero and less than or equal to 1')

        # Extract tiles
        for address in self.iterateTiles():
            if extractTissueTilesOnly:
                if 'tissueLevel' not in self.tileDictionary[address]:
                    raise ValueError('Deep tissue detection must be performed before extractTissueTilesOnly can be set to True')
                if self.tileDictionary[address]['tissueLevel'] >= tissueLevelThreshold: # do not extract background and artifact tiles
                    for extractionClass in extractionClasses:
                        if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
                            area = self.getTile(address)
                            area.write_to_file(os.path.join(extractionDirectory, id,
                                extractionClass, id+'_'+str(self.tileDictionary[address]['x'])+'x_'+str(self.tileDictionary[address]['y'])+'y'+'_'+str(self.tileDictionary[address]['height'])+'tilesize.jpg'), Q=100)

            elif extractForegroundTilesOnly:
                if 'foreground' not in self.tileDictionary[address]:
                    raise ValueError('Foreground detection must be performed with detectForeground() before extractForegroundTilesOnly can be set to True')
                if self.tileDictionary[address]['foreground']:
                    for extractionClass in extractionClasses:
                        if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
                            area = self.getTile(address)
                            area.write_to_file(os.path.join(extractionDirectory, id,
                                extractionClass, id+'_'+str(self.tileDictionary[address]['x'])+'x_'+str(self.tileDictionary[address]['y'])+'y'+'_'+str(self.tileDictionary[address]['height'])+'tilesize.jpg'), Q=100)

            else:
                for extractionClass in extractionClasses:
                    if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
                        area = self.getTile(address)
                        area.write_to_file(os.path.join(extractionDirectory, id,
                            extractionClass, id+'_'+str(self.tileDictionary[address]['x'])+'x_'+str(self.tileDictionary[address]['y'])+'y'+'_'+str(self.tileDictionary[address]['height'])+'tilesize.jpg'), Q=100)


    # ADAM EXPERIMENTAL
    def extractRandomUnannotatedTiles(self, extractionDirectory, numTilesToExtract=50, unannotatedClassName='unannotated', caseId='getFromSlideFilePath', extractForegroundTilesOnly=False, extractTissueTilesOnly=False, tissueLevelThreshold=0.5, seed=366):
        random.seed(366)
