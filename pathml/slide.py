# This is the experimental version of slide with annotation and tile extraction
# capabilities

import numpy as np
import pyvips as pv
from PIL import Image, ImageDraw
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.morphology import binary_dilation, remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray, rgb2lab
from skimage.transform import resize
import matplotlib.pyplot as plt
from pathlib import Path
from pathml.processor import Processor
from pathml.models.tissuedetector import tissueDetector
import xml.etree.ElementTree as ET
from shapely import geometry
from tqdm import tqdm
import random
import json
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

    def detectForeground(self, threshold, level=2):
        if not hasattr(self, 'tileDictionary'):
            raise PermissionError(
                'setTileProperties must be called before foreground detection')
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
        """
        Returns a pyvips Region or, if writeToNumpy is set to true, a numpy array
        """
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


    def saveTileDictionary(self, fileName, folder=os.getcwd()):
        pickle.dump(self.tileDictionary, open(os.path.join(folder, fileName)+'.pml', 'wb'))

    def saveSelf(self, fileName=False, folder=os.getcwd()):
        if not hasattr(self, 'tileDictionary'):
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
            if hasattr(self, 'annotationClassMultiPolygons'):
                outputDict = {'slideFilePath': self.slideFilePath, 'tileDictionary': self.tileDictionary, 'rawTissueDetectionMap': self.rawTissueDetectionMap, 'annotationClassMultiPolygons': self.annotationClassMultiPolygons}
            else:
                outputDict = {'slideFilePath': self.slideFilePath, 'tileDictionary': self.tileDictionary, 'rawTissueDetectionMap': self.rawTissueDetectionMap}
        else:
            if hasattr(self, 'annotationClassMultiPolygons'):
                outputDict = {'slideFilePath': self.slideFilePath, 'tileDictionary': self.tileDictionary, 'annotationClassMultiPolygons': self.annotationClassMultiPolygons}
            else:
                outputDict = {'slideFilePath': self.slideFilePath, 'tileDictionary': self.tileDictionary}

        pickle.dump(outputDict, open(os.path.join(folder, id)+'.pml', 'wb'))

    def appendTag(self, tileAddress, key, val):
        if not hasattr(self, 'tileDictionary'):
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
    def addAnnotations(self, annotationFilePath, classesToAdd=False, negativeClass=False, magnificationLevel=0, overwriteExistingAnnotations=False):
        """
        Adds the overlap between all (desired) classes present in an annotation file and each tile in the tile dictionary
        Annotations within groups in ASAP are taken to be within one class, where the name of the group is the name of the class
        annotationFilePath must point to either an xml file from the ASAP software or a GeoJSON file from the QuPath software
        classesToAdd is a list of classes to add from the annotation file. If not specified, all annotation classes will be used (except the negativeClass if one is specified)
        negativeClass is the name of the class of negative annotations (donut holes) to subtract from the other annotations
        """

        #if fileType != ['asap', 'Asap', 'ASAP', 'qupath', 'Qupath', 'QuPath', 'QUPATH']:
        #    raise ValueError('fileType must be ASAP or QuPath')
        #if fileType in ['qupath', 'Qupath', 'QuPath', 'QUPATH']: # REMOVE ONCE FIXED
        #    raise ValueError('QuPath annotation files not currently supported')
        if not hasattr(self, 'tileDictionary'):
            raise PermissionError(
                'setTileProperties must be called before adding annotations')

        foundOverlap = False
        for k in self.tileDictionary[list(self.tileDictionary.keys())[0]]:
            if 'Overlap' in k:
                foundOverlap = True
                if not overwriteExistingAnnotations:
                    raise Warning('Annotatons have already been added to the tile dictionary. Use overwriteExistingAnnotations if you wish to write over them')
        if foundOverlap:
            for address in self.iterateTiles():
                for key in self.tileDictionary[address].copy():
                    if 'Overlap' in key:
                        del self.tileDictionary[address][key]

        if (not type(magnificationLevel) == int) or (magnificationLevel < 0):
            raise ValueError('magnificationLevel must be an integer 0 or greater')
        if 'openslide.level['+str(magnificationLevel)+'].downsample' not in self.slideProperties:
            raise ValueError('magnificationLevel not present in slide')
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
        annotationScalingFactor = float(self.slideProperties['openslide.level[0].downsample'])/float(self.slideProperties['openslide.level['+str(magnificationLevel)+'].downsample'])
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
            print('xml file valid - ' + str(len(allAnnotations)) + ' annotations found.') # Display number of found annotations

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
                poly = geometry.Polygon(polygonNp).buffer(0)

                if (negativeClass) and (annotationClass == negativeClass):
                    negativePolys.append(poly)
                else:
                    classPolys[annotationClass].append(poly)

        # Turn QuPath GeoJSON annotations into a dict of multipolygons, with one multipolygon for each class
        else:
            with open(annotationFilePath) as f:
                allAnnotations = json.load(f)
            if not isinstance(allAnnotations, list):
                raise Warning('GeoJSON file does not have an outer list structure')
            else:
                print('JSON file valid - ' + str(len(allAnnotations)) + ' annotations found.')

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
                poly = geometry.Polygon(polygonNp).buffer(0)

                if (negativeClass) and (annotationClass == negativeClass):
                    negativePolys.append(poly)
                else:
                    classPolys[annotationClass].append(poly)


        # Make a Shapely MultiPolygon for each class
        classMultiPolys = {ancl:geometry.MultiPolygon(ply) for (ancl,ply) in classPolys.items()}

        if (negativeClass) and (len(negativePolys) == 0):
            print('Warning: 0 '+negativeClass+' annotations found')
        elif (negativeClass):
            print(str(len(negativePolys))+' '+negativeClass+' annotations found')

        # Remove negativeClass polygons from each class multipolygon
        for ancl in classMultiPolys:
            for nply in negativePolys:
                if classMultiPolys[ancl].intersects(nply):
                    classMultiPolys[ancl] = classMultiPolys[ancl].difference(nply)


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
    def getAnnotationTileMask(self, tileAddress, maskClass, verbose=False):
        """
        Returns a PIL image
        """

        if not hasattr(self, 'tileDictionary'):
            raise PermissionError('setTileProperties must be called before extracting tiles')
        if not hasattr(self, 'annotationClassMultiPolygons'):
            raise PermissionError('addAnnotations must be called before extracting tiles')
        if tileAddress not in self.tileDictionary:
            raise ValueError('tileAddress must be in tileDictionary')
        if maskClass not in self.annotationClassMultiPolygons:
            raise ValueError(maskClass+' not in annotationClassMultiPolygons')

        slideHeight = int(self.slideProperties['height'])
        x = self.tileDictionary[tileAddress]['x']
        y = self.tileDictionary[tileAddress]['y']
        height = self.tileDictionary[tileAddress]['height']
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


    # ADAM EXPERIMENTAL
    def extractAnnotationTiles(self, pathToTileDir, tileDirName=False, numTilesToExtractPerClass='all', classesToExtract='all', extractSegmentationMasks=False, tileAnnotationOverlapThreshold=0.5, extractForegroundTilesOnly=False, extractTissueTilesOnly=False, tissueLevelThreshold=0.5, seed=False):
        """
        Extract tiles that overlap with annotations into directory structure amenable to torch.utils.data.ConcatDataset
        pathToTileDir is expected to be of the form: '/path/to/tiles'
        numTilesToExtractPerClass is expected to be positive integer, a dictionary with class names as keys and positive integers as values, or 'all' to extract all suitable tiles for each class
        classesToExtract is expected to be 'all' or a list
        tileAnnotationOverlapThreshold is expected to be a number greater than 0 and less than or equal to 1, or a dictionary of such values, with a key for each class to extract
        """

        if not hasattr(self, 'tileDictionary'):
            raise PermissionError(
                'setTileProperties must be called before extracting tiles')

        if not hasattr(self, 'annotationClassMultiPolygons'):
            raise PermissionError(
                'addAnnotations must be called before extracting tiles')

        if extractForegroundTilesOnly and extractTissueTilesOnly:
            raise ValueError('Only one of extractTissueTilesOnly and extractForegroundTilesOnly can be set to True, not both')

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
        if classesToExtract == 'all':
            for key, value in self.tileDictionary[list(self.tileDictionary.keys())[0]].items():
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
        for address in self.iterateTiles():
            if extractTissueTilesOnly:
                if 'tissueLevel' not in self.tileDictionary[address]:
                    raise ValueError('Deep tissue detection must be performed with detectTissue() before extractTissueTilesOnly can be set to True')
                if self.tileDictionary[address]['tissueLevel'] >= tissueLevelThreshold: # do not extract background and artifact tiles
                    for extractionClass in extractionClasses:
                        if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
                            annotatedTileAddresses[extractionClass].append(address)

            elif extractForegroundTilesOnly:
                if 'foreground' not in self.tileDictionary[address]:
                    raise ValueError('Foreground detection must be performed with detectForeground() before extractForegroundTilesOnly can be set to True')
                if self.tileDictionary[address]['foreground']:
                    for extractionClass in extractionClasses:
                        if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
                            annotatedTileAddresses[extractionClass].append(address)

            else:
                for extractionClass in extractionClasses:
                    if self.tileDictionary[address][extractionClass+'Overlap'] >= annotationOverlapThresholdDict[extractionClass]:
                        annotatedTileAddresses[extractionClass].append(address)

        annotatedTilesToExtract = {} #{extractionClass: [] for extractionClass in extractionClasses}
        if type(numTilesToExtractPerClass) == int:
            if numTilesToExtractPerClass <= 0:
                raise ValueError('If numTilesToExtractPerClass is an integer, it must be greater than 0')
            for extractionClass in extractionClasses:
                if len(annotatedTileAddresses[extractionClass]) == 0:
                    raise Warning('0 suitable '+extractionClass+' tiles found')
                if len(annotatedTileAddresses[extractionClass]) < numTilesToExtractPerClass:
                    print('Warning: '+str(len(annotatedTileAddresses[extractionClass]))+' suitable '+extractionClass+' tiles found but requested '+str(numTilesToExtractPerClass)+' tiles to extract. Extracting all suitable tiles...')
                    annotatedTilesToExtract[extractionClass] = annotatedTileAddresses[extractionClass]
                else:
                    annotatedTilesToExtract[extractionClass] = random.sample(annotatedTileAddresses[extractionClass], numTilesToExtractPerClass)
            #annotatedTilesToExtract = {extractionClass: random.sample(annotatedTileAddresses[extractionClass], numTilesToExtractPerClass) for extractionClass in extractionClasses}
        elif numTilesToExtractPerClass == 'all':
            for extractionClass in extractionClasses:
                if len(annotatedTileAddresses[extractionClass]) == 0:
                    raise Warning('0 suitable '+extractionClass+' tiles found')
                if len(annotatedTileAddresses[extractionClass]) > 500:
                    print('Warning: '+str(len(annotatedTileAddresses[extractionClass]))+' suitable '+extractionClass+' tiles found')
                annotatedTilesToExtract[extractionClass] = annotatedTileAddresses[extractionClass]

        elif type(numTilesToExtractPerClass) == dict:
            for ec,tc in numTilesToExtractPerClass.items():
                if ec not in extractionClasses:
                    raise Warning('Class '+ec+' present as a key in numTilesToExtractPerClass dictionary but absent from the tileDictionary')
            for extractionClass in extractionClasses:
                if len(annotatedTileAddresses[extractionClass]) == 0:
                    raise Warning('0 suitable '+extractionClass+' tiles found')
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

        # Create empty class tile directories
        for extractionClass in extractionClasses:
            try:
                os.makedirs(os.path.join(pathToTileDir, id, extractionClass), exist_ok=True)
            except:
                raise ValueError(os.path.join(pathToTileDir, id, extractionClass)+' is not a valid path')

        # Extract tiles
        for ec,tte in annotatedTilesToExtract.items():
            if extractSegmentationMasks:
                print("Extracting "+str(len(tte))+" of "+str(len(annotatedTileAddresses[ec]))+" "+ec+" tiles and segmentation masks...")
            else:
                print("Extracting "+str(len(tte))+" of "+str(len(annotatedTileAddresses[ec]))+" "+ec+" tiles...")
            for tl in tte:
                area = self.getTile(tl)
                area.write_to_file(os.path.join(pathToTileDir, id, ec,
                    id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize.jpg'), Q=100)
                if extractSegmentationMasks:
                    mask = self.getAnnotationTileMask(tl, ec)
                    mask.save(os.path.join(pathToTileDir, id, ec,
                        id+'_'+ec+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_mask.gif'))



            # support per class number of tiles to extract
            # raise warning if fewer than number found in a slide's class
            # raise warning if no slides found in a slide's class
            # print how many of each tile are being extracted for each class
            # numTilesToExtractPerClass (allow dictionary?), can be 'all'
            # automatically detect whether it is an asap xml or qupath geojson (and if not throw error)

    # ADAM EXPERIMENTAL
    def extractRandomUnannotatedTiles(self, pathToTileDir, tileDirName=False, numTilesToExtract=50, unannotatedClassName='unannotated', extractSegmentationMasks=False, extractForegroundTilesOnly=False, extractTissueTilesOnly=False, tissueLevelThreshold=0.5, seed=False):
        """
        Extract randomly selected tiles that don't overlap any annotations into directory structure amenable to torch.utils.data.ConcatDataset
        pathToTileDir is expected to be of the form: '/path/to/tiles'
        Note: all segmentation masks extracted for this class will obviously be blank
        """

        if not hasattr(self, 'tileDictionary'):
            raise PermissionError(
                'setTileProperties must be called before extracting tiles')

        if not hasattr(self, 'annotationClassMultiPolygons'):
            raise PermissionError(
                'addAnnotations must be called before extracting tiles')

        if extractForegroundTilesOnly and extractTissueTilesOnly:
            raise ValueError('Only one of extractTissueTilesOnly and extractForegroundTilesOnly can be set to True, not both')

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
        for address in self.iterateTiles():
            if extractTissueTilesOnly:
                if 'tissueLevel' not in self.tileDictionary[address]:
                    raise ValueError('Deep tissue detection must be performed with detectTissue() before extractTissueTilesOnly can be set to True')
                if self.tileDictionary[address]['tissueLevel'] >= tissueLevelThreshold: # do not extract background and artifact tiles
                    overlapsAnnotation = False
                    for annotationClass in annotationClasses:
                        if self.tileDictionary[address][annotationClass] > 0:
                            overlapsAnnotation = True
                            break
                    if not overlapsAnnotation:
                        unannotatedTileAddresses.append(address)

            elif extractForegroundTilesOnly:
                if 'foreground' not in self.tileDictionary[address]:
                    raise ValueError('Foreground detection must be performed with detectForeground() before extractForegroundTilesOnly can be set to True')
                if self.tileDictionary[address]['foreground']:
                    overlapsAnnotation = False
                    for annotationClass in annotationClasses:
                        if self.tileDictionary[address][annotationClass] > 0:
                            overlapsAnnotation = True
                            break
                    if not overlapsAnnotation:
                        unannotatedTileAddresses.append(address)

            else:
                overlapsAnnotation = False
                for annotationClass in annotationClasses:
                    if self.tileDictionary[address][annotationClass] > 0:
                        overlapsAnnotation = True
                        break
                if not overlapsAnnotation:
                    unannotatedTileAddresses.append(address)

        if len(unannotatedTileAddresses) == 0:
            raise Warning('0 unannotated tiles found')
        if len(unannotatedTileAddresses) < numTilesToExtract:
            print('Warning: '+str(len(unannotatedTileAddresses))+' unannotated tiles found but requested '+str(numTilesToExtract)+' tiles to extract. Extracting all suitable tiles...')
            unannotatedTilesToExtract = unannotatedTileAddresses
        else:
            unannotatedTilesToExtract = random.sample(unannotatedTileAddresses, numTilesToExtract)
            #print(str(len(unannotatedTileAddresses))+' unannotated tiles found in total')

        # Create empty class tile directory
        try:
            os.makedirs(os.path.join(pathToTileDir, id, unannotatedClassName), exist_ok=True)
        except:
            raise ValueError(os.path.join(pathToTileDir, id, unannotatedClassName)+' is not a valid path')

        # Extract the desired number of unannotated tiles

        if extractSegmentationMasks:
            print("Extracting "+str(len(unannotatedTilesToExtract))+" of "+str(len(unannotatedTileAddresses))+" "+unannotatedClassName+" tiles and segmentation masks...")
        else:
            print("Extracting "+str(len(unannotatedTilesToExtract))+" of "+str(len(unannotatedTileAddresses))+" "+unannotatedClassName+" tiles...")
        #print("Tiles to extract:", unannotatedTilesToExtract)
        for tl in unannotatedTilesToExtract:
            #print("On tile:", tl)
            area = self.getTile(tl)
            area.write_to_file(os.path.join(pathToTileDir, id, unannotatedClassName,
                id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize.jpg'), Q=100)
            if extractSegmentationMasks:
                height = self.tileDictionary[tl]['height']
                mask = Image.new('1', (height, height), 0) # blank mask
                mask.save(os.path.join(pathToTileDir, id, unannotatedClassName,
                    id+'_'+unannotatedClassName+'_'+str(self.tileDictionary[tl]['x'])+'x_'+str(self.tileDictionary[tl]['y'])+'y'+'_'+str(self.tileDictionary[tl]['height'])+'tilesize_mask.jpg'))



    # ADAM EXPERIMENTAL
    def detectTissue(self, tissueDetectionLevel=1, tissueDetectionTileSize=512, tissueDetectionTileOverlap=0, tissueDetectionUpsampleFactor=4, batchSize=20, overwriteExistingTissueDetection=False):
        if not hasattr(self, 'tileDictionary'):
            raise PermissionError(
                'setTileProperties must be called before applying tissue detector')
        if hasattr(self, 'rawTissueDetectionMap') and (not overwriteExistingTissueDetection):
            raise Warning('Tissue detection has already been performed. Use overwriteExistingTissueDetection if you wish to write over it')

        print("Detecting tissue of "+self.slideFilePath)
        tissueForegroundSlide = Slide(self.slideFilePath, level=tissueDetectionLevel).setTileProperties(tileSize=tissueDetectionTileSize, tileOverlap=tissueDetectionTileOverlap) # tile size and overlap for tissue detector, not final tiles
        tmpProcessor = Processor(tissueForegroundSlide)
        tissueForegroundTmp = tmpProcessor.applyModel(tissueDetector(), batch_size=batchSize, predictionKey='tissue_detector').adoptKeyFromTileDictionary(upsampleFactor=tissueDetectionUpsampleFactor)

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

    # ADAM EXPERIMENTAL
    def plotResizedTissueDetectionMap(self, fileName=False, folder=os.getcwd()):
        if not hasattr(self, 'tileDictionary'):
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
        plt.imsave(os.path.join(folder, id+'.png'), map)
