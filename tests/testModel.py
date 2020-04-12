import sys
sys.path.append('/Users/gehrun01/Desktop/pathml')
from pathml.slide import Slide
from pathml.analysis import Analysis
from pathml.processor import Processor
from pathml.models.tissuedetector import tissueDetector
import matplotlib.pyplot as plt


pathmlSlide = Slide('/Users/gehrun01/Desktop/BEST2_CAM_0654_HE_1.svs').setTileProperties(tileSize=224)

# Needs to be at 40x
tissueForegroundProcessor = Processor(Slide('/Users/gehrun01/Desktop/BEST2_CAM_0654_HE_1.svs',level=1).setTileProperties(tileSize=448,tileOverlap=0.5))
theGoodStuff = tissueForegroundProcessor.applyModel(tissueDetector(), batch_size=20, predictionKey='tissue_detector').tileDictionary
pathmlSlide.adoptTileDictionary('tissue_detector', theGoodStuff, 'tissue_detector', upsampleFactor=4)

print(theGoodStuff.tileDictionary)
quit()
testAnalysis = Analysis(pathmlSlide.tileDictionary)
mapClass0=testAnalysis.generateInferenceMap(predictionSelector=0,predictionKey='foreground')
mapClass1=testAnalysis.generateInferenceMap(predictionSelector=1,predictionKey='foreground')
mapClass2=testAnalysis.generateInferenceMap(predictionSelector=2,predictionKey='foreground')

plt.figure()
plt.imshow(mapClass0,vmin=0, vmax=1)
plt.colorbar()
plt.title('Class 0')
plt.show(block=False)
plt.figure()
plt.imshow(mapClass1,vmin=0, vmax=1)
plt.colorbar()
plt.title('Class 1')
plt.show(block=False)
plt.figure()
plt.imshow(mapClass2,vmin=0, vmax=1)
plt.colorbar()
plt.title('Class 2')
plt.show()
