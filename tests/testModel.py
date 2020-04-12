import sys
sys.path.append('/home/gehrun01/Desktop/pathml')
from pathml.slide import Slide
from pathml.analysis import Analysis
from pathml.processor import Processor
from pathml.models.tissuedetector import tissueDetector
import matplotlib.pyplot as plt


pathmlSlide = Slide('/media/gehrun01/Cytosponge-Store/miRNA-1/Biopsies/OCCAMS_AH_183_OAC_Biopsy.svs').setTileProperties(tileSize=224)
tissueForegroundProcessor = Processor(Slide('/media/gehrun01/Cytosponge-Store/miRNA-1/Biopsies/OCCAMS_AH_183_OAC_Biopsy.svs',level=1).setTileProperties(tileSize=448,tileOverlap=0.5))
theGoodStuff = tissueForegroundProcessor.applyModel(tissueDetector(), batch_size=20, predictionKey='tissue_detector')
print(theGoodStuff.adoptKeyFromTileDictionary(upsampleFactor=4))

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
