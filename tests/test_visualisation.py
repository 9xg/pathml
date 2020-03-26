import sys
sys.path.append('/Users/gehrun01/Desktop/pathml')
from pathml.analysis import Analysis
import matplotlib.pyplot as plt

demoPath = '/Users/gehrun01/Downloads/tumor_001.tif.pml'

testAnalysis = Analysis(demoPath)
mapClass1=testAnalysis.generateInferenceMap(predictionSelector=1)

plt.imshow(mapClass1)
plt.show()
