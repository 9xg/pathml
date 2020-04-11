import sys
sys.path.append('/home/gehrun01/Desktop/pathml')
from pathml import slide
from pathml.WholeSlideImageDataset import WholeSlideImageDataset
from pathml.analysis import Analysis
from torchvision import datasets, models, transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image

# For now required PyTorch housekeeping
model_ft = models.resnet18(pretrained=False)
model_ft.fc = torch.nn.Linear(512, 3)
model_ft = models.densenet121()
model_ft.classifier = torch.nn.Linear(1024, 3)
# It might struggle finding this dict here. Use full path to pathml
model_ft.load_state_dict(torch.load('../pathml/models/deep-tissue-detector_densenet_state-dict.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft.to(device).eval()


pathmlSlide = slide.Slide('OC-RS-024_OGD.svs',level=1).setTileProperties(tileSize=448,tileOverlap=0.5)

pathmlSlide.applyModel(device, model_ft, batch_size=20, predictionKey='foreground')
print(pathmlSlide.tileDictionary)

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
