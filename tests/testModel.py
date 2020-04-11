import sys
sys.path.append('/Users/gehrun01/Desktop/pathml')
from pathml import slide
from pathml.WholeSlideImageDataset import WholeSlideImageDataset
from pathml.analysis import Analysis
from torchvision import datasets, models, transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image


data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


model_ft = models.resnet18(pretrained=False)
model_ft.fc = torch.nn.Linear(512, 3)
#model_ft = models.densenet121()
#model_ft.classifier = torch.nn.Linear(1024, 3)
model_ft.load_state_dict(torch.load('/Users/gehrun01/Desktop/pathml/pathml/models/deep-tissue-detector_resnet_state-dict.pt',map_location=torch.device('cpu')),strict=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft.to(device).eval()

pathmlSlide = slide.Slide('/Users/gehrun01/Desktop/BEST2_CAM_0654_HE_1.svs',level=1)
pathmlSlide.setTileProperties(tileSize=448)
pathSlideDataset = WholeSlideImageDataset(pathmlSlide, transform=data_transforms)

pathmlSlide.applyModel(device, model_ft,pathSlideDataset,batch_size=30,prediction_key='foreground')
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
