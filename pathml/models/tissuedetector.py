import torch
from torchvision import transforms, models



def tissueDetector():

    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # For now required PyTorch housekeeping
    model_ft = models.densenet121(pretrained=False)
    model_ft.classifier = torch.nn.Linear(1024, 3)
    # It might struggle finding this dict here. Use full path to pathml
    model_ft.load_state_dict(torch.load('../pathml/models/deep-tissue-detector_densenet_state-dict.pt',map_location=torch.device('cpu')))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)
    model_ft.to(device).eval()

    return device, model_ft, data_transforms
