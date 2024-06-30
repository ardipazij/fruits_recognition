import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import rembg
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from PIL import Image

epochs = 5
batch_size = 32
configuration_dict = {
    'number_of_epochs': epochs,
    'batch_size': batch_size,
    'base_lr': 1e-4,
    'weight_decay': 1e-4,
    'rescale_size': 64
}

data_dir = 'path/to/model'

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(os.path.join(data_dir, 'Training'), transform=train_transforms)
valid_data = datasets.ImageFolder(os.path.join(data_dir, 'Test'), transform=valid_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)


class CustomShuffleNet(nn.Module):
    def __init__(self, num_classes=131):
        super(CustomShuffleNet, self).__init__()
        self.model = models.shufflenet_v2_x0_5(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ModelInference:
    def __init__(self, model_path: str, classes: list, device='cpu'):
        self.device = device
        self.model = CustomShuffleNet(num_classes=131)
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.classes = classes

    def inference(self, img_path):
        img = Image.open(img_path).convert('RGB')
        tensor = self.transforms(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor)
            _, predicted = torch.max(preds, 1)
        return self.classes[predicted.item()]
