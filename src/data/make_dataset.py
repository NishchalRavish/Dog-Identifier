import os 
from torchvision import datasets,transforms

def load_date(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    data = datasets.ImageFolder(data_dir, transform=transform)
    
    return data