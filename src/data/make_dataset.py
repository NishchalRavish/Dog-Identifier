import os 
import torch
from torchvision import datasets,transforms

def load_data(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    data = datasets.ImageFolder(data_dir, transform=transform)
    
    return data

def prepare_data(raw_data_dir,processed_data_dir):
    train_data = load_data(os.path.join(raw_data_dir, 'train'))
    val_data = load_data(os.path.join(processed_data_dir, 'val'))
    
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
        
    torch.save(train_data, os.path.join(processed_data_dir, 'train_data.pth'))
    torch.save(val_data, os.path.join(processed_data_dir, 'val_data.pth'))
    
if __name__ == "__main__":
    prepare_data("data/raw", "data/processed")