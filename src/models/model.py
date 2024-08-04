import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes):
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                

            running_loss = 0.0
            running_corrects = 0
            
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase =='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs,labels)
                    
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size[0]
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss  = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
        if (epoch+1) % 5 ==0:
            torch.save(model.state_dict(), f'models/checkpoints/model_epoch_{epoch+1}.pth')
            
    return model
                
    