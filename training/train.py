import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
from preprocess import get_data_loaders
import copy

def train_model(data_dir='../dataset', num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Trains a pre-trained ResNet50 model for water pollution classification.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
        print(f"Classes found: {class_names}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your dataset is located in the 'dataset' folder, with classes like 'clean', 'oil', etc.")
        return

    num_classes = len(class_names)

    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze convolutional layers (optional transfer learning step)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save model
    os.makedirs('../model', exist_ok=True)
    save_path = '../model/water_pollution_model.pth'
    torch.save(model.state_dict(), save_path)
    
    # Save classes list as well for backend to use
    with open('../model/classes.txt', 'w') as f:
        for c in class_names:
            f.write(f"{c}\n")
            
    print(f"Model saved to {save_path}")
    return model

if __name__ == '__main__':
    train_model(data_dir='../dataset', num_epochs=6)
