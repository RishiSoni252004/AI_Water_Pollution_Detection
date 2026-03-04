import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(data_dir='../dataset', model_path='../model/water_pollution_model.pth'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Needs to match val_transforms
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found.")
        return

    full_dataset = ImageFolder(root=data_dir, transform=val_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # Use a small batch size for evaluation
    test_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load model architecture
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load weights
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' not found.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate reports
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    os.makedirs('../model', exist_ok=True)
    plt.savefig('../model/confusion_matrix.png')
    print("Confusion matrix plot saved to ../model/confusion_matrix.png")

if __name__ == '__main__':
    evaluate_model()
