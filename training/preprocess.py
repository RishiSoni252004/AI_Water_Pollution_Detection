import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32, train_split=0.8):
    """
    Reads images from data_dir, applies basic preprocessing/augmentation,
    and returns DataLoaders for train and validation.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found. Please add images.")

    # Data augmentation and normalization for training
    # Just normalization for validation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = ImageFolder(root=data_dir)
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Please add images to the dataset directory.")

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply specific transforms
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = full_dataset.classes
    return train_loader, val_loader, class_names
