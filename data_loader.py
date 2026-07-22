import os
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
from augmentation import get_data_transforms

def get_data_loaders(data_dir, img_size=(224,224), batch_size=32):
    train_transform, val_transform = get_data_transforms(img_size[0])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes

    return train_loader, val_loader,test_loader, class_names
