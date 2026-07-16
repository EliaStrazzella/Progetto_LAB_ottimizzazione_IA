import os
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
from augmentation import get_data_transforms

def get_data_loaders(data_dir, img_size=(224,224), batch_size=32):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

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

def load_plantvillage_data(data_dir, img_size=(224, 224), batch_size=64, val_split=0.2, seed=42):
    train_transform, val_transform = get_data_transforms(img_size[0])

    dataset = datasets.ImageFolder(data_dir)
    num_data = len(dataset)
    indices = list(range(num_data))
    
    #Calcolo Immagini da usare
    split = int(val_split * num_data)

    torch.manual_seed(seed)
    train_idx, val_idx = indices[split:], indices[:split]

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)

    return train_loader, val_loader, dataset.classes