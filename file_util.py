#!/usr/bin/env python3
from pathlib import Path
import torch
from datetime import datetime
from torchvision import datasets, transforms, models

def load_train_data(train_dir):
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2)),  # Random scale between 80% and 120%
                                       transforms.RandomRotation(degrees=(-90, 90)),                        # Rotate randomly between -90° and 90°
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)), 
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)

    return train_datasets, train_loaders

def load_valid_data(valid_dir):
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    valid_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)

    return valid_datasets, valid_loaders

def process_image(pil_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a tensor image with shape (1, 3, 224, 224)
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    shorter_side = 256
    
    # Get original dimensions (width, height)
    width, height = pil_image.size
    
    # Calculate new dimensions while preserving aspect ratio
    if width < height:
        # Resize the shorter side (width) to 256, scale the other side proportionally
        new_width = shorter_side
        new_height = int((shorter_side * height) / width)
    else:
        # Resize the shorter side (height) to 256, scale the other side proportionally
        new_height = shorter_side
        new_width = int((shorter_side * width) / height)
        
    # Define the transform    
    transform = transforms.Compose([transforms.Resize((new_height, new_width)),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    tensor_image = transform(pil_image)
    return tensor_image.unsqueeze(0)  # Shape: (1, 3, 224, 224)

def save_checkpoint(checkpoint_filepath, model_state_dict, 
                    architecture_name, num_classes, hidden_units,
                    optimizer_state_dict, class_to_idx, train_loss, valid_accuracy, epochs):
    print(f"Save to checkpoint file {checkpoint_filepath}")
    checkpoint = {
        'model_state_dict': model_state_dict,
        'architecture_name': architecture_name,
        'hidden_units': hidden_units,
        'num_classes': num_classes,
        'optimizer_state_dict': optimizer_state_dict,
        'class_to_idx': class_to_idx,
        'train_loss': train_loss,
        'valid_accuracy': valid_accuracy,
        'epoch': epochs
    }
    torch.save(checkpoint, checkpoint_filepath)
    return checkpoint

def load_checkpoint(filepath):
    return torch.load(filepath, weights_only=False)

def get_cat_to_name(file_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def generate_filename():
    # Get the current date and time
    current_time = datetime.now()
    # Format it as yyyyMMddhhmmss
    filename = current_time.strftime("%Y%m%d%H%M%S")
    return filename

def check_dir_path(dir_path):
    path = Path(dir_path)
    if not path.is_dir() or not path.exists():
        return False
    return str(path)