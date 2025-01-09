#!/usr/bin/env python3
import torch
from torch import nn
from torch import optim
from torchvision import models
from PIL import Image
from file_util import *

def update_model_with_hidden_layer(architecture_name, num_classes, hidden_units=512):
    try:
        # Load the pre-trained model dynamically
        model = getattr(models, architecture_name)(weights='DEFAULT')
        
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
    
        # Replace the output layer based on the model's architecture
        if hasattr(model, 'fc'):  # For models like ResNet
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features, hidden_units),  # Hidden layer
                nn.ReLU(),  # Activation
                nn.Dropout(0.2),  # Dropout for regularization
                nn.Linear(hidden_units, num_classes)  # Output layer
            )
        elif hasattr(model, 'classifier'):  # For models like VGG, DenseNet, MobileNet
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[0].in_features
                model.classifier = nn.Sequential(
                    nn.Linear(in_features, hidden_units),  # Hidden layer
                    nn.ReLU(),  # Activation
                    nn.Dropout(0.2),  # Dropout for regularization
                    nn.Linear(hidden_units, num_classes)  # Output layer
                )
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Linear(in_features, hidden_units),  # Hidden layer
                    nn.ReLU(),  # Activation
                    nn.Dropout(0.2),  # Dropout for regularization
                    nn.Linear(hidden_units, num_classes)  # Output layer
                )
        else:
            raise ValueError(f"Unsupported architecture: {architecture_name}")
        
        return model

    except AttributeError:
        raise ValueError(f"Model {architecture_name} is not available in torchvision.")
        
def train_model(model, train_loaders, valid_loaders, epochs):
    criterion = nn.CrossEntropyLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # Add a Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    
    for epoch in range(epochs):
        
        # Training Phase
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loaders:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
        train_accuracy = correct / total
        train_loss = running_loss/len(train_loaders)
    
        # Validation Phase
        model.eval()
        running_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loaders:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
    
                # Accumulate validation loss and accuracy
                running_loss += batch_loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    
        val_accuracy = correct / total
        val_loss = running_loss/len(valid_loaders)
        
        # Print result for each epoch
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train Loss: {running_loss/len(train_loaders):.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss/len(valid_loaders):.4f}, Validation Accuracy: {val_accuracy:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Step the scheduler
        scheduler.step()
        
    return optimizer.state_dict(), train_loss, val_accuracy

# TODO: Implement the code to predict the class from an image file
def predict(image_path, checkpoint, idx_to_class, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Recreate the model from the pre-trained model and update the classifier
    architecture_name = checkpoint['architecture_name'] if 'architecture_name' in checkpoint else 'vgg13'
    num_classes = checkpoint['num_classes'] if 'num_classes' in checkpoint else 102
    hidden_units = checkpoint['hidden_units'] if 'hidden_units' in checkpoint else 512
    model = update_model_with_hidden_layer(architecture_name, num_classes, hidden_units)

    # Load parameters from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    
    # Set the model to evaluation mode
    model.eval()
    
    # Open the image
    pil_image = Image.open(image_path).convert('RGB')
    
    # Process the image
    input_tensor = process_image(pil_image)
    
    # Perform inference
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        
    # Convert logits to probabilities
    probabilities = torch.softmax(output, dim=1)
   
    # Get the top 5 predictions
    top_probs, top_indices = torch.topk(probabilities, k=topk, dim=1)
    
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    top_probs = top_probs.squeeze().tolist()
    
    print(top_probs)
    print(top_classes)

    return top_probs, top_classes
