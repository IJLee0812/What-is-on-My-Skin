# Running the model on the test data on 'Google Colaboratory'

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image
from torchvision import models
from torchvision.models import DenseNet121_Weights

# Base directory for saving the model and data
base_dir = "/content/drive/My Drive/Colab Notebooks/What-is-on-My-Skin/results"
test_dir = "/content/drive/My Drive/Colab Notebooks/What-is-on-My-Skin/data/test"

data_transforms = torch.load(os.path.join(base_dir, "data_transforms.pth"))
class_names = torch.load(os.path.join(base_dir, "class_names.pth"))

def initialize_model(num_classes):
    # for Transfer Learning
    model = models.densenet121(weights = DenseNet121_Weights.DEFAULT)

    # freezing the parameters
    for param in model.parameters():
        param.requires_grad = False

    # modify last FC Layer(Fine-Tuning)
    num_ftrs = model.classifier.in_features # fc -> classifier
    model.classifier = nn.Linear(num_ftrs, num_classes) # fc -> classifier
    return model

def load_test_data(test_dir, data_transforms, batch_size=32):
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset.classes

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(len(class_names))
    model_path = os.path.join(base_dir, "best_model_fold_1.pth")
    
    if not os.path.exists(model_path):
        print("Model file does not exist.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    test_loader, _ = load_test_data(test_dir, data_transforms)
    
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"\n\n\nTest Accuracy: {test_accuracy * 100:.2f}%\n\n\n")

if __name__ == "__main__":
    main()
