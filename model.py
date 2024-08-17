import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

def initialize_model(num_classes):
    # for Transfer Learning
    model = models.densenet121(weights = DenseNet121_Weights.DEFAULT)

    # freezing(모델의 기존 Hidden Layer의 Weight를 고정하여 그대로 사용)
    for param in model.parameters():
        param.requires_grad = False

    # modify last FC Layer(Fine-Tuning)
    num_ftrs = model.classifier.in_features # fc -> classifier
    model.classifier = nn.Linear(num_ftrs, num_classes) # fc -> classifier
    return model