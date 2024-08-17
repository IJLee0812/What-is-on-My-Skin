import os
import random
import shutil
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

# path setting
data_dir = "./data/train_kfold" # Data will be processed privately to protect privacy
base_dir = "./results"

# data transforms setting
data_transforms = transforms.Compose([
    transforms.CenterCrop(224), # 리사이징, 이미지를 중앙에서 crop 한다.
    transforms.RandomHorizontalFlip(), # 좌우반전
    # 색상 변환 ColorJitter 제거
    transforms.RandomRotation(degrees = 20), # 20도 회전 추가
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet기반 모델의 최적 정규화 수치
])

# load dataset
dataset = datasets.ImageFolder(data_dir, data_transforms)

# save data informations
torch.save(dataset, os.path.join(base_dir, "dataset.pth"))
torch.save(data_transforms, os.path.join(base_dir, "data_transforms.pth"))
torch.save(dataset.classes, os.path.join(base_dir, "class_names.pth"))