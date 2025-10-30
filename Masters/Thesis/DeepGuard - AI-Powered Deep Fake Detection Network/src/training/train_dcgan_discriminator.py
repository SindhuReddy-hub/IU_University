import os, torch, random
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pandas as pd
from google.colab import drive

# === Mount & Setup ===
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/celeba_df"
SAVE_DIR = "/content/drive/MyDrive/celeba_models"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", DEVICE)

# === Data ===
IMG_SIZE = 128
BATCH_SIZE = 16
MAX_TRAIN, MAX_VAL, MAX_TEST = 4000, 1000, 1000

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def subset_dataset(ds, n):
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    return Subset(ds, idxs[:min(n, len(ds))])

train_ds = subset_dataset(datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_train), MAX_TRAIN)
val_ds   = subset_dataset(datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform_eval), MAX_VAL)
test_ds  = subset_dataset(datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform_eval), MAX_TEST)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ Data ready: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
