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
# =============================================================================
# 🧠 Xception Training (3 epochs) - MODIFIED FOR TIMM & CLASS IMBALANCE
# =============================================================================

# --- FIX 1: Use timm for reliable Xception loading ---
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- FIX 2: Define Class Weights for Imbalanced Data ---
# CRITICAL: Replace these placeholders with the actual calculated weights.
# Example: If 80% of samples are Class 0 (Negative) and 20% are Class 1 (Positive),
#          the weights should be higher for the minority class (Class 1).
#          weights = [1.0/0.8, 1.0/0.2] or similar inverse frequency.

# Placeholder example (REPLACE WITH YOUR CALCULATED WEIGHTS)
# e.g., if you found 85% negative (Class 0) and 15% positive (Class 1)
# You might use: weights = [1.0, 85/15]
class_weights = torch.tensor([1.0, 5.66], dtype=torch.float32).to(DEVICE) # <-- REPLACE THIS

# Load model via timm
model = timm.create_model("xception", pretrained=True, num_classes=2)

# --- FIX 3: FREEZE THE BACKBONE ---
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classifier head parameters created by timm
for param in model.get_classifier().parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# --- FIX 4: Use Weighted CrossEntropyLoss ---
criterion = nn.CrossEntropyLoss(weight=class_weights)

# --- FIX 5: Optimize ONLY the trainable parameters (the classification head) ---
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3) # Increased LR to 1e-3 for faster head training

EPOCHS = 20 # You should increase this to at least 15-20 after confirming the code works

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    num_samples = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        num_samples += imgs.size(0)

    print(f"Train acc: {correct/num_samples:.4f}, Weighted Loss: {total_loss/num_samples:.4f}")

torch.save(model.state_dict(), f"{SAVE_DIR}/xception_quick.pt")
print("✅ Saved Xception →", f"{SAVE_DIR}/xception_quick.pt")
