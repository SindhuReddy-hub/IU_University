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
# =============================
# 🧩 AutoEncoder Training (5 epochs)
# =============================
# =============================
# 🧩 AutoEncoder (FIXED FOR 128x128 INPUT)
# =============================
class AE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # 128 -> 64 -> 32 -> 16. Final flattened size is 256 * 16 * 16 = 65,536
        FLATTENED_SIZE = 256 * 16 * 16

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),    # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),   # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),  # 32 -> 16
            nn.Flatten(),
            # --- FIX: Use the correct size for 128x128 input ---
            nn.Linear(FLATTENED_SIZE, latent_dim)
        )
        self.decoder = nn.Sequential(
            # --- FIX: Use the correct size for decoder input/unflatten ---
            nn.Linear(latent_dim, FLATTENED_SIZE),
            nn.ReLU(),
            nn.Unflatten(1, (256, 16, 16)), # Unflatten to 16x16

            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),  # 32 -> 64
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()     # 64 -> 128 (Final Output)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = AE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        recon = model(imgs)
        loss = criterion(recon, imgs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Train MSE: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), f"{SAVE_DIR}/autoencoder_quick.pt")
print("✅ Saved AE →", f"{SAVE_DIR}/autoencoder_quick.pt")
