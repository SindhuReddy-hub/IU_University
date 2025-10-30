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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Ensure F is imported here
from tqdm import tqdm
# Assuming DEVICE, train_loader, and SAVE_DIR are defined elsewhere

# ====================================================================
# 🧩 VAE Architecture (Fixed for 128x128 Input)
# ====================================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # 128x128 input -> 16x16 feature map.
        FLATTENED_SIZE = 256 * 16 * 16 # CORRECT SIZE: 65536

        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),    # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),   # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),  # 32 -> 16
            nn.Flatten()
        )

        # Linear layers use the correct FLATTENED_SIZE
        self.fc_mu = nn.Linear(FLATTENED_SIZE, latent_dim)
        self.fc_logvar = nn.Linear(FLATTENED_SIZE, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, FLATTENED_SIZE)

        self.dec = nn.Sequential(
            # Unflatten to the correct 16x16 size
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),  # 32 -> 64
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()     # 64 -> 128
        )

    def encode(self, x):
        h = self.enc_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.dec(self.fc_dec(z)), mu, logvar

# ====================================================================
# 💡 VAE Loss Function (CRITICAL FIX: Correct Scaling)
# ====================================================================
def vae_loss(recon_x, x, mu, logvar):
    # The sum is divided by x.size(0) to get a per-sample average MSE.
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL Divergence is also averaged per sample.
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + kl_loss

# ====================================================================
# 🚀 Training Loop
# ====================================================================
model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=5e-4) # You could try 5e-4 if 1e-4 is too slow

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0

    for imgs, _ in tqdm(train_loader, desc=f"VAE Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = model(imgs)
        loss = vae_loss(recon, imgs, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    print(f"Train VAE loss: {total_loss/num_batches:.4f}")

torch.save(model.state_dict(), f"{SAVE_DIR}/vae_final.pt")
print("✅ Saved VAE →", f"{SAVE_DIR}/vae_final.pt")