import torch
import torch.nn as nn
from torchvision import models
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# === PATHS ===
MODEL_DIR = "/content/drive/MyDrive/celeba_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === 1️⃣ ResNet50 ===
resnet_model = models.resnet50(weights=None, num_classes=2)
resnet_path = os.path.join(MODEL_DIR, "resnet50_best.pth")
resnet_model.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
resnet_model.to(DEVICE).eval()
print("✅ Loaded ResNet50")

# === 2️⃣ Xception ===
# (if trained in PyTorch)
xception_path = os.path.join(MODEL_DIR, "xception_best.pth")
try:
    xception_model = torch.load(xception_path, map_location=DEVICE)
    xception_model.eval()
    print("✅ Loaded Xception")
except Exception as e:
    print("⚠️ Could not load Xception:", e)

# === 3️⃣ AutoEncoder ===
class AutoEncoder(nn.Module):
    def __init__(self, img_channels=3, feature_dim=64, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(feature_dim, feature_dim*2, 4, 2, 1), nn.BatchNorm2d(feature_dim*2), nn.ReLU(True),
            nn.Conv2d(feature_dim*2, feature_dim*4, 4, 2, 1), nn.BatchNorm2d(feature_dim*4), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*4, feature_dim*2, 4, 2, 1), nn.BatchNorm2d(feature_dim*2), nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim*2, feature_dim, 4, 2, 1), nn.BatchNorm2d(feature_dim), nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim, img_channels, 4, 2, 1), nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

ae_model = AutoEncoder(img_channels=3, feature_dim=64, latent_dim=256).to(DEVICE)
ae_path = os.path.join(MODEL_DIR, "autoencoder_best.pth")
ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
ae_model.eval()
print("✅ Loaded AutoEncoder")

# === 4️⃣ Variational AutoEncoder (VAE) ===
class VAE(nn.Module):
    def __init__(self, img_channels=3, feature_dim=64, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(feature_dim, feature_dim*2, 4, 2, 1), nn.BatchNorm2d(feature_dim*2), nn.ReLU(True),
            nn.Conv2d(feature_dim*2, feature_dim*4, 4, 2, 1), nn.BatchNorm2d(feature_dim*4), nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(feature_dim*4*16*16, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim*4*16*16, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, feature_dim*4*16*16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*4, feature_dim*2, 4, 2, 1), nn.BatchNorm2d(feature_dim*2), nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim*2, feature_dim, 4, 2, 1), nn.BatchNorm2d(feature_dim), nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim, img_channels, 4, 2, 1), nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z).view(z.size(0), 256, 16, 16)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

vae_model = VAE(img_channels=3, feature_dim=64, latent_dim=128).to(DEVICE)
vae_path = os.path.join(MODEL_DIR, "vae_best.pth")
vae_model.load_state_dict(torch.load(vae_path, map_location=DEVICE))
vae_model.eval()
print("✅ Loaded VAE")
models_list = {
    "ResNet50": resnet_model,
    "Xception": xception_model if 'xception_model' in locals() else None,
    "AutoEncoder": ae_model,
    "VAE": vae_model
}

for name, model in models_list.items():
    if model is not None:
        print(f"✅ {name} loaded and ready on {DEVICE}")
    else:
        print(f"❌ {name} missing")
