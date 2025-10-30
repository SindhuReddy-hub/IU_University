import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import pandas as pd
import os

# ==============================================================
# CONFIG
# ==============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_METRICS_PATH = "/content/drive/MyDrive/deepguard_metrics.csv"

# ==============================================================
# CLASSIFICATION MODEL EVALUATION (ResNet50 & Xception)
# ==============================================================
def evaluate_classification_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, prec, rec, f1

# ==============================================================
# RECONSTRUCTION MODEL EVALUATION (AE, VAE, DCGAN)
# ==============================================================
def evaluate_reconstruction_model(model, dataloader, device, is_gan=False):
    model.eval()
    mse_scores, psnr_scores, ssim_scores = [], [], []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            if is_gan:
                # For DCGAN, generate fake images from random noise
                noise = torch.randn(imgs.size(0), 100, 1, 1, device=device)
                recon = model(noise)
            else:
                recon = model(imgs)
            imgs_np = imgs.cpu().numpy().transpose(0, 2, 3, 1)
            recon_np = recon.cpu().numpy().transpose(0, 2, 3, 1)
            for orig, rec in zip(imgs_np, recon_np):
                orig = np.clip((orig + 1) / 2, 0, 1)
                rec = np.clip((rec + 1) / 2, 0, 1)
                m = np.mean((orig - rec) ** 2)
                p = psnr(orig, rec, data_range=1)
                s = ssim(orig, rec, channel_axis=2, data_range=1)
                mse_scores.append(m)
                psnr_scores.append(p)
                ssim_scores.append(s)
    return np.mean(mse_scores), np.mean(psnr_scores), np.mean(ssim_scores)

# ==============================================================
# LOAD MODELS (adjust paths as per your Drive)
# ==============================================================
resnet_model = torch.load("/content/drive/MyDrive/models/resnet50.pt", map_location=DEVICE)
xception_model = torch.load("/content/drive/MyDrive/models/xception.pt", map_location=DEVICE)
ae_model = torch.load("/content/drive/MyDrive/models/autoencoder.pt", map_location=DEVICE)
vae_model = torch.load("/content/drive/MyDrive/models/vae.pt", map_location=DEVICE)

# For DCGAN, load generator only
dcgan_ckpt = torch.load("/content/drive/MyDrive/models/dcgan_quick.pt", map_location=DEVICE)
from torch import nn

class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

dcgan_G = DCGAN_Generator().to(DEVICE)
dcgan_G.load_state_dict(dcgan_ckpt['G'])

# ==============================================================
# EVALUATION (assuming you have test_loader ready)
# ==============================================================
results = []

# 1️⃣ ResNet50
acc, prec, rec, f1 = evaluate_classification_model(resnet_model, test_loader, DEVICE)
results.append(["ResNet50", acc, prec, rec, f1, np.nan, np.nan, np.nan])

# 2️⃣ Xception
acc, prec, rec, f1 = evaluate_classification_model(xception_model, test_loader, DEVICE)
results.append(["Xception", acc, prec, rec, f1, np.nan, np.nan, np.nan])

# 3️⃣ AutoEncoder
mse, p, s = evaluate_reconstruction_model(ae_model, test_loader, DEVICE)
results.append(["AutoEncoder", np.nan, np.nan, np.nan, np.nan, mse, p, s])

# 4️⃣ VAE
mse, p, s = evaluate_reconstruction_model(vae_model, test_loader, DEVICE)
results.append(["VAE", np.nan, np.nan, np.nan, np.nan, mse, p, s])

# 5️⃣ DCGAN
mse, p, s = evaluate_reconstruction_model(dcgan_G, test_loader, DEVICE, is_gan=True)
results.append(["DCGAN", np.nan, np.nan, np.nan, np.nan, mse, p, s])

# ==============================================================
# RESULTS TABLE
# ==============================================================
df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "MSE", "PSNR", "SSIM"])
print("✅ Final Metrics Table:")
print(df_results)

# Save to Drive
os.makedirs(os.path.dirname(SAVE_METRICS_PATH), exist_ok=True)
df_results.to_csv(SAVE_METRICS_PATH, index=False)
print(f"\n📁 Metrics saved to: {SAVE_METRICS_PATH}")
