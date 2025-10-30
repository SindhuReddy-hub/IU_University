# ============================================================
# 🔍 DeepGuard Unified Evaluation + Explainability Visualization
# ============================================================
import os, torch, numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_METRICS_PATH = "/content/drive/MyDrive/DeepGuard/deepguard_final_metrics.csv"
os.makedirs(os.path.dirname(SAVE_METRICS_PATH), exist_ok=True)

# =========================
# Grad-CAM helper (ResNet/Xception)
# =========================
def gradcam_visualize(model, layer, dataloader, device, title):
    try:
        import cv2
    except:
        print("⚠️ OpenCV not installed, skipping Grad-CAM.")
        return

    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:4].to(device)
    imgs.requires_grad = True

    features, gradients = [], []
    def forward_hook(module, inp, out): features.append(out)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_backward_hook(backward_hook)

    outputs = model(imgs)
    preds = torch.argmax(outputs, 1)
    loss = outputs[range(len(preds)), preds].sum()
    loss.backward()

    fmap = features[0].detach()
    grad = gradients[0].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(128,128), mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    imgs_np = (imgs.cpu().permute(0,2,3,1).numpy() + 1)/2
    cam_np = cam.cpu().permute(0,2,3,1).numpy()

    plt.figure(figsize=(10,4))
    for i in range(4):
        plt.subplot(2,4,i+1); plt.imshow(imgs_np[i]); plt.axis("off")
        plt.subplot(2,4,i+5); plt.imshow(imgs_np[i]); plt.imshow(cam_np[i,:,:,0], cmap='jet', alpha=0.5); plt.axis("off")
    plt.suptitle(f"Grad-CAM: {title}")
    plt.show()
    handle_f.remove(); handle_b.remove()

# =========================
# Reconstruction visualization (AE/VAE/DCGAN)
# =========================
def visualize_reconstruction(model, dataloader, device, title, is_gan=False):
    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:8].to(device)
    with torch.no_grad():
        if is_gan:
            noise = torch.randn(8, 100, 1, 1, device=device)
            recon = model(noise)
        else:
            recon = model(imgs)
    imgs_np = (imgs.cpu().permute(0,2,3,1).numpy() + 1)/2
    recon_np = (recon.cpu().permute(0,2,3,1).numpy() + 1)/2
    plt.figure(figsize=(12,3))
    for i in range(8):
        plt.subplot(2,8,i+1); plt.imshow(np.clip(imgs_np[i],0,1)); plt.axis("off")
        plt.subplot(2,8,i+9); plt.imshow(np.clip(recon_np[i],0,1)); plt.axis("off")
    plt.suptitle(f"{title}: Original (top) vs Reconstructed/Generated (bottom)")
    plt.show()

# =========================
# Evaluation metrics functions
# =========================
def evaluate_classification_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            all_preds += preds.cpu().numpy().tolist()
            all_labels += labels.cpu().numpy().tolist()
    return (
        accuracy_score(all_labels, all_preds),
        precision_score(all_labels, all_preds, average='macro', zero_division=0),
        recall_score(all_labels, all_preds, average='macro', zero_division=0),
        f1_score(all_labels, all_preds, average='macro', zero_division=0)
    )

def evaluate_reconstruction_model(model, dataloader, device, is_gan=False):
    model.eval()
    mse_list, psnr_list, ssim_list = [], [], []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            if is_gan:
                noise = torch.randn(imgs.size(0), 100, 1, 1, device=device)
                recon = model(noise)
            else:
                recon = model(imgs)
            imgs_np = imgs.cpu().numpy().transpose(0,2,3,1)
            recon_np = recon.cpu().numpy().transpose(0,2,3,1)
            for a,b in zip(imgs_np, recon_np):
                a, b = (a+1)/2, (b+1)/2
                mse = np.mean((a-b)**2)
                ps = psnr(a,b,data_range=1)
                ss = ssim(a,b,channel_axis=2,data_range=1)
                mse_list.append(mse); psnr_list.append(ps); ssim_list.append(ss)
    return np.mean(mse_list), np.mean(psnr_list), np.mean(ssim_list)

# =========================
# Load your trained models
# =========================
resnet_model = torch.load("/content/drive/MyDrive/models/resnet50.pt", map_location=DEVICE)
xception_model = torch.load("/content/drive/MyDrive/models/xception.pt", map_location=DEVICE)
ae_model = torch.load("/content/drive/MyDrive/models/autoencoder.pt", map_location=DEVICE)
vae_model = torch.load("/content/drive/MyDrive/models/vae.pt", map_location=DEVICE)

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
    def forward(self, x): return self.main(x)

dcgan_ckpt = torch.load("/content/drive/MyDrive/models/dcgan_quick.pt", map_location=DEVICE)
dcgan_G = DCGAN_Generator().to(DEVICE)
dcgan_G.load_state_dict(dcgan_ckpt['G'])

# =========================
# Evaluate all models
# =========================
results = []

# Classification
for name, model in [("ResNet50", resnet_model), ("Xception", xception_model)]:
    acc, prec, rec, f1 = evaluate_classification_model(model, test_loader, DEVICE)
    results.append([name, acc, prec, rec, f1, np.nan, np.nan, np.nan])

# AE, VAE, DCGAN
for name, model, is_gan in [("AutoEncoder", ae_model, False), ("VAE", vae_model, False), ("DCGAN", dcgan_G, True)]:
    mse, ps, ss = evaluate_reconstruction_model(model, test_loader, DEVICE, is_gan)
    results.append([name, np.nan, np.nan, np.nan, np.nan, mse, ps, ss])

# =========================
# 📊 Results Table
# =========================
df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1","MSE","PSNR","SSIM"])
display(df)
df.to_csv(SAVE_METRICS_PATH, index=False)
print(f"\n✅ Saved metrics to {SAVE_METRICS_PATH}")

# =========================
# 🔥 Visualizations
# =========================
gradcam_visualize(resnet_model, resnet_model.layer4[-1], test_loader, DEVICE, "ResNet50")
gradcam_visualize(xception_model, xception_model.block12[-1], test_loader, DEVICE, "Xception")

visualize_reconstruction(ae_model, test_loader, DEVICE, "AutoEncoder")
visualize_reconstruction(vae_model, test_loader, DEVICE, "VAE")
visualize_reconstruction(dcgan_G, test_loader, DEVICE, "DCGAN", is_gan=True)
