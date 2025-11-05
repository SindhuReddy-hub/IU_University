# DeepGuard: Multimodal Deepfake Detection with Explainability & Vision‑Language Reasoning
DeepGuard is a hybrid supervised + unsupervised DeepFake detection framework designed to provide **high classification accuracy** along with **explainable and forensic-grade evidence outputs**. The system integrates CNN-based classification (ResNet50, Xception) with generative anomaly detection modules (AE, VAE, DCGAN) and semantic reasoning via vision-language models.

## ✅ Key Capabilities

- **Supervised Detection**: ResNet50 and Xception classify real vs fake faces.
- **Unsupervised Anomaly Detection**: Autoencoder (AE), VAE, and DCGAN identify reconstruction/latent inconsistencies.
- **Hybrid Explainability**:
  - Pixel-level anomaly maps (AE/VAE)
  - Latent deviation analysis (DCGAN)
  - Grad-CAM & saliency visualizations (CNNs)
  - Semantic reasoning (BLIP + CLIP text-based explanations)
  - 
DeepGuard is a multimodal **Deepfake Detection** framework combining:

| Component | Role |
|---|---|
| ResNet50, Xception | Supervised deepfake classification |
| Autoencoder | Pixel‑level anomaly detection |
| VAE | Latent anomaly + KL divergence |
| DCGAN | Fake‑pattern residual learning |
| Grad‑CAM + Guided Backprop | Visual explainability |
| CLIP / BLIP | Explainability through text |
| Fusion Classifier | Final robust prediction |

---

## 🚨 Motivation
Deepfakes threaten **media trust, elections, cybersecurity, and identity integrity**.  
DeepGuard ensures **high‑accuracy detection + transparent explainability**.

---

## ✅ Key Features
- Hybrid **supervised + self‑supervised** architecture  
- **Pixel + feature + latent + language‑layer** explainability  
- Robust **model fusion** strategy  
- Cross‑model interpretability (CNN + AE/VAE + GAN + VLMs)

---

## 🧠 Architecture

```
Input → Face Alignment → CNN Classification  
     → AE/VAE Reconstruction → DCGAN Residual  
     → Grad‑CAM + GuidedBP → BLIP/CLIP Caption  
     → Final Fusion Decision
```

---

## 🧪 Dataset
- CelebA / Celeb‑DF dataset  
- Face alignment + 224x224 normalization  
- Train / Validation / Test: **80 / 10 / 10**

---

## 📊 Performance (example ‑ replace placeholders)
| Model | Accuracy | F1 | AUC |
|---|---|---|---|
ResNet50 | xx | xx | xx |
Xception | xx | xx | xx |
AE | xx | xx | xx |
VAE | xx | xx | xx |
Fusion (Ours) | **Best** | **Highest** | **Strongest** |

---

## 🧾 Example Output

```
Final Prediction: FAKE ✅  (0.89)
Pixel anomaly high near face edges
Latent KL: 412 (abnormal)
GAN residual artifacts detected
BLIP: mild face distortion text evidence
Conclusion: Highly likely synthetic
```

---

## 🚀 Future Work
- DFDC, FaceForensics++, Stable Diffusion, Sora video deepfakes  
- Deploy via **Streamlit / FastAPI**  
- Add **LLM forensic judge module**  

---
## 📁 Repository Structure
```
DeepGuard/
 ├── src/
 │   ├── training/              # Model training scripts
 │   │   ├── train_resnet50.py
 │   │   ├── train_xception.py
 │   │   ├── train_autoencoder.py
 │   │   ├── train_vae.py
 │   │   └── train_dcgan_discriminator.py
 │   ├── evaluation/            # Evaluation + visualization scripts
 │   │   ├── evaluate_models.py
 │   │   └── visualization.py
 │   ├── utils/                 # Preprocessing + metric utilities
 │   │   ├── preprocess.py
 │   │   ├── metrics.py
 │   │   └── visualization.py
 │   └── docs/                  # Notes and development logs
 │
 ├── output/                    # Model outputs (images, maps, reports)
 │   ├── AE_Results/
 │   ├── VAE_Results/
 │   ├── DCGAN_Results/
 │   ├── ResNET50_Results/
 │   └── Xception_Results/
 │
 ├── requirements.txt           # Dependencies
 └── README.md                  # (This file)
```

---

## 🧠 Trained Model Weights (Download)

Because model checkpoints exceed GitHub file size limits, all trained weights are stored on **Google Drive**:

🔗 **Download Weights:**  
https://drive.google.com/drive/folders/1_CyzGGwZRT_fqQtjleoITHm3UVgn0maj

After downloading, place them here:

```
DeepGuard/
 └── models/
     ├── resnet50_quick.pt
     ├── xception_quick.pt
     ├── autoencoder_retrained_compat.pt
     ├── vae_celeba_latent_200_epochs.pth
     ├── dcgan_discriminator.pth
     ├── mlp_classifier_v4.pth
     └── mlp_classifier_vae_v3_balanced.pth
```

---

## 🛠 Setup & Installation

```
git clone https://github.com/SindhuReddy-hub/IU_University.git
cd Masters/Thesis/DeepGuard - AI-Powered Deep Fake Detection Network
pip install -r requirements.txt
```

---

## ▶️ Running Evaluation

```
cd src/evaluation
python evaluate_models.py
```

---

## 📊 Explainability Outputs

This project generates:

| Model | Explainability Output |
|------|----------------------|
| ResNet50 / Xception | Grad-CAM + Saliency Maps |
| Autoencoder (AE) | Pixel reconstruction error heatmap |
| VAE | Latent distance abnormality score |
| DCGAN | Discriminator anomaly boundary visualization |
| Ensemble | Weighted confidence + final decision reasoning |

---

## 🔍 Citation (for Thesis / Research Use)

```
Reddy, S. (2025). DeepGuard – AI-Powered DeepFake Detection Network [Source code].
GitHub. https://github.com/SindhuReddy-hub/IU_University/
```

---

## 🙌 Acknowledgements
- IU University  
- Guide: **Dr. Aditya**  
- Open‑source ML community  

---

## 📜 License
MIT License

---

### ⭐ If this repo helped, drop a star!
