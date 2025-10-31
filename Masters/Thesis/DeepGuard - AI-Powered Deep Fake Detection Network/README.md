# DeepGuard: Multimodal Deepfake Detection with Explainability & Vision‑Language Reasoning

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

## 🙌 Acknowledgements
- IU University  
- Guide: **Dr. Aditya**  
- Open‑source ML community  

---

## 📜 License
MIT License

---

### ⭐ If this repo helped, drop a star!
