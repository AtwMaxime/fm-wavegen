# fm-wavegen

A two-stage image generation pipeline using Flow Matching and wavelet decomposition.

---

## 🔍 What it does

- **Model 1 (LL Generator)**: Generates low-resolution images (LL) from Gaussian noise using Flow Matching.
- **Model 2 (HF Generator)**: Predicts high-frequency wavelet bands (HL, LH, HH), conditioned on the LL image and target resolution.
- **Final image**: Reconstructed via inverse Haar wavelet transform. The process is repeated to upscale progressively.

---

## 🗂️ Project Structure

- `data/` – raw and processed wavelet data
- `models/` – LL and HF generators
- `scripts/` – training and evaluation code
- `utils/` – wavelet transforms, FID, etc.
- `outputs/` – generated samples and logs

---

## 🚀 Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data
python scripts/preprocess_ll.py
python scripts/preprocess_hf.py

# Train models
python scripts/train_ll.py
python scripts/train_hf.py

# Evaluate
python scripts/eval_ll.py
python scripts/eval_hf.py
```
## 📝 Notes 

- Wavelets: Haar transform
- Trained progressively at multiple resolutions (16 → 256)
