# Emotion Recognition using EEG and ECG Signals

This project extracts features from EEG and ECG signals using the DREAMER dataset and performs emotion classification using KMeans clustering and a PyTorch-based neural network. A Gradio web UI is provided for real-time emotion prediction.

## 📁 Files Overview

- `EEG.py` – Extracts PSD-based EEG features
- `ECG.py` – Extracts heart-rate-related ECG features using NeuroKit2
- `main.py` – Combines features, clusters them using KMeans, trains a PyTorch model, and launches a Gradio UI
- `Extracted_EEG.csv` and `ECG.csv` – Extracted features from the raw dataset
- `clustered_data_kmeans.csv` – Final feature set with labels
- `eegnet_emotion_recognition_kmeans.pth` – Trained model checkpoint

## 🧠 Libraries Used

- `neurokit2`
- `scipy`, `pandas`, `numpy`
- `sklearn`, `matplotlib`
- `torch`, `gradio`

## 🚀 How to Run

```bash
pip install -r requirements.txt

# Step 1: Feature Extraction
python EEG.py
python ECG.py

# Step 2: Clustering + Model Training + Gradio
python main.py
