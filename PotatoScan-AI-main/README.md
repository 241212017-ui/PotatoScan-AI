# 🥔 PotatoScan AI

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CNN](https://img.shields.io/badge/Model-Custom_CNN-success?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-99.57%25-brightgreen?style=for-the-badge)

**An AI-powered potato leaf disease detection system built using a custom Convolutional Neural Network trained entirely from scratch.**

🌐 **Live Demo:** https://itzmoulik-potato-leaf-prediction-model.hf.space/

</div>

---

## 📋 Overview

PotatoScan AI is a deep learning-powered web application designed to detect potato leaf diseases from images.

- Disease prediction
- Confidence score
- Disease severity assessment
- Treatment recommendations
- Prevention guidelines
- AI-powered agricultural assistance

Classes:
- 🍂 Early Blight
- 🌧️ Late Blight
- 🌿 Healthy

---

## ✨ Features

### 🔬 Disease Detection
- Custom CNN trained from scratch
- Three-class potato leaf classification
- Confidence-calibrated predictions
- Test-Time Augmentation (TTA)
- Temperature Scaling

### 🌾 Agricultural Support
- Disease descriptions
- Treatment recommendations
- Prevention guidelines
- Severity assessment

### 🌐 Web Application
- FastAPI backend
- Modern responsive UI
- Drag-and-drop image upload
- AI-powered advisory chatbot

---

## 📊 Model Performance

| Metric | Value |
|----------|----------:|
| Validation Accuracy | 99.89% |
| Test Accuracy | 99.57% |
| Mean Prediction Confidence | 99.19% |
| Classes | 3 |
| Parameters | 10.6M+ |

---

## 📂 Dataset

Merged dataset from:
- PlantVillage
- PlantDoc

Total Images: ~9,280

Classes:
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

See `dataset/README.md` for details.

---

## 📸 Screenshots

![Home Page](screenshots/home_page.png)

![Early Blight](screenshots/prediction_early_blight.png)

![Healthy](screenshots/prediction_healthy.png)

---

## 🛠️ Tech Stack

- PyTorch
- TorchVision
- FastAPI
- Uvicorn
- HTML
- CSS
- JavaScript
- NumPy
- Pillow
- Pandas

---

## 👨‍💻 Authors

**Moulik Dotasara**

GitHub: https://github.com/moulik3637

**Razzak**

GitHub: https://github.com/241212017-ui

---

⭐ If you found this project useful, consider giving it a star!
<!-- minor README update -->
