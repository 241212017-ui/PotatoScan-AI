# 🥔 PotatoScan AI

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CNN](https://img.shields.io/badge/Model-Custom_CNN-success?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-99.57%25-brightgreen?style=for-the-badge)

An AI-powered potato leaf disease detection system built with a custom Convolutional Neural Network and a modern web interface.

🌐 Live Demo: https://itzmoulik-potato-leaf-prediction-model.hf.space/

</div>

---

## 📌 Overview

PotatoScan AI is a deep learning project that identifies potato leaf diseases from uploaded images. It can classify healthy leaves and common diseases such as Early Blight and Late Blight, while also providing confidence scores, disease details, and recommendations.

### Supported Classes
- 🍂 Early Blight
- 🌧️ Late Blight
- 🌿 Healthy

---

## ✨ Features

- Custom CNN trained from scratch
- Image-based disease classification
- Confidence score for predictions
- Disease severity insights
- Treatment and prevention guidance
- FastAPI backend with a responsive frontend
- AI-powered advisory chatbot support

---

## 🧠 Model Performance

| Metric | Value |
|---|---:|
| Validation Accuracy | 99.89% |
| Test Accuracy | 99.57% |
| Mean Prediction Confidence | 99.19% |
| Classes | 3 |
| Parameters | 10.6M+ |

---

## 📂 Dataset

The model was trained on a merged dataset from:
- PlantVillage
- PlantDoc

Total images: approximately 9,280

---

## 🛠️ Tech Stack

- Python
- PyTorch
- TorchVision
- FastAPI
- Uvicorn
- HTML/CSS/JavaScript
- NumPy
- Pillow
- Pandas

---

## ▶️ Run Locally

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   uvicorn main_v5:app --reload --port 8000
   ```
4. Open:
   ```text
   http://127.0.0.1:8000/static/index.html
   ```

> For the chatbot feature, set the environment variable `OPENROUTER_API_KEY`.

---

## 📁 Project Structure

```text
PotatoScan-AI-main/
├── main_v5.py
├── requirements.txt
├── models_scratch/
├── dataset/
├── static/
└── screenshots/
```

---

## 📸 Screenshots

![Home Page](screenshots/home_page.png)

![Early Blight Prediction](screenshots/prediction_early_blight.png)

![Healthy Prediction](screenshots/prediction_healthy.png)

---

## 🔗 GitHub Repository

Repository: https://github.com/241212017-ui/PotatoScan_AI

---

## 👨‍💻 Authors

- Moulik Dotasara — https://github.com/moulik3637
- Razzak — https://github.com/241212017-ui

---

⭐ If you found this project useful, consider giving it a star on GitHub.
