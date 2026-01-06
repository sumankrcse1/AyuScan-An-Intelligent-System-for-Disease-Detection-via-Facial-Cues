# AyuScan-An-Intelligent-System-for-Disease-Detection-via-Facial-Cues

# AyuScan 2.0 – Intelligent Disease Detection via Facial Cues

## 🧠 Overview
AyuScan 2.0 is an AI-powered health analyzer that detects potential diseases and imbalances from facial images using computer vision and deep learning techniques. The system integrates DeepFace analysis, TensorFlow models, and OpenCV-based feature extraction to provide medical and Ayurvedic insights.

---

## ✨ Features
- Image upload-based disease prediction
- Multi-disease confidence analysis
- Emotion, age, and race estimation
- Eye-based jaundice detection
- Skin texture and symmetry analysis
- Ayurvedic Tri-Dosha assessment
- Personalized health recommendations

---

## 🛠 Technology Stack
- **Backend:** Python, Flask
- **Computer Vision:** OpenCV, NumPy
- **Deep Learning:** TensorFlow, Keras
- **Facial Analysis:** DeepFace, MTCNN
- **Explainability:** Grad-CAM
- **Utilities:** gdown, PyYAML, dotenv

---
<img width="1366" height="768" alt="Screenshot (362)" src="https://github.com/user-attachments/assets/7335252e-d878-426e-be05-d66d13194f3e" />
<img width="1366" height="768" alt="Screenshot (363)" src="https://github.com/user-attachments/assets/5057fc32-feeb-4c68-bb1d-16689f621515" />
<img width="1366" height="768" alt="Screenshot (364)" src="https://github.com/user-attachments/assets/2793a37d-a4a9-41a7-a0df-54331a0e03e6" />



## 📂 Project Structure

AyuScan 2.0
│── app.py
│── requirements.txt
│── utils/
│ │── analyzer.py
│ │── color_utils.py
│ │── overlay.py
│ │── quality.py
│ │── regions.py
│── static/
│── templates/
│── uploads/
│── logs/
│── venv/


---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/AyuScan-2.0.git
cd AyuScan-2.0
pip install -r requirements.txt
python app.py

