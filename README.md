# Emotion Detection using ResNet

A Deep Learning project that classifies human facial expressions into different emotion categories (e.g., Happy, Sad, Angry, Surprise, etc.) using a Residual Network (ResNet) architecture.

##  Project Overview
This project leverages the power of **ResNet (Residual Networks)** to overcome the vanishing gradient problem, allowing for a deeper and more accurate feature extraction from facial images. The model was built using Python and VS Code, and is designed to recognize emotions in real-time or from static images.

---

## 🔗 Live Demo

Try the live application here:  
**https://huggingface.co/spaces/aa-chal01/EmotionDetection**

> Note: Allow camera access for real-time emotion detection.

---

##  Model Architecture
- **Base Model:** ResNet50
- **Framework:** Keras
- **Dataset:** FER2013 (https://www.kaggle.com/datasets/msambare/fer2013).
---

##  Features

- Detects emotions from facial images
- Uses deep learning for accurate classification
- Supports multiple emotion classes
- Can be extended for real-time emotion detection
- Easy-to-understand and modular code structure

---

##  How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

###  Install Dependencies
```bash
pip install -r requirements.txt
```
### Train the Model
```bash
python train.py
```

### Run Emotion Detection
```bash
python test.py
```
---

## Results
The trained model is able to classify facial expressions into different emotion categories with reasonable accuracy. The performance depends on factors such as:
- Dataset size and quality
- Lighting and facial orientation in images
- Model architecture and training epochs


