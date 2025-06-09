
# 🫀 Heart Attack Risk Prediction from Retinal Images using Hybrid ResNet-LSTM 🧠📸

## 🔍 Overview
This project predicts **heart attack risk** by analyzing **retinal fundus images** using a **deep learning pipeline** that combines:

- Image preprocessing and feature extraction
- Medical rule-based risk classification
- A **hybrid deep neural network**: **ResNet50 + LSTM** enhanced with the **Mish activation function**
- Model interpretability using prediction logs and plots

The project aims to build a reliable risk stratification system for cardiovascular conditions based on retinal biomarkers.

---

## 🧠 Key Features

- 🔬 Feature extraction from retina: AVR, Tortuosity, Microaneurysms, CDR, Fractal Dimension  
- ⚙️ Risk scoring using medical rules: Low, Medium, High  
- 🏗️ Deep learning classifier using **Transfer Learning** (ResNet50) + **Temporal modeling** (LSTM)  
- 🔁 Mish activation improves gradient flow  
- 📊 Visual tracking of training/validation loss and accuracy  
- 🧪 Final prediction export to CSV for deployment

---

## 🧰 Technologies Used

- **Python**
- **PyTorch** (for CNN + LSTM model)
- **OpenCV, PIL, skimage** (for image handling)
- **Scikit-learn** (for preprocessing and metrics)
- **Matplotlib** (for plotting)
- **ResNet50** (pretrained on ImageNet)
- **Mish Activation Function** (custom PyTorch module)

---

## 📁 Folder Structure
```
📦HeartAttack-Retinal/
├── mish_AF.py                 # Main training and prediction script
├── retinal_features_1.csv     # Extracted features with labels (auto-generated)
├── hybrid_resnet_lstm.pth     # Trained model weights (auto-saved)
├── Output_Results.csv         # Final predictions (auto-saved)
└── README.md                  # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/heart-attack-retinal.git
cd heart-attack-retinal
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary><strong>📦 Required Packages</strong></summary>

- torch  
- torchvision  
- numpy  
- opencv-python  
- scikit-learn  
- pandas  
- pillow  
- matplotlib  
- scikit-image  
</details>

### 3. Place Dataset
Put your retinal `.jpg` images in the dataset folder:
```
C:/Users/Zaid Chikte/Downloads/Diagnosis of Diabetic Retinopathy.v3i.folder/
```

### 4. Run the Script
```bash
python mish_AF.py
```

---

## 🔢 Risk Classification Criteria

The risk is determined based on the following extracted features:
- **AVR (Arteriolar-to-Venular Ratio)**
- **Tortuosity**
- **Microaneurysms**
- **Cup-to-Disc Ratio (CDR)**
- **Fractal Dimension**

A rule-based risk score is computed, then labeled as:
- `Low Risk`: score ≤ 2
- `Medium Risk`: score 3–5
- `High Risk`: score > 5

---

## 📈 Sample Output Plots

Training and validation curves are plotted at the end of training:
- 📉 Training vs Validation Loss  
- 📈 Training vs Validation Accuracy  

Also, all predictions are saved to:
```bash
Output_Results.csv
```

---

## 🧪 Example Prediction Output
```
Image_Name           Predicted_Risk
------------------------------------
image_001.jpg        High Risk
image_002.jpg        Low Risk
image_003.jpg        Medium Risk
```

---

## 🩺 Medical Disclaimer
> This model is intended for **research and educational purposes** only. It is **not a certified diagnostic tool**. Always consult a medical professional for actual clinical decisions.

---

