# **Deepfake Detection - Final Year Project**

Enhanced for modern deepfake datasets like **WildRF**, **CollabDiff**, and GenAI-generated images. Includes a **Streamlit-based frontend** for real-time deepfake prediction.

---

## 🧠 PART 1: Training Pipeline and Backend (Jupyter Notebooks)

### 📚 Overview

This project integrates **ResNet-50**, **ViT**, and **XceptionNet** into a **two-module hierarchical ensemble** architecture for robust deepfake detection. All training and testing was done in notebooks before deployment into a real-time application.

---

### 📂 Project Structure (Training + Backend)

```
Training and Testing ( Notebook )/
│
├── Module 1/               # Global features
│   ├── DNNs/
│   ├── ResNet/
│   ├── ViTs/
│   ├── Module 1.ipynb
│   └── Module1_feature_extraction.ipynb
│
├── Module 2/               # Local features
│   └── Module2.ipynb
│
├── Ensembling/             # Final ensemble logic
├── models/                 # Trained model weights
```

---

### 🧬 Model Architecture

![arch2 (1)](https://github.com/user-attachments/assets/99befa93-e85e-4d6c-9eec-848e3c8e3214)

#### **Module 1: Global Feature Fusion**

* **ResNet-50**: Extracts multi-level semantic features
* **ViT**: Captures global contextual features using attention from image patches
* Fusion through attention mechanisms

#### **Module 2: Local Multi-Stream Analysis**

* **YOLOv8**: Detects and crops facial regions

* Streams:

  * **XceptionNet**: Semantic features from facial textures
  * **Sobel Edge Detection**: Captures edge-level manipulations

* Features are fused and passed through a classifier

#### **Final Ensemble**

* Logits from both modules are combined for final classification

---

### 📦 Prepare Data

Download and extract the datasets:

```bash
!pip install gdown
from pathlib import PosixPath

# WildRF
!gdown --id 1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn -c
!unzip -q -n WildRF.zip

# CollabDiff
!gdown --id 1GpGvkxQ7leXqCnfnEAsgY_DXFnJwIbO4 -c
!unzip -q -n CollabDiff.zip
```

---

### 🔧 Step-by-Step Training Pipeline

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sonu2164/Deepfake-Image-Detection
cd "Training and Testing ( Notebook )"
```

---

### 🧪 Module 1: Global Features

#### ✅ Step 1: Fine-tune ViT and ResNet-50

* Use `ViTb16_finetuned.ipynb` and `ResNetFinetuned.ipynb` to fine-tune and save the models.
* The resulting files will be:

  * `pretrained_vit_state_dict.pth`
  * `pretrained_resnet50_state_dict.pth`

#### ✅ Step 2: Feature Extraction

* Run `Module1_feature_extraction.ipynb` using the above weights.
* This will extract and save feature embeddings as CSVs.

#### ✅ Step 3: Logit Generation

* Use `DNN_M1_WildRF.ipynb` or the refined `Module 1.ipynb` to compute Module 1 logits.

---

### 🔬 Module 2: Local Streams

#### ✅ Step 4: Local Stream Processing

Run `Module2.ipynb` to:

* Detect faces via YOLOv8
* Extract features using:

  * **XceptionNet**
  * **Sobel Filter**
* Fuse and classify using a custom architecture

---

### 🔄 Step 5: Final Ensemble

* Use `Ensemble.ipynb` to combine predictions from both modules and generate final outputs.

---

### ⚠️ Ensure that all model weights are correctly downloaded and placed.

## 📁 Model Checkpoints

📦 [Google Drive Link for Trained Models](https://drive.google.com/drive/folders/1aXe_5_m0Hmg8D9bV6mu4_LQucGnWYDfX?usp=sharing)

---

## 🌈 PART 2: Streamlit Frontend

### 🎯 Overview

Once training was complete, the model was converted into a modular inference pipeline and deployed using **Streamlit**.

---

### 📂 Project Structure (Frontend)

```
deepfake detection streamlit application/
│
├── app.py                     # Main Streamlit app
├── predict.py                 # Model inference logic
├── feature_extraction.py      # Feature handlers for ViT and ResNet
├── yolov8n.pt                 # YOLOv8 face detection weights
├── uploaded_images/           # Uploaded image cache
├── uploaded_images_csv/       # Result log CSVs
├── models/                    # Trained model weights
├── requirements.txt
```

---

### ▶️ Run the Streamlit App

1. **Navigate to frontend folder**:

```bash
cd "deepfake detection streamlit application"
```

2. **Install requirements**:

```bash
pip install -r requirements.txt
```

3. **Download and place models** in `models/` using the link above.

4. **Launch the app**:

```bash
streamlit run app.py
```

---

### 🖼️ App Features

* **Upload Support**: Upload images/videos or capture via webcam
* **Live Inference**: Uses ensemble model to classify as real or fake

---

## 🔮 Future Scope
* 🎥 Extend support for video/audio deepfakes and streaming input
* ☁️ Deploy using **Docker**, **Render**, or **HuggingFace Spaces**


