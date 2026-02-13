# 🩺 PneumoniaCare — AI-Based Pneumonia Detection from Chest X-ray

## 🌐 Live Demo
🔗 https://pneumonia-detection-ai-riuy.onrender.com


## 📌 Overview
PneumoniaCare is an AI-powered medical image analysis system that detects **Pneumonia from Chest X-ray images** using a Convolutional Neural Network (CNN).  
The application classifies X-ray images into **Normal, Viral, or Bacterial Pneumonia**, providing prediction confidence, infection stage, medical comment, downloadable report, and diagnosis history.

---

## 🎯 Objective
To develop an intelligent diagnostic assistant that:
- Detects pneumonia accurately from chest X-rays  
- Provides fast automated analysis  
- Generates confidence-based results  
- Classifies infection severity (Healthy / Moderate / Severe)  
- Produces structured medical reports  
- Maintains diagnosis history for tracking  

---

## 🚀 Key Features

- 📤 Upload Chest X-ray image for diagnosis  
- 🧠 AI Prediction: **NORMAL / VIRAL / BACTERIAL**  
- 📊 Confidence score for prediction accuracy  
- 🏥 Stage classification: *Healthy / Moderate Infection / Severe Infection*  
- 💬 Automated medical comment generation  
- 📄 Downloadable **PDF diagnosis report**  
- 🕘 Diagnosis history tracking with timestamp  
- 🌐 Web interface built using Flask  

---

## 🧠 Technical Implementation

- Developed a **CNN-based Deep Learning model** using PyTorch for medical image classification  
- Preprocessed and structured Chest X-ray dataset for training and validation  
- Integrated trained model into a **Flask web application** for real-time prediction  
- Implemented probability-based confidence scoring and stage mapping  
- Generated automated medical comments based on prediction class  
- Built PDF report generation system for structured output  
- Stored diagnosis results in JSON-based history tracking system  

---

## 🛠️ Tech Stack

**Programming Language:** Python  
**Deep Learning:** PyTorch, CNN  
**Image Processing:** OpenCV, NumPy  
**Web Framework:** Flask  
**Frontend:** HTML, CSS  
**Other:** JSON, PDF generation  

---

## 📊 Output

The system produces:

- Pneumonia classification (**Normal / Viral / Bacterial**)  
- Confidence percentage of prediction  
- Infection stage (Healthy / Moderate / Severe)  
- Automated medical interpretation  
- Downloadable PDF medical report  
- Diagnosis history with timestamp  

---

## ▶️ How to Run

pip install -r requirements.txt  
python app.py  

Open in browser:  
http://127.0.0.1:5000

---

## 📁 Project Structure

app.py                  → Flask web application  
train_pneumonia.py      → Model training script  
split_dataset.py        → Dataset preparation  
pneumonia_model.pth     → Trained CNN model  
history.json            → Diagnosis history storage  
templates/              → HTML UI pages  
static/                 → CSS and UI resources  

---

## 📌 Dataset Note

Due to repository size limitations, only sample images are included.  
The model was trained on a larger Chest X-ray dataset.

## 📸 Screenshots

![Normal](docs/screenshots/Normal%20result.jpg)  
![Viral](docs/screenshots/Viral%20result.jpg)  
![Bacterial](docs/screenshots/Bacterial%20result.jpg)  
![History](docs/screenshots/Diagnosis%20history.jpg)

