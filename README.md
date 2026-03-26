🌿 Plant Disease Classification using CNN (VGG16 Fine-Tuning)
🚀 Project Overview

This project focuses on automatic plant disease detection using deep learning. The model is trained on a Kaggle dataset and is capable of identifying:

🌱 Plant type (13 classes)
🦠 Disease type (29 classes)

A fine-tuned VGG16 Convolutional Neural Network (CNN) is used with a dual-output architecture, allowing the model to simultaneously predict both the plant and its disease.

The application is deployed online using Render, making it accessible via a web interface.

📊 Dataset
Source: Kaggle Plant Disease Dataset
Contains images of healthy and diseased plant leaves
Total Classes:
🌿 13 Plant types
🦠 29 Disease categories
🧠 Model Architecture
🔹 Base Model
Pretrained VGG16
Weights initialized from ImageNet
🔹 Custom Layers
Feature extraction using VGG16 convolutional base
Fully connected layers added on top
🔹 Dual Output Design

The model has two output heads:

Plant Classification Head
Disease Classification Head

This allows:

Better feature sharing
Improved accuracy
Reduced computational cost compared to separate models
⚙️ Training Details
Loss Function:
Categorical Crossentropy (for both outputs)
Optimizer:
Adam
Techniques Used:
Transfer Learning
Fine-Tuning (unfreezing top VGG layers)
Data Augmentation (rotation, zoom, flip)
🌐 Deployment
Platform: Render
Backend: Python (Flask / FastAPI if used)
Users can:
Upload leaf images
Get predictions for plant and disease
🛠️ Tech Stack
Python 🐍
TensorFlow / Keras
OpenCV
NumPy & Pandas
Flask / FastAPI
Render (Deployment)
📸 Features
✅ Upload plant leaf image
✅ Predict plant type
✅ Detect disease
✅ Fast inference
✅ Web-based interface
📂 Project Structure
├── dataset/
├── models/
│   └── vgg16_finetuned.h5
├── app/
│   ├── main.py
│   ├── utils.py
│   └── templates/
├── static/
├── requirements.txt
└── README.md
📈 Future Improvements
Add more plant species
Improve accuracy with larger datasets
Mobile app integration
Real-time detection using camera
🤝 Contribution

Contributions are welcome! Feel free to fork the repo and submit pull requests.
