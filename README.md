# GrainPalette - Rice Type Classification Using Deep Learning

GrainPalette is a deep learning-based project focused on classifying five types of rice grains using image data. This project is part of the SmartInternz Internship program and uses transfer learning with MobileNetV4 to achieve accurate classification.

---

## 📌 Project Overview

- **Project Title:** GrainPalette - A Deep Learning Odyssey In Rice Type Classification Through Transfer Learning  
- **Domain:** Computer Vision, Deep Learning  
- **Frameworks:** TensorFlow, Keras  
- **Model:** MobileNetV4 (Transfer Learning)  
- **Dataset:** [Rice Image Dataset (Kaggle)](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)

---

## 📂 Dataset Info

The dataset contains images of 5 rice varieties:
1. Arborio
2. Basmati
3. Ipsala
4. Jasmine
5. Karacadag

Each class contains 200+ high-resolution images of individual rice grains.

---

## 🧠 Model Architecture

We use **MobileNetV4** with transfer learning for feature extraction and a custom fully connected head for classification.

- Input shape: `224x224x3`
- Output: 5-class softmax layer
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/grainpalette-rice-classifier.git
   cd grainpalette-rice-classifier
2.Install dependencies:
```bash
  pip install -r requirements.txt
Train the model:
  python train.py
Evaluate or Predict:
  python predict.py --image path_to_image.jpg
