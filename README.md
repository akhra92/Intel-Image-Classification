# Objects Classification using ResNext101

This project focuses on classifying various objects using a ResNext101 pre-trained model from timm library. The goal is to achieve high accuracy in identifying different objects based on visual features.

You can also deploy this model using streamlit. Demo can accessed using the following link: [Intel Image Classification Demo](https://intel-image-classification-demo.streamlit.app/)

---

## 📊 Project Overview

- **Task**: Multi-class image classification
- **Dataset**: Images of cars belonging to various objects (e.g., buildings, forest, mountain, etc.)
- **Model**: ResNext101 (pre-trained model from timm library)
- **Framework**: PyTorch
- **Evaluation Metrics**: Accuracy, Loss

---

## 🗂️ Dataset Samples

Below are some random samples from the dataset with corresponding labels:

![Dataset Samples](assets/samples.png)

---

## 📈 Training & Validation Curves

Here are the learning curves showing **Accuracy** and **Loss** over epochs:

### Train and Validation Curves
![Curves](assets/learning_curve.png)

---

## 🚀 How to Run

1. Clone the repository:
   
   ```
   git clone https://github.com/akhra92/Intel-Image-Classification.git
   cd Intel-Image-Classification
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
   
4. Train and test the model:

   ```
   python main.py
   ```

5. Deploy in local using streamlit:
   
   ```
   streamlit run demo.py
   ```