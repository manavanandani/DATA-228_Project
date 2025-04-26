# Big Data Pipeline for Real-Time Multi-Source Breast Cancer Risk Analysis

![Memory Usage Comparison](images/memory_graph.jpeg)

---

## 🌟 Badges

![Accuracy](https://img.shields.io/badge/Accuracy-88%25-brightgreen)
![Latency](https://img.shields.io/badge/Latency-<10ms-blue)
![Streaming Rate](https://img.shields.io/badge/Kafka%20Throughput-1000%2B%20msg%2Fs-orange)
![Deployment](https://img.shields.io/badge/Deployment-Spark%20Streaming%20%7C%20Kafka-yellow)

---

## 🗓 Project Overview

This project presents a **production-grade real-time big data pipeline** for **breast cancer risk analysis** using multimodal data:

- **Structured Electronic Health Records (EHR)**
- **High-resolution mammography images**

It combines the power of:

- Apache Kafka
- Apache Spark Structured Streaming
- Bloom Filters
- XGBoost Classifier
- CNN for medical imaging
- SHAP & Grad-CAM for Explainability

Focus: **Speed**, **Scalability**, **Interpretability**, and **Privacy**.

---

## 🌍 System Architecture

![System Architecture](images/img_pipeline_comparison.png.jpeg)

- Upper Flow: Local Python inference
- Lower Flow: Distributed PySpark inference
- Benchmarked on accuracy, scalability, and latency

---

## 🔍 Dataset and Data Preparation

### Structured Dataset
- BCSC Risk Factors Dataset (~2 Million Records)

### Image Dataset
- CBIS-DDSM Mammography Dataset (~3000 images)

### Preprocessing Flow

![Image Preprocessing](images/img_image_preprocessing.jpeg)

- Split into Train, Validation, Test
- Resize: 224x224
- Normalize: 1/255
- Data Augmentation (Flip, Zoom)

---

## 🎓 Model Building

### CNN for Mammography

![CNN Training](images/img_cnn_training_flow.jpeg)

- Conv2D layers + MaxPooling + Dropout
- Compiled with Adam optimizer
- Binary Cross Entropy Loss
- Validation Split: 20%
- ModelCheckpoint to save best model

### XGBoost for Structured Data
- Trained on full dataset
- Serialized using `joblib`
- Deployed in Spark for distributed inference

---

## 💡 Real-Time Risk Assessment Pipeline

### Streaming Flow

![Real-Time Flow](images/img_realtime_decision_flow.jpeg.jpeg)

- Kafka Streams incoming records
- Spark Structured Streaming micro-batches (every 500ms)
- Bloom Filter checks:
  - No Match → Risk = 0
  - Match → XGBoost Prediction
- Inline SHAP explanations

### Bloom Filter Training & Deployment

![Model Deployment Flow](images/img_model_training_deployment.jpeg.jpeg)

- Positive Cases only in Bloom Filter (13 million elements)
- XGBoost trained on full dataset
- Both serialized and deployed

---

## 🧠 Explainable AI (XAI)

### SHAP for XGBoost
- Feature importance explanations
- Global and Local interpretations

### Grad-CAM for CNN
- Heatmaps on mammography images

**Grad-CAM Samples:**

![GradCAM Output 1](images/gradcam_output.jpg)
![GradCAM Output 2](images/gradcam_output%20(1).jpg)
![GradCAM Output 3](images/gradcam_output%20(2).jpg)

---

## 📈 Performance Benchmarks

### Memory Usage

![Memory Usage Graph](images/memory_graph.jpeg)

### Inference Time

![Inference Time Graph](images/time_graph.jpeg)

---

## 🌐 Privacy Techniques

- **K-Anonymity:** 5-year age binning
- **Laplace Noise:** Noise addition
- **Race Generalization:** Rare race codes → 'Other'
- **l-Diversity:** Protects attribute disclosure

---

## 🛠 Commands and Project Flow

### 1. Setup Environment
```bash
pip install pyspark kafka-python xgboost shap lime tensorflow keras joblib
```

### 2. Train Models
```bash
python train_bloom_filter.py
python train_xgboost_model.py
python train_cnn_model.py
```

### 3. Start Kafka Streaming
```bash
python faker_data_gen.py
```

### 4. Real-Time Inference
```bash
spark-submit stream_inference.py
```

### 5. Image Inference
```bash
python local_image_inference.py  # Local Python
spark-submit distributed_image_inference.py  # Distributed PySpark
```

---

## 📢 Demo Pipeline Example

```
Kafka Topic -> Spark Micro-Batch -> Bloom Filter -> XGBoost Model -> SHAP Explainability
```
```
Kafka Topic -> Spark Micro-Batch -> CNN Model -> Grad-CAM Visualizations
```

---

## 💻 Technologies Used

| Component         | Technology |
|-------------------|------------|
| Data Storage      | HDFS       |
| Streaming Engine  | Kafka      |
| Processing Engine | Spark Structured Streaming |
| Modeling          | XGBoost, TensorFlow/Keras |
| Explainability    | SHAP, LIME, Grad-CAM |
| Privacy           | K-Anonymity, Laplace Noise |

---

## 👩‍💻 Team

- Basanth Periyapatna Roopa Kumar
- Mayank Kapadia
- Manav Anandani
- Nischitha Nagendran

> San Jose State University, Master of Data Analytics

---

## 🔗 References

(IEEE-style references listed in the full project report)

---

# ✨ Thank you for checking out our project!
