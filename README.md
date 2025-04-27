# 🚀 Deep Learning Projects

This repository is a hands-on collection of deep learning projects from the Zero to Mastery TensorFlow course. Each folder contains a Jupyter Notebook demonstrating a different problem domain and showcasing various techniques for data analysis and prediction.

- `notebooks/`: all project notebooks (.ipynb) with code, explanations, and visualizations

- `saved_models/`: pre-trained TensorFlow models produced during the course

- `utils/`: reusable functions for model training & evaluation, plotting learning curves, and preprocessing image data

---

## 🔹 Project Descriptions

1. **Binary Text Classification with Kaggle’s “Natural Language Processing with Disaster Tweets”**

In this project, I compare multiple architectures for the binary classification task of detecting disaster-related tweets using Kaggle’s `nlp_getting_started` dataset. Implemented models include **Naive Bayes**, a **Feed-Forward Neural Network**, **LSTM**, **GRU**, **Bidirectional LSTM**, **1D Convolutional Neural Network**, and a pretrained feature extractor from TensorFlow Hub. By evaluating each model’s performance, using metrics such as accuracy, precision, recall, and F1 score; I identify the most effective approach for this disaster-tweet classification problem.

2. **Transfer Learning with Feature Extraction and Fine-tuning**

In this project, I examine transfer learning on a 10% subset of the `Food-101 image` dataset using the **EfficientNet** architecture. I compare two approaches: **feature extraction** with frozen convolutional layers and full-model **fine-tuning**, and evaluate each method’s performance using metrics such as accuracy and validation loss. To shed light on the model’s behavior, I also visualize the most frequently misclassified images, revealing its strengths and failure modes across various food categories.
   
4. **Fashion MNIST Classification**

In this project, I develop Convolutional Neural Network models to automatically classify the 70,000 grayscale images in the `Fashion MNIST` dataset into 10 clothing categories. I implement and compare multiple CNN architectures, incorporate **data augmentation** and **regularization techniques** (dropout, batch normalization), and **tune hyperparameters** (learning rate, batch size, number of filters) to maximize generalization. Model performance is evaluated through accuracy and loss curves, confusion matrices, and classification reports.
   

5. **Binary Classification using sklearn make_moons**

This project explores binary classification task with the `make_moons` dataset from scikit-learn, aiming to build a model that accurately separates data points into two classes.
   
   

