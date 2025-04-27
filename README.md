# 🚀 Deep Learning Projects

This repository is a hands-on collection of deep learning projects from the Zero to Mastery TensorFlow course. Each folder contains a Jupyter Notebook demonstrating a different problem domain and showcasing various techniques for data analysis and prediction.

- `notebooks/`: all project notebooks (.ipynb) with code, explanations, and visualizations

- `saved_models/`: pre-trained TensorFlow models produced during the course

- `utils/`: reusable functions for model training & evaluation, plotting learning curves, and preprocessing image data

---

## 🔹 Project Descriptions

1. **Binary Text Classification with Kaggle’s “Natural Language Processing with Disaster Tweets” Project**

In this project, I compare multiple architectures for the binary classification task of detecting disaster-related tweets using Kaggle’s `nlp_getting_started` dataset. Implemented models include **Naive Bayes**, a **Feed-Forward Neural Network**, **LSTM**, **GRU**, **Bidirectional LSTM**, **1D Convolutional Neural Network**, and a pretrained feature extractor from TensorFlow Hub. By evaluating each model’s performance, using metrics such as accuracy, precision, recall, and F1 score; I identify the most effective approach for this disaster-tweet classification problem.

2. **Food101 Computer Vision Project**

This computer vision project tackles food image classification on the `Food101 dataset` by leveraging **mixed precision training** to boost both model performance and training speed. I began by using TensorFlow Datasets to download and explore all 101 food categories, then optimized data throughput with **parallelized loading**, **prefetching**, and **batching**. During training, I integrated a suite of callbacks: **TensorBoard** for live metrics visualization, **ModelCheckpoint** to preserve the best weights, **EarlyStopping** to halt stagnating runs, and **ReduceLROnPlateau** to adaptively lower the learning rate—to ensure robust monitoring and control. Mixed precision training was employed throughout, harnessing lower-precision computations for faster iterations without compromising accuracy. The core model is built on **EfficientNet pretrained on ImageNet**, which I first used as a frozen feature extractor and then fine-tuned on Food101 to further refine its capability. Finally, I compared the outcomes of different experimental configurations in TensorBoard, using the visual analytics to guide iterative improvements.

3. **Transfer Learning with Feature Extraction and Fine-tuning Project**

In this project, I examine transfer learning on a 10% subset of the `Food-101 image` dataset using the **EfficientNet** architecture. I compare two approaches: **feature extraction** with frozen convolutional layers and full-model **fine-tuning**, and evaluate each method’s performance using metrics such as accuracy and validation loss. To shed light on the model’s behavior, I also visualize the most frequently misclassified images, revealing its strengths and failure modes across various food categories.
   
4. **Fashion MNIST Classification Project**

In this project, I develop Convolutional Neural Network models to automatically classify the 70,000 grayscale images in the `Fashion MNIST` dataset into 10 clothing categories. I implement and compare multiple CNN architectures, incorporate **data augmentation** and **regularization techniques** (dropout, batch normalization), and **tune hyperparameters** (learning rate, batch size, number of filters) to maximize generalization. Model performance is evaluated through accuracy and loss curves, confusion matrices, and classification reports.
   

5. **Binary Classification with Scikit-Learn Project**

This project explores binary classification task with the `make_moons` dataset from scikit-learn, aiming to build a model that accurately separates data points into two classes.
   
   

