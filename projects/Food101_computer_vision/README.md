# Food101 Computer Vision project

This is a computer vision project that focuses on classifying food images using the Food101 dataset. 
The project leverages mixed precision training techniques to improve the model's performance and training speed.

## Project Overview
In this project, I explored various techniques and approaches to enhance the performance of the food image classification model. 
The key highlights of the project include:

- Utilizing TensorFlow Datasets to download and explore the Food101 dataset, which consists of 101 food categories.
- Incorporating advanced data loading techniques such as parallelizing, prefetching, and batching to optimize the loading and processing speed of the dataset.
- Employing several modeling callbacks, including the TensorBoard callback, ModelCheckpoint callback, EarlyStopping callback, and ReduceLROnPlateau callback, to monitor and track the model's performance during training.
- Implementing mixed precision training, a technique that utilizes lower-precision numerical representations to speed up training without sacrificing model accuracy.
- Building a feature extraction model with EfficientNet as the base model, taking advantage of its pretrained weights on the ImageNet dataset.
- Utilizing fine-tuning techniques to further improve the model's performance by training the feature extraction layers on the Food101 dataset.
- Comparing the training results of different experiments using TensorBoard, providing visual insights into the model's performance.

## Requirements
To run the project, make sure you have the following dependencies installed:

- TensorFlow
- TensorFlow Datasets
- NumPy
- Matplotlib

## Acknowledgements
This project was inspired by the "Food101 - Computer Vision with Mixed Precision Training" from the Zero to Mastery (ZTM) TensorFlow course. Some of the project's components and techniques were learned and adapted from the course materials.

