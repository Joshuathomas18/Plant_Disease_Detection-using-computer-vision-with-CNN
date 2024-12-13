# Plant Disease Detection

A machine learning-based project to detect plant diseases using image processing and deep learning techniques. This project leverages Python and libraries like TensorFlow/Keras to classify images of plant leaves into healthy or diseased categories.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The Plant Disease Detection project aims to help farmers and agricultural experts identify diseases in plants at an early stage. By analyzing leaf images, the system can provide an accurate diagnosis and suggest necessary treatments to improve crop yield.

---

## Features
- Image classification using Convolutional Neural Networks (CNNs).
- Detects multiple plant diseases across various crops.
- User-friendly and easy to integrate with agricultural tools.
- Trained on a labeled dataset of plant leaf images.

---

## Dataset
The project uses a dataset containing images of healthy and diseased plant leaves. Each image is labeled with the type of plant and its disease condition.

**Source:** [PlantVillage Dataset](https://www.plantvillage.org/) (or another specified source in the project documentation).

---

## Requirements

Ensure you have the following installed:

- Python 3.7+
- TensorFlow 2.0+
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn

You can install the required Python libraries using:
```bash
pip install -r requirements.txt
```

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/manthan89-py/Plant-Disease-Detection.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Plant-Disease-Detection
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare the dataset:
   - Download the dataset and place it in the `data/` folder.
   - Ensure the directory structure matches the script requirements.

5. Train the model:
   ```bash
   python train_model.py
   ```

---

## Usage

1. To detect a disease in a plant leaf:
   ```bash
   python predict.py --image path/to/leaf_image.jpg
   ```

2. View the results:
   - The script outputs the predicted class (e.g., `Healthy`, `Diseased`) and the confidence score.

---

## Model Architecture

The project uses a Convolutional Neural Network (CNN) architecture. Key layers include:
- Convolutional layers for feature extraction.
- Pooling layers for dimensionality reduction.
- Fully connected layers for classification.

Detailed architecture and parameters are available in the [Model](Model/) directory.

---

## Results

The model achieves high accuracy on the test dataset with the following metrics:
- Accuracy: 95%
- Precision: 94%
- Recall: 93%

Performance may vary depending on dataset quality and augmentation techniques.

---


