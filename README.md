🌿<h1>PLANT DISEASE DETECTION USING COMPUTER VISION</h1>

📜 **Introduction**

Plant diseases significantly impact agricultural productivity, leading to economic losses and food insecurity. Early and accurate detection of plant diseases is crucial for taking timely actions to prevent their spread.

This project utilizes Convolutional Neural Networks (CNN), a powerful deep learning technique, to analyze leaf images and classify them into healthy or diseased categories. By leveraging machine learning, the model automates the detection process, providing a cost-effective and scalable solution for farmers and agricultural experts.

🚀 **Features**

Automatic Disease Detection: Classifies plant leaves as healthy or diseased.
Deep Learning-based: Utilizes a Convolutional Neural Network (CNN) for image classification.
High Accuracy: Trained on a dataset of labeled leaf images to ensure reliable results.
User-Friendly: Designed for easy integration and usability for farmers and researchers.

🗂️ **Dataset**


The project uses a publicly available dataset containing images of healthy and diseased plant leaves.

**Source**: Kaggle PlantVillage Dataset
**Categories**: Includes multiple classes such as "Healthy", "Blight", "Rust", etc.
🛠️ **Technologies Used**
**Python**: Programming language.
**TensorFlow/Keras**: Deep learning frameworks for building the CNN model.
**OpenCV**: For image preprocessing.
**NumPy & Pandas**: Data manipulation.
**Matplotlib**: Visualization of results.

📥 **Installation**

<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

</code>
</pre>




🚦 **Usage**

<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=30, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training and validation data
train_generator = datagen.flow_from_directory(
    'C:/Users/User/Desktop/disease detection with CNN/Plant_leave_diseases_dataset_without_augmentation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'C:/Users/User/Desktop/disease detection with CNN/Plant_leave_diseases_dataset_without_augmentation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

</code>
</pre>

**It identifies**:
      1.Found 44371 images belonging to 39 classes.  
      2.Found 11077 images belonging to 39 classes.

**Training the Model**: Use train_model.py to train the CNN model on the dataset.
<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
</code>
</pre>


**Prediction**: Use predict.py to classify new leaf images.
<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
def hello_world():
    print("Hello, world!")
</code>
</pre>


**Output**:
<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stopping]
)
</code>
</pre>

<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
# Test dataset generator
test_generator = datagen.flow_from_directory(
    'C:/Users/User/Desktop/disease detection with CNN/Plant_leave_diseases_dataset_with_augmentation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
</code>
</pre>


Predicted Class: Blight  

Confidence: 90.3%

# VISUALIZATIONS


