# RetinaAI-Decoding-Retinal-Eye-Diseases-Using-Deep-Learning

An intelligent deep learning-based system for automated detection and classification of retinal eye diseases from medical images.

## Introduction

Retinal diseases such as diabetic retinopathy, glaucoma, and age-related macular degeneration are among the leading causes of vision impairment and blindness worldwide. Early diagnosis of these conditions is essential for effective treatment and prevention of permanent vision loss. However, traditional diagnostic methods rely heavily on manual analysis of retinal fundus images by ophthalmologists, which is time-consuming and prone to human error.

RetinaAI aims to develop an automated deep learning-based system that can analyze retinal images and accurately detect various eye diseases. By leveraging convolutional neural networks (CNNs), the system assists medical professionals by providing fast, consistent, and reliable diagnostic support.

## Overview

This project focuses on developing an intelligent system using Deep Learning to automatically detect and classify retinal eye diseases from fundus images. The system takes retinal images as input, processes them using trained convolutional neural network models, and produces accurate predictions regarding the presence and type of eye disease.

The goal of RetinaAI is to enhance diagnostic accuracy, reduce the workload of ophthalmologists, and enable early intervention for better patient outcomes.

## 🎯 Objectives

To collect and prepare a high-quality retinal image dataset.

To preprocess and augment images for better model performance.

To design and train a CNN-based deep learning model.

To evaluate the model using standard performance metrics.

To provide an automated prediction system for retinal disease detection.

## 🧠 Technologies Used

Python

TensorFlow / PyTorch

OpenCV

NumPy, Pandas, Matplotlib

Jupyter Notebook

## ⚙️ System Workflow

Retinal Image Collection

Image Preprocessing and Augmentation

Feature Extraction using CNN

Model Training and Validation

Disease Classification and Prediction

## 📂 Project Structure
```
RetinaAI/
│
├── dataset/          # Retinal images (train, test, validation)
├── models/           # Trained deep learning models
├── notebooks/        # Jupyter notebooks for experiments
├── src/              # Source code
├── results/          # Performance metrics and outputs
└── README.md         # Project documentation
```
## 📄 Abstract

Early detection of retinal diseases is critical for preventing irreversible vision loss. This project proposes a deep learning-based automated system for detecting retinal eye diseases using fundus images. The system employs convolutional neural networks to extract meaningful features from images and classify them into different disease categories.

Experimental results demonstrate that the proposed model achieves high accuracy and generalizes well on unseen data. RetinaAI can serve as an effective decision-support tool for ophthalmologists and has the potential to be integrated into real-world clinical systems.

## 🧩 Problem Statement

Manual diagnosis of retinal diseases requires expert ophthalmologists and is often subject to variability and delays. In many regions, especially rural areas, access to specialized eye care is limited, resulting in late diagnosis and preventable vision loss.

There is a need for an automated, accurate, and scalable system that can analyze retinal images and detect eye diseases at an early stage. The challenge is to design a system that can handle large volumes of medical images while maintaining high diagnostic accuracy. RetinaAI addresses this problem by using deep learning techniques to provide fast and reliable retinal disease classification.


##  Architecture Diagram

<img width="3839" height="1437" alt="main" src="https://github.com/user-attachments/assets/eef1e677-6d8b-4c0d-ba39-82a290817d96" />



## Code Implementation
```
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR

def load_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_data = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_data, val_data


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from utils.config import IMG_SIZE, NUM_CLASSES

def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from utils.config import IMG_SIZE, NUM_CLASSES

def build_hybrid_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = tf.expand_dims(x, axis=1)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization()(attn_output + x)
    x = tf.squeeze(x, axis=1)

    x = Dense(256, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


from utils.data_loader import load_data
from models.hybrid_model import build_hybrid_model
from utils.config import EPOCHS

train_data, val_data = load_data()
model = build_hybrid_model()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

model.save("retinaai_model.h5")



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.config import IMG_SIZE, TEST_DIR

model = tf.keras.models.load_model("retinaai_model.h5")

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy = model.evaluate(test_data)
print("Test Accuracy:", accuracy)
 


```


## 🔮 Future Work

Improve accuracy using larger and more diverse datasets.

Extend the system to detect more retinal conditions.

Deploy the model as a web or mobile application.

Integrate explainable AI for better clinical interpretability.
