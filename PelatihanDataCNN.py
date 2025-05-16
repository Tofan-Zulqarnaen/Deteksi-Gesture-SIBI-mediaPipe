import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path dataset
data_dir = "gesture_dataset"  # Ganti dengan lokasi dataset kamu
categories = sorted(os.listdir(data_dir))  # Label berdasarkan nama folder
img_size = 64  # Ukuran gambar yang digunakan untuk training

# Load dataset
data = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load sebagai grayscale
        img = cv2.resize(img, (img_size, img_size))  # Resize gambar
        data.append(img)
        labels.append(label)

# Konversi ke numpy array
data = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0  # Normalisasi
labels = np.array(labels)

# One-hot encoding label
labels = to_categorical(labels, num_classes=len(categories))

# Split dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
epochs = 20
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=16)

# Simpan model
model.save("gesture_model.h5")

print("Model berhasil disimpan sebagai gesture_model.h5")
