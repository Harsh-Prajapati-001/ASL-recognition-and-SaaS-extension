import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Constants
DATADIR = "asl_alphabet_train/asl_alphabet_train"  # Adjust path if needed
IMG_SIZE = 64

# Load categories (folder names A-Z)
CATEGORIES = ['A', 'B', 'C', 'D', 'E']

print("Loading data...")
data = []
labels = []

for idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATADIR, category)
    for img_name in os.listdir(folder_path):
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            try:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

print("Finished loading data.")

# Preprocessing
X = np.array(data) / 255.0
y = to_categorical(labels, num_classes=len(CATEGORIES))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(CATEGORIES), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
print("Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Save
model.save("asl_model.h5")
print("Model saved as asl_model.h5")
