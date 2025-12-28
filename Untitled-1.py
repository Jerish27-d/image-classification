# =====================================================
# 4-CLASS IMAGE CLASSIFICATION FROM 4 SEPARATE ZIP FILES
# =====================================================

# 1️⃣ Install required libraries
!pip install tensorflow matplotlib numpy pillow

# 2️⃣ Import libraries
import os
import zipfile
import random
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 3️⃣ Define paths
photo_folder = "photo"     # folder where zip files exist
dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# 4️⃣ ZIP files and class names (VERY IMPORTANT)
zip_classes = {
    "catvsdog": "catvsdog.zip",
    "cat": "cat.zip",
    "multi": "multi.zip",
    "trafic": "trafic.zip"
}

# 5️⃣ Create train/test folders
for cls in zip_classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

print("✅ Train/Test folders created")

# 6️⃣ Extract ZIP files
for cls, zip_name in zip_classes.items():
    zip_path = os.path.join(photo_folder, zip_name)
    extract_path = f"temp_{cls}"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("✅ ZIP files extracted")

# 7️⃣ Move images → Train (80%) and Test (20%)
for cls in zip_classes:
    image_files = []

    for root, _, files in os.walk(f"temp_{cls}"):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))

    random.shuffle(image_files)
    split = int(0.8 * len(image_files))

    train_images = image_files[:split]
    test_images = image_files[split:]

    for img in train_images:
        shutil.copy(img, os.path.join(train_dir, cls))

    for img in test_images:
        shutil.copy(img, os.path.join(test_dir, cls))

    print(f"✅ {cls}: {len(train_images)} train | {len(test_images)} test")

# 8️⃣ Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)

# 9️⃣ Build CNN Model (4 classes)
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(4, activation="softmax")   # 4 classes
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 🔟 Train the model
history = model.fit(
    train_set,
    epochs=10,
    validation_data=test_set
)

# 1️⃣1️⃣ Plot accuracy
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()

# 1️⃣2️⃣ Save model
model.save("4_class_image_classifier.h5")
print("✅ Model saved as 4_class_image_classifier.h5")
