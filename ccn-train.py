import os
import numpy as np
import cv2
from random import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.data import Dataset

# Check for GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and will be used for training.")
else:
    print("No GPU detected. Training will use the CPU.")

# Constants
TRAIN_DIR = 'train'
IMG_SIZE = 128
IMAGE_CHANNELS = 3
FIRST_NUM_CHANNEL = 32
FILTER_SIZE = 3
LR = 0.0001
EPOCHS = 50
BATCH_SIZE = 32
MODEL_NAME = 'animals_cnn'
TRAIN_SPLIT_RATIO = 0.8

# Define classes and labels dynamically
def define_classes():
    all_classes = sorted(os.listdir(TRAIN_DIR))
    return all_classes, len(all_classes)

def define_labels(all_classes):
    return np.eye(len(all_classes))

# Load and preprocess images with normalization
def preprocess_image(filepath, img_size):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize pixel values
    return img

def create_data(all_classes, all_labels):
    data = []
    for label_index, class_name in enumerate(all_classes):
        folder_path = os.path.join(TRAIN_DIR, class_name)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = preprocess_image(img_path, IMG_SIZE)
            data.append([img, all_labels[label_index]])
    shuffle(data)
    return data

# Define classes and labels
all_classes, NUM_CLASSES = define_classes()
all_labels = define_labels(all_classes)

# Load dataset
data = create_data(all_classes, all_labels)

# Split into training and validation sets
split_idx = int(len(data) * TRAIN_SPLIT_RATIO)
train_data, val_data = data[:split_idx], data[split_idx:]

# Prepare input and labels
X_train = np.array([item[0] for item in train_data])
Y_train = np.array([item[1] for item in train_data])
X_val = np.array([item[0] for item in val_data])
Y_val = np.array([item[1] for item in val_data])

# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 30:
        return lr * 0.1
    return lr

# Build the model
model = Sequential([
    Conv2D(FIRST_NUM_CHANNEL, (FILTER_SIZE, FILTER_SIZE), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(FIRST_NUM_CHANNEL * 2, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(FIRST_NUM_CHANNEL * 4, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(FIRST_NUM_CHANNEL * 8, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(FIRST_NUM_CHANNEL * 16, activation='relu'),
    Dropout(0.5),  # Reduced dropout for better retention
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    LearningRateScheduler(lr_schedule)
]

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, Y_val),
    callbacks=callbacks,
    verbose=1
)

# Save the model in TensorFlow's SavedModel format
model.save(MODEL_NAME)
print(f"Model saved to {MODEL_NAME}")
