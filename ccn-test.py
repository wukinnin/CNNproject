import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
IMG_SIZE = 128
IMAGE_CHANNELS = 3
MODEL_NAME = 'animals_cnn'  # Ensure this matches the training script
TRAIN_DIR = 'train'
TEST_IMG = 'cat146-test.jpg'  # Replace with the path to your test image

# Define classes dynamically from the train directory
def define_classes():
    all_classes = sorted(os.listdir(TRAIN_DIR))
    return all_classes, len(all_classes)

# Preprocess the test image
def preprocess_image(filepath, img_size):
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Error reading image at {filepath}. Check the file path and format.")
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Load model and make predictions
def classify_image(model_name, test_img_path):
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Model file '{model_name}' not found.")
    
    print("Loading model...")
    model = load_model(model_name)

    print("Defining classes...")
    all_classes, _ = define_classes()

    print(f"Preprocessing image '{test_img_path}'...")
    img_data = preprocess_image(test_img_path, IMG_SIZE)

    print("Classifying image...")
    predictions = model.predict(img_data)[0]
    results = {all_classes[i]: predictions[i] for i in range(len(all_classes))}
    return results

# Main execution
if __name__ == "__main__":
    try:
        results = classify_image(MODEL_NAME, TEST_IMG)
        print("\nClassification Results:")
        for class_name, probability in results.items():
            print(f"{class_name}: {probability:.4f}")
    except Exception as e:
        print(f"An error occurred: {e}")
