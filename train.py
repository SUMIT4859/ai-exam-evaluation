# train.py

# Import ImageDataGenerator for image preprocessing and augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import CNN model creation function from model.py
from model import create_model


# -------------------------------
# 1️⃣ Image Preprocessing & Augmentation
# -------------------------------

# ImageDataGenerator is used to:
# - Normalize pixel values (rescale)
# - Split dataset into training and validation
# - Apply data augmentation (rotate, zoom, shift)
#   to improve model generalization

datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values (0-255 → 0-1)
    validation_split=0.2,    # 80% training, 20% validation
    rotation_range=10,       # Random rotation up to 10 degrees
    zoom_range=0.1,          # Random zoom
    width_shift_range=0.1,   # Horizontal shift
    height_shift_range=0.1   # Vertical shift
)


# -------------------------------
# 2️⃣ Load Training Data
# -------------------------------

# Load images from dataset folder
# It automatically assigns labels based on folder names

train_data = datagen.flow_from_directory(
    'dataset/',              # Main dataset folder
    target_size=(224,224),   # Resize images to 224x224
    color_mode='grayscale',  # Convert images to grayscale
    class_mode='categorical',# Multi-class classification
    subset='training'        # Select training portion (80%)
)


# -------------------------------
# 3️⃣ Load Validation Data
# -------------------------------

val_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(224,224),
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'      # Select validation portion (20%)
)


# -------------------------------
# 4️⃣ Create CNN Model
# -------------------------------

# Create model architecture defined in model.py
model = create_model()


# -------------------------------
# 5️⃣ Train the Model
# -------------------------------

# Train the model using training data
# Validate performance on validation data

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10   # Train for 20 iterations
)


# -------------------------------
# 6️⃣ Save Trained Model
# -------------------------------

# Save the trained model to file
# This file will be used later for prediction

model.save("answer_model.h5")

print("Model Training Complete & Saved!")