# model.py

# Import required layers from TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def create_model():

    # Sequential means layers are added one after another
    model = Sequential([

        # 🔹 First Convolution Layer
        # 32 filters, 3x3 kernel size
        # ReLU activation helps introduce non-linearity
        # input_shape defines image size (224x224) and 1 channel (grayscale)
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),

        # 🔹 MaxPooling reduces image size
        # It keeps important features and reduces computation
        MaxPooling2D(2,2),

        # 🔹 Second Convolution Layer
        # 64 filters to learn more complex features
        Conv2D(64, (3,3), activation='relu'),

        # 🔹 Second MaxPooling
        MaxPooling2D(2,2),

        # 🔹 Flatten layer converts 2D feature maps into 1D vector
        Flatten(),

        # 🔹 Fully Connected Layer
        # 128 neurons for learning patterns
        Dense(128, activation='relu'),

        # 🔹 Dropout prevents overfitting
        # 50% neurons randomly disabled during training
        Dropout(0.5),

        # 🔹 Output Layer
        # 3 neurons because we have 3 classes:
        # correct, partial, wrong
        # Softmax gives probability for each class
        Dense(3, activation='softmax')
    ])


    # Compile the model
    # Adam optimizer updates weights automatically
    # Categorical crossentropy used for multi-class classification
    # Accuracy used to measure performance
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model