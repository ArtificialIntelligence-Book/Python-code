import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN model
def create_cnn(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential()
    # First convolutional layer with 32 filters, 3x3 kernel
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # MaxPooling for downsampling spatial dimensions
    model.add(layers.MaxPooling2D((2, 2)))
    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten feature maps into 1D vector
    model.add(layers.Flatten())
    # Dense fully connected layer
    model.add(layers.Dense(64, activation='relu'))
    # Output layer with softmax for multi-class classification
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

if __name__ == "__main__":
    cnn_model = create_cnn()
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.summary()