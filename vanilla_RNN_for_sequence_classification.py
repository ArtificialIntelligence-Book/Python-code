import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple RNN model for sequence classification
def create_rnn(input_shape=(100, 1), num_classes=2):
    model = models.Sequential()
    # SimpleRNN layer with 50 units
    model.add(layers.SimpleRNN(50, activation='tanh', input_shape=input_shape))
    # Output dense layer for classification
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    rnn_model = create_rnn()
    rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    rnn_model.summary()