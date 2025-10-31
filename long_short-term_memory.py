import tensorflow as tf
from tensorflow.keras import layers, models

def create_lstm(input_shape=(100, 1), num_classes=2):
    model = models.Sequential()
    # LSTM layer with 50 units
    model.add(layers.LSTM(50, input_shape=input_shape))
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    lstm_model = create_lstm()
    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm_model.summary()