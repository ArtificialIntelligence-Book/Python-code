import tensorflow as tf
from tensorflow.keras import layers, models

def create_autoencoder(input_dim=784, encoding_dim=32):
    # Input layer (flattened image)
    input_img = layers.Input(shape=(input_dim,))
    # Encoding layer - compressed representation
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # Decoding layer - reconstruct original input dimensions
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    # Autoencoder model maps input to reconstruction
    autoencoder = models.Model(input_img, decoded)
    
    # Encoder model for compressed representation
    encoder = models.Model(input_img, encoded)
    
    return autoencoder, encoder

if __name__ == "__main__":
    autoencoder, encoder = create_autoencoder()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()