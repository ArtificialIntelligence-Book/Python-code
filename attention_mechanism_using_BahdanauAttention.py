import tensorflow as tf
from tensorflow.keras import layers

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)  # for hidden state
        self.W2 = layers.Dense(units)  # for encoder output
        self.V = layers.Dense(1)       # for scalar score
    
    def call(self, query, values):
        # query shape: (batch_size, hidden size)
        # values shape: (batch_size, seq_len, hidden size)
        
        # Expand query to seq_len dimension for addition
        query_with_time_axis = tf.expand_dims(query, 1)  # (batch_size, 1, hidden size)
        
        # Calculate score = V(tanh(W1(query) + W2(values)))
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))  # (batch_size, seq_len, 1)
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, seq_len, 1)
        
        # Context vector as weighted sum of values
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, hidden size)
        
        return context_vector, attention_weights


# Example usage: simple test
if __name__ == "__main__":
    attention_layer = BahdanauAttention(10)

    batch_size = 2
    seq_len = 5
    hidden_size = 20

    # Random query and values tensors
    query = tf.random.normal(shape=(batch_size, hidden_size))
    values = tf.random.normal(shape=(batch_size, seq_len, hidden_size))

    context_vector, attn_weights = attention_layer(query, values)
    print("Context vector shape:", context_vector.shape)
    print("Attention weights shape:", attn_weights.shape)