"""
Simple Implementation of a Basic Transformer Model Using PyTorch

This example implements a minimal Transformer encoder model from scratch using PyTorch.
It includes the key components:
- Multi-Head Self-Attention
- Position-wise Feed-Forward Network
- Positional Encoding
- Encoder Layer
- Transformer Encoder (stack of encoder layers)

This basic Transformer can be used for sequence encoding tasks such as classification or
sequence-to-sequence models with additional decoder layers.

Note:
- This example focuses on the encoder part only.
- The implementation omits some advanced features like masking for simplicity.
- Designed for educational purposes to illustrate Transformer architecture.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant positional encoding matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)      # even indices
        pe[:, 1::2] = torch.cos(position * div_term)      # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

# === Multi-Head Self-Attention ===
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Define linear layers for query, key, value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Output linear layer
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.q_linear(x)  # (batch_size, seq_len, d_model)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)

        # Final linear layer
        output = self.out_linear(context)  # (batch_size, seq_len, d_model)

        return output

# === Position-wise Feedforward Network ===
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# === Encoder Layer ===
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x

# === Transformer Encoder ===
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input embedding (linear projection)
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        # Stacked encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_embedding(x) * math.sqrt(self.d_model)  # scale embedding
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return x  # shape: (batch_size, seq_len, d_model)

# === Example Usage ===
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    input_dim = 16
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 2
    max_seq_len = 100

    # Create dummy input: batch of sequences with input_dim features
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # Instantiate Transformer Encoder
    transformer = TransformerEncoder(input_dim, d_model, num_heads, d_ff, num_layers, max_seq_len)

    # Forward pass
    output = transformer(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)