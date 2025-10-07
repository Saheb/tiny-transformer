from typing import Dict, List
import jax.numpy as jnp
import jax
import os
from data import load_dataset_and_vocab, tokenize_dataset, generate_targets, get_data_iterator
import optax
import matplotlib.pyplot as plt
from flax.serialization import to_bytes

from typing import NamedTuple
import optax

class ReduceLROnPlateauState(NamedTuple):
    """Holds the state for the ReduceLROnPlateau scheduler."""
    learning_rate: float
    best_loss: float
    patience: int

def update_lr_on_plateau(state: ReduceLROnPlateauState, val_loss: float, factor: float = 0.5, patience_limit: int = 10, min_lr: float = 1e-6):
    """
    Updates the scheduler state based on the latest validation loss.
    
    Returns:
        A new scheduler state with potentially updated learning rate and patience.
    """
    new_lr = state.learning_rate
    
    # If validation loss improved, reset patience and update best_loss
    if val_loss < state.best_loss:
        new_patience = 0
        new_best_loss = val_loss
    # If it didn't improve, increase patience
    else:
        new_patience = state.patience + 1
        new_best_loss = state.best_loss

    # If patience limit is reached, reduce the learning rate
    if new_patience >= patience_limit:
        print(f"Validation loss plateaued. Reducing LR from {state.learning_rate:.6f} to {max(state.learning_rate * factor, min_lr):.6f}")
        new_lr = max(state.learning_rate * factor, min_lr)
        new_patience = 0  # Reset patience after reducing LR

    return ReduceLROnPlateauState(learning_rate=new_lr, best_loss=new_best_loss, patience=new_patience)

def text_to_token_ids(sentences: List[str], vocab: Dict[str, int], max_len: int = 32) -> jnp.ndarray:
    """Converts a batch of sentences to token IDs.
    
    Args:
        sentences: List of sentences (batch_size sentences).
        vocab: Dictionary mapping tokens to integer IDs.
        max_len: Maximum sequence length (default: 32).
    
    Returns:
        jnp.Array: Shape (batch_size, max_len), dtype int32, padded token IDs.
    """

    def sentence_to_token_ids(sentence: str) -> jnp.ndarray:

        words = filter(None, map(str.strip, sentence.split(' ')))

        tokens = jnp.array(list(map(lambda word: vocab[word], words)), dtype=jnp.int32)
       # print(tokens)
        if len(tokens) < max_len:
            return jnp.pad(tokens, ((0, max_len - len(tokens)),), constant_values=0)
        else:
            return tokens[:max_len]

    data = jnp.stack(list(map(sentence_to_token_ids, sentences)))
    return jnp.array(data, jnp.int32)


def token_embeddings(token_ids: jnp.ndarray, embed_matrix: jnp.ndarray, d_model: int = 64) -> jnp.ndarray:
    """Converts token IDs to dense vectors capturing meaning, learned during training.
    
    Args:
        token_ids: Shape (batch_size, max_len), dtype int32, token IDs from text_to_token_ids.
        embed_matrix: Shape (vocab_size, d_model), dtype float32, embedding weights.
    
    Returns:
        jnp.ndarray: Shape (batch_size, max_len, d_model), dtype float32, token embeddings.
    """
    return embed_matrix[token_ids]


def positional_embeddings(max_len: int = 32, d_model: int = 64) -> jnp.ndarray:
    """Generates positional embeddings for token positions using sinusoidal functions.
    
    Args:
        max_len: Maximum sequence length (default: 32).
        d_model: Embedding dimension (default: 64).
    
    Returns:
        jnp.ndarray: Shape (1, max_len, d_model), dtype float32, for batch compatibility.
    """
    
    # pe = jnp.zeros((max_len, d_model))
    # for pos in range(max_len):
    #     for i in range(int(d_model / 2) - 1):
    #         arg = pos / jnp.pow(10000, ((2*i) / d_model))
    #         pe[pos][2*i] = jnp.sin(arg)
    #         pe[pos][2*i + 1] = jnp.cos(arg)
    
    # vectorized version below:

    # Generate array of position indices from 0 to max_len-1
    # Shape: (max_len,) e.g., (32,) for max_len=32
    pos = jnp.arange(max_len)

    # Generate array of dimension indices from 0 to d_model//2-1
    # Shape: (d_model//2,) e.g., (32,) for d_model=64
    i = jnp.arange(d_model // 2)

    # Compute the argument for sine and cosine functions
    # pos[:, None] adds a new axis: (max_len, 1)
    # 2*i[None, :] broadcasts i: (1, d_model//2)
    # jnp.pow(10000, 2*i / d_model) creates frequency scaling: (1, d_model//2)
    # Division broadcasts to (max_len, d_model//2) e.g., (32, 32)
    arg = pos[:, None] / jnp.pow(10000, 2 * i[None, :] / d_model)

    # Apply sine function to get values for even indices
    # Shape: (max_len, d_model//2) e.g., (32, 32)
    sine_vals = jnp.sin(arg)

    # Apply cosine function to get values for odd indices
    # Shape: (max_len, d_model//2) e.g., (32, 32)
    cos_vals = jnp.cos(arg)
    
    # Create indices for even columns (0, 2, 4, ..., d_model-2)
    # Shape: (d_model//2,) e.g., [0, 2, 4, ..., 62] for d_model=64
    even_indices = jnp.arange(d_model // 2) * 2

    # Create indices for odd columns (1, 3, 5, ..., d_model-1)
    # Shape: (d_model//2,) e.g., [1, 3, 5, ..., 63] for d_model=64
    odd_indices = jnp.arange(d_model // 2) * 2 + 1

    # Initialize positional embedding array with zeros
    # Shape: (max_len, d_model) e.g., (32, 64)
    pe = jnp.zeros((max_len, d_model))

    # Update even-indexed columns with sine values using functional indexing
    # .at[:, even_indices].set() creates a new array with sine_vals at even positions
    # Shape remains (max_len, d_model)
    pe = pe.at[:, even_indices].set(sine_vals)

    # Update odd-indexed columns with cosine values using functional indexing
    # .at[:, odd_indices].set() creates a new array with cos_vals at odd positions
    # Shape remains (max_len, d_model)
    pe = pe.at[:, odd_indices].set(cos_vals)

    # Add a batch dimension for compatibility with batched inputs
    # Shape: (max_len, d_model) -> (1, max_len, d_model) e.g., (1, 32, 64)
    pe = jnp.expand_dims(pe, 0)

    return pe

def combine_embeddings(token_emb: jnp.ndarray, pos_emb: jnp.ndarray) -> jnp.ndarray:
    """Combines token and positional embeddings.
    Args:
        token_emb: Shape (batch_size, max_len, d_model), dtype float32.
        pos_emb: Shape (1, max_len, d_model), dtype float32.
    Returns:
        jnp.ndarray: Shape (batch_size, max_len, d_model), dtype float32.
    """
    return jnp.add(token_emb, pos_emb)

def layer_norm(x, axis=-1, eps=1e-6, scale=None, offset=None):  # Increased eps from 1e-5
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.var(x, axis=axis, keepdims=True)
    scale = scale if scale is not None else jnp.ones_like(mean)
    offset = offset if offset is not None else jnp.zeros_like(mean)
    return (x - mean) / jnp.sqrt(jnp.maximum(var, eps)) * scale + offset  # Use maximum to avoid zero var

def encoder_layer_(emb: jnp.ndarray, params: dict, key, n_heads: int = 8, d_model: int = 64, d_ff: int = 256, dropout_rate: float = 0.0, training: bool = False) -> jnp.ndarray:
    """Applies an encoder layer with multi-head self-attention and feed-forward network.
    
    Args:
        emb: Input embeddings, shape (batch_size, max_len, d_model), dtype float32.
        n_heads: Number of attention heads (default: 8).
        d_model: Dimension of the model/embeddings (default: 64).
        d_ff: Dimension of the feed-forward layer (default: 256).
    
    Returns:
        jnp.ndarray: Output embeddings, shape (batch_size, max_len, d_model), dtype float32.
    """
    batch_size, max_len, _ = emb.shape

    W_q = params['self_attention']['W_q']
    W_k = params['self_attention']['W_k']
    W_v = params['self_attention']['W_v']
    W_o = params['self_attention']['W_o']
    b_q = params['self_attention']['b_q']
    b_k = params['self_attention']['b_k']
    b_v = params['self_attention']['b_v']
    b_o = params['self_attention']['b_o']

    # Step 1: Multi-Head Self-Attention
    # - Project emb to queries, keys, values: (batch_size, max_len, d_model) -> (batch_size, max_len, d_model)
    # - Split into n_heads: (batch_size, max_len, d_model) -> (batch_size, max_len, n_heads, d_model/n_heads)
    # - Compute attention scores and weighted sum
    
    # Project and split into heads
    Q = jnp.dot(emb, W_q) + b_q
    K = jnp.dot(emb, W_k) + b_k
    V = jnp.dot(emb, W_v) + b_v

    Q = Q.reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)

    E = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_model // n_heads) # (16, 8, 32, 32)
    pad_mask = (jnp.sum(emb, axis=-1) != 0)[:, None, None, :]  # Shape: (batch_size, 1, 1, max_len)
    E = E + (1.0 - pad_mask) * -1e9  # Mask padded positions with -inf
    # print("E shape:", E.shape)
    A = jax.nn.softmax(E, axis=-1)  # (16, 8, 32, 32)
    # print("A shape:", A.shape)
    Y = jnp.matmul(A, V)  # (16, 8, 32, 8)
    # print("Y shape:", Y.shape)

    # Concatenate and project output
    Y = Y.transpose(0, 2, 1, 3).reshape(batch_size, max_len, d_model)
    attention_out = jnp.dot(Y, W_o) + b_o

    residual = emb + attention_out
    scale = jnp.ones((d_model,))  # Initial scale
    offset = jnp.zeros((d_model,))  # Initial offset
    norm1 = layer_norm(residual, axis=-1, scale=scale, offset=offset)  # (16, 32, 64)

    # Feed-Forward Network
    W1 = params['feed_forward']['W_ff1']
    W2 = params['feed_forward']['W_ff2']
    b1 = params['feed_forward']['b_ff1']
    b2 = params['feed_forward']['b_ff2']
    ffn_out = jnp.dot(jnp.maximum(0, jnp.dot(norm1, W1) + b1), W2) + b2
    
    key, subkey = jax.random.split(key)

    if training:
        key, subkey = jax.random.split(key)
        dropout_mask = jax.random.bernoulli(subkey, 1 - dropout_rate, ffn_out.shape)
        ffn_out = jnp.where(dropout_mask, ffn_out / (1 - dropout_rate), 0)  # Apply scaling

    # Residual Connection and Layer Norm 2
    residual2 = norm1 + ffn_out
    norm2 = layer_norm(residual2, axis=-1, scale=scale, offset=offset)  # (16, 32, 64)
    
    return norm2

def encoder_layer(emb: jnp.ndarray, params: dict, key, n_heads: int = 8, d_model: int = 256, d_ff: int = 1024, dropout_rate: float = 0.0, training: bool = False) -> jnp.ndarray:
    batch_size, max_len, _ = emb.shape

    ## PRE-LN FIX 1: Normalize BEFORE self-attention
    norm1 = layer_norm(emb)

    # Multi-Head Self-Attention
    W_q, W_k, W_v, W_o = params['self_attention']['W_q'], params['self_attention']['W_k'], params['self_attention']['W_v'], params['self_attention']['W_o']
    b_q, b_k, b_v, b_o = params['self_attention']['b_q'], params['self_attention']['b_k'], params['self_attention']['b_v'], params['self_attention']['b_o']

    Q = (jnp.dot(norm1, W_q) + b_q).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    K = (jnp.dot(norm1, W_k) + b_k).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    V = (jnp.dot(norm1, W_v) + b_v).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)

    E = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_model // n_heads)
    pad_mask = (jnp.sum(emb, axis=-1) != 0)[:, None, None, :]
    E = jnp.where(pad_mask == 0, -1e9, E)
    A = jax.nn.softmax(E, axis=-1)
    
    Y = jnp.matmul(A, V).transpose(0, 2, 1, 3).reshape(batch_size, max_len, d_model)
    attention_out = jnp.dot(Y, W_o) + b_o
    
    ## PRE-LN FIX 2: Add residual connection AFTER sublayer
    residual = emb + attention_out

    ## PRE-LN FIX 3: Normalize BEFORE feed-forward
    norm2 = layer_norm(residual)
    
    # Feed-Forward Network
    W1, W2 = params['feed_forward']['W_ff1'], params['feed_forward']['W_ff2']
    b1, b2 = params['feed_forward']['b_ff1'], params['feed_forward']['b_ff2']
    ffn_out = jnp.dot(jax.nn.relu(jnp.dot(norm2, W1) + b1), W2) + b2
    
    key, subkey = jax.random.split(key)
    if training:
        dropout_mask = jax.random.bernoulli(subkey, 1 - dropout_rate, ffn_out.shape)
        ffn_out = jnp.where(dropout_mask, ffn_out / (1 - dropout_rate), 0)

    ## PRE-LN FIX 4: Add final residual connection
    final_out = residual + ffn_out
    
    return final_out
def decoder_layer(dec_emb: jnp.ndarray, enc_output: jnp.ndarray, params: dict, key, n_heads: int = 8, d_model: int = 256, d_ff: int = 1024, dropout_rate: float = 0.0, training: bool = False) -> jnp.ndarray:
    batch_size, max_len, _ = dec_emb.shape

    # 1. Masked Multi-Head Self-Attention (with Pre-LN)
    norm1 = layer_norm(dec_emb)
    W_q, W_k, W_v, W_o = params['self_attention']['W_q'], params['self_attention']['W_k'], params['self_attention']['W_v'], params['self_attention']['W_o']
    b_q, b_k, b_v, b_o = params['self_attention']['b_q'], params['self_attention']['b_k'], params['self_attention']['b_v'], params['self_attention']['b_o']

    Q = (jnp.dot(norm1, W_q) + b_q).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    K = (jnp.dot(norm1, W_k) + b_k).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    V = (jnp.dot(norm1, W_v) + b_v).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)

    E = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_model // n_heads)
    pad_mask = (jnp.sum(dec_emb, axis=-1) != 0)[:, None, None, :]
    look_ahead_mask = jnp.tril(jnp.ones((1, 1, max_len, max_len)))
    mask = jnp.logical_and(pad_mask, look_ahead_mask)
    E = jnp.where(mask == 0, -1e9, E)
    A = jax.nn.softmax(E, axis=-1)

    Y = jnp.matmul(A, V).transpose(0, 2, 1, 3).reshape(batch_size, max_len, d_model)
    self_attention_out = jnp.dot(Y, W_o) + b_o
    residual1 = dec_emb + self_attention_out

    # 2. Cross-Attention (with Pre-LN)
    norm2 = layer_norm(residual1)
    W_cq, W_ck, W_cv, W_co = params['cross_attention']['W_q'], params['cross_attention']['W_k'], params['cross_attention']['W_v'], params['cross_attention']['W_o']
    b_cq, b_ck, b_cv, b_co = params['cross_attention']['b_q'], params['cross_attention']['b_k'], params['cross_attention']['b_v'], params['cross_attention']['b_o']

    Q = (jnp.dot(norm2, W_cq) + b_cq).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    K = (jnp.dot(enc_output, W_ck) + b_ck).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    V = (jnp.dot(enc_output, W_cv) + b_cv).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)

    enc_pad_mask = (jnp.sum(enc_output, axis=-1) != 0)[:, None, None, :]
    E = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_model // n_heads)
    E = jnp.where(enc_pad_mask == 0, -1e9, E)
    A = jax.nn.softmax(E, axis=-1)

    Y = jnp.matmul(A, V).transpose(0, 2, 1, 3).reshape(batch_size, max_len, d_model)
    cross_attention_out = jnp.dot(Y, W_co) + b_co
    residual2 = residual1 + cross_attention_out

    # 3. Feed-Forward Network (with Pre-LN)
    norm3 = layer_norm(residual2)
    W1, W2 = params['feed_forward']['W_ff1'], params['feed_forward']['W_ff2']
    b1, b2 = params['feed_forward']['b_ff1'], params['feed_forward']['b_ff2']
    ffn_out = jnp.dot(jax.nn.relu(jnp.dot(norm3, W1) + b1), W2) + b2

    key, subkey = jax.random.split(key)
    if training:
        dropout_mask = jax.random.bernoulli(subkey, 1 - dropout_rate, ffn_out.shape)
        ffn_out = jnp.where(dropout_mask, ffn_out / (1 - dropout_rate), 0)
        
    final_out = residual2 + ffn_out

    return final_out

def decoder_layer_(dec_emb: jnp.ndarray, enc_output: jnp.ndarray, params: dict, key, n_heads: int = 8, d_model: int = 64, d_ff: int = 256, dropout_rate: float = 0.0, training: bool = False) -> jnp.ndarray:
    """
    Single layer of the Transformer decoder.
    
    Args:
        dec_emb: Decoder input embeddings, shape (batch_size, max_len, d_model)
        enc_output: Encoder output, shape (batch_size, max_len, d_model)
        params: Dictionary of layer parameters
        key: PRNG key for dropout
        n_heads: Number of attention heads
        d_model: Model dimension
        d_ff: Feed-forward network dimension
        dropout_rate: Dropout probability
        training: Boolean for training mode
    
    Returns:
        Output of the decoder layer, shape (batch_size, max_len, d_model)
    """
    batch_size, max_len, _ = dec_emb.shape
    # print("dec_emb entering decoder_layer:", dec_emb.shape, "min:", jnp.min(dec_emb), "max:", jnp.max(dec_emb), "NaN:", jnp.any(jnp.isnan(dec_emb)))

    check_params_for_nan(params)

    # Self-Attention
    W_q = params['self_attention']['W_q']
    W_k = params['self_attention']['W_k']
    W_v = params['self_attention']['W_v']
    W_o = params['self_attention']['W_o']
    b_q = params['self_attention']['b_q']
    b_k = params['self_attention']['b_k']
    b_v = params['self_attention']['b_v']
    b_o = params['self_attention']['b_o']
    if jnp.any(jnp.isnan(W_q)) or jnp.any(jnp.isnan(b_q)):
        print("NaN in self-attention weights/biases!")

    Q = jnp.dot(dec_emb, W_q) + b_q
    # print("Q shape:", Q.shape, "min:", jnp.min(Q), "max:", jnp.max(Q), "NaN:", jnp.any(jnp.isnan(Q)))
    K = jnp.dot(dec_emb, W_k) + b_k
    # print("K shape:", K.shape, "min:", jnp.min(K), "max:", jnp.max(K), "NaN:", jnp.any(jnp.isnan(K)))
    V = jnp.dot(dec_emb, W_v) + b_v
    # print("V shape:", V.shape, "min:", jnp.min(V), "max:", jnp.max(V), "NaN:", jnp.any(jnp.isnan(V)))

    Q = Q.reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)

    pad_mask = (jnp.sum(dec_emb, axis=-1) != 0)[:, None, None, :]
    mask = jnp.tril(jnp.ones((1, 1, max_len, max_len))) * pad_mask

    E = jnp.matmul(Q, K.transpose(0,1,3,2)) / jnp.sqrt(d_model // n_heads)
    if mask is not None:
        E = jnp.where(mask, E, -1e9)
    A = jax.nn.softmax(E, axis=-1)

    Y = jnp.matmul(A, V)
    Y = Y.transpose(0, 2, 1, 3).reshape(batch_size, max_len, d_model)
    self_attention_out = jnp.dot(Y, W_o) + b_o
    # print("self_attention_out shape:", self_attention_out.shape, "min:", jnp.min(self_attention_out), "max:", jnp.max(self_attention_out), "NaN:", jnp.any(jnp.isnan(self_attention_out)))

    # Dropout on Self-Attention
    key, subkey = jax.random.split(key)
    dropout_mask = jax.random.bernoulli(subkey, 1 - dropout_rate, self_attention_out.shape)
    self_attention_out = jnp.where(training, self_attention_out * dropout_mask / (1 - dropout_rate), self_attention_out)
    # print("self_attention_out after dropout shape:", self_attention_out.shape, "min:", jnp.min(self_attention_out), "max:", jnp.max(self_attention_out), "NaN:", jnp.any(jnp.isnan(self_attention_out)))

    # Cross-Attention
    W_cq = params['cross_attention']['W_q']
    W_ck = params['cross_attention']['W_k']
    W_cv = params['cross_attention']['W_v']
    W_co = params['cross_attention']['W_o']
    b_cq = params['cross_attention']['b_q']
    b_ck = params['cross_attention']['b_k']
    b_cv = params['cross_attention']['b_v']
    b_co = params['cross_attention']['b_o']
    if jnp.any(jnp.isnan(W_cq)) or jnp.any(jnp.isnan(b_cq)):
        print("NaN in cross-attention weights/biases!")

    Q = jnp.dot(self_attention_out, W_cq) + b_cq
    # print("Q (cross) shape:", Q.shape, "min:", jnp.min(Q), "max:", jnp.max(Q), "NaN:", jnp.any(jnp.isnan(Q)))
    K = jnp.dot(enc_output, W_ck) + b_ck
    # print("K (cross) shape:", K.shape, "min:", jnp.min(K), "max:", jnp.max(K), "NaN:", jnp.any(jnp.isnan(K)))
    V = jnp.dot(enc_output, W_cv) + b_cv
    # print("V (cross) shape:", V.shape, "min:", jnp.min(V), "max:", jnp.max(V), "NaN:", jnp.any(jnp.isnan(V)))

    Q = Q.reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)

    enc_pad_mask = (jnp.sum(enc_output, axis=-1) != 0)[:, None, None, :]
    E = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_model // n_heads)
    # print("E (cross) shape:", E.shape, "min:", jnp.min(E), "max:", jnp.max(E), "NaN:", jnp.any(jnp.isnan(E)))
    E = E + (1.0 - enc_pad_mask) * -1e9
    A = jax.nn.softmax(E, axis=-1, where=~jnp.isnan(E))
    # print("A (cross) shape:", A.shape, "min:", jnp.min(A), "max:", jnp.max(A), "NaN:", jnp.any(jnp.isnan(A)))

    Y = jnp.matmul(A, V)
    Y = Y.transpose(0, 2, 1, 3).reshape(batch_size, max_len, d_model)
    cross_attention_out = jnp.dot(Y, W_co) + b_co
    # print("cross_attention_out shape:", cross_attention_out.shape, "min:", jnp.min(cross_attention_out), "max:", jnp.max(cross_attention_out), "NaN:", jnp.any(jnp.isnan(cross_attention_out)))

    # Dropout on Cross-Attention
    key, subkey = jax.random.split(key)
    dropout_mask = jax.random.bernoulli(subkey, 1 - dropout_rate, cross_attention_out.shape)
    cross_attention_out = jnp.where(training, cross_attention_out * dropout_mask / (1 - dropout_rate), cross_attention_out)
    # print("cross_attention_out after dropout shape:", cross_attention_out.shape, "min:", jnp.min(cross_attention_out), "max:", jnp.max(cross_attention_out), "NaN:", jnp.any(jnp.isnan(cross_attention_out)))

    # Feed-Forward Network
    W1 = params['feed_forward']['W_ff1']
    W2 = params['feed_forward']['W_ff2']
    b1 = params['feed_forward']['b_ff1']
    b2 = params['feed_forward']['b_ff2']
    if jnp.any(jnp.isnan(W1)) or jnp.any(jnp.isnan(b1)):
        print("NaN in FFN weights/biases!")

    ffn_out = jnp.dot(jnp.maximum(0, jnp.dot(cross_attention_out, W1) + b1), W2) + b2
    # print("ffn_out shape:", ffn_out.shape, "min:", jnp.min(ffn_out), "max:", jnp.max(ffn_out), "NaN:", jnp.any(jnp.isnan(ffn_out)))

    # Dropout on FFN
    key, subkey = jax.random.split(key)
    dropout_mask = jax.random.bernoulli(subkey, 1 - dropout_rate, ffn_out.shape)
    ffn_out = jnp.where(training, ffn_out * dropout_mask / (1 - dropout_rate), ffn_out)
    # print("ffn_out after dropout shape:", ffn_out.shape, "min:", jnp.min(ffn_out), "max:", jnp.max(ffn_out), "NaN:", jnp.any(jnp.isnan(ffn_out)))

    # Residual Connections and Normalization
    residual1 = dec_emb + self_attention_out
    # print("residual1 shape:", residual1.shape, "min:", jnp.min(residual1), "max:", jnp.max(residual1), "NaN:", jnp.any(jnp.isnan(residual1)))
    norm1 = layer_norm(residual1, axis=-1)
    # print("norm1 shape:", norm1.shape, "min:", jnp.min(norm1), "max:", jnp.max(norm1), "NaN:", jnp.any(jnp.isnan(norm1)))

    residual2 = norm1 + cross_attention_out
    # print("residual2 shape:", residual2.shape, "min:", jnp.min(residual2), "max:", jnp.max(residual2), "NaN:", jnp.any(jnp.isnan(residual2)))
    norm2 = layer_norm(residual2, axis=-1)
    # print("norm2 shape:", norm2.shape, "min:", jnp.min(norm2), "max:", jnp.max(norm2), "NaN:", jnp.any(jnp.isnan(norm2)))

    residual3 = norm2 + ffn_out
    # print("residual3 shape:", residual3.shape, "min:", jnp.min(residual3), "max:", jnp.max(residual3), "NaN:", jnp.any(jnp.isnan(residual3)))
    norm3 = layer_norm(residual3, axis=-1)
    # print("norm3 shape:", norm3.shape, "min:", jnp.min(norm3), "max:", jnp.max(norm3), "NaN:", jnp.any(jnp.isnan(norm3)))

    # print("dec_emb exiting decoder_layer:", norm3.shape, "min:", jnp.min(norm3), "max:", jnp.max(norm3), "NaN:", jnp.any(jnp.isnan(norm3)))
    return norm3

def output_linerity(dec_output: jnp.ndarray, params) -> jnp.ndarray:
    assert dec_output.shape[-1] == params['W_out'].shape[0]
    prediction = jnp.dot(dec_output, params['W_out']) + params['b_out'] # shape 16, 32, 10k
    predicted_probs = jax.nn.softmax(prediction, axis=-1)
    return predicted_probs

def transformer_decoder(dec_emb: jnp.ndarray, enc_output: jnp.ndarray, params: dict, keys, n_layers: int = 6, d_model=64, **kwargs) -> jnp.ndarray:
    # print("dec_emb entering transformer_decoder:", dec_emb.shape, "min:", jnp.min(dec_emb), "max:", jnp.max(dec_emb), "NaN:", jnp.any(jnp.isnan(dec_emb)))
    x = dec_emb
    layer_keys = keys
    for i in range(n_layers):
        layer_key = layer_keys[i]
        x = decoder_layer(x, enc_output, params['layers'][i], key=layer_key, d_model=d_model, **kwargs)
        # print(f"dec_emb after layer {i}:", x.shape, "min:", jnp.min(x), "max:", jnp.max(x), "NaN:", jnp.any(jnp.isnan(x)))
    return x

def transformer_encoder(emb: jnp.ndarray, params: dict, keys, n_layers: int = 2, d_model=64, **kwargs) -> jnp.ndarray:
    """Stacks multiple encoder layers to form the transformer encoder.
    
    Args:
        emb: Input embeddings, shape (batch_size, max_len, d_model), dtype float32.
        n_layers: Number of encoder layers to stack (default: 2).
        **kwargs: Arguments for encoder_layer (n_heads, d_model, d_ff, training).
    
    Returns:
        jnp.ndarray: Output embeddings, shape (batch_size, max_len, d_model), dtype float32.
    """
    x = emb
    layer_keys = keys  # Use the pre-split keys directly
    for i in range(n_layers):
        layer_key = layer_keys[i]  # Unique key per layer
        x = encoder_layer(x, params['layers'][i], key=layer_key, d_model=d_model, **kwargs)  # Pass key and other kwargs
    return x

def forward(params, enc_input, dec_input, vocab_size=10000, dropout_rate=0.0, key=jax.random.PRNGKey(0), training=False, d_model=64):
    try:
        # Split key into 12 subkeys (6 for encoder, 6 for decoder)
        keys = jax.random.split(key, 12)
        enc_keys = keys[:6]
        dec_keys = keys[6:]

        # Embedding for encoder
        enc_emb = jnp.take(params['embedding']['W_emb'], enc_input, axis=0, mode='clip')
        if jnp.any(jnp.isnan(enc_emb)):
            print("NaN in enc_emb after lookup!")
        enc_emb = enc_emb + positional_embeddings(max_len=enc_input.shape[1], d_model=d_model)

        # Embedding for decoder
        # print("dec_input shape:", dec_input.shape, "min:", jnp.min(dec_input), "max:", jnp.max(dec_input), "NaN:", jnp.any(jnp.isnan(dec_input)))
        dec_emb = jnp.take(params['embedding']['W_emb'], dec_input, axis=0, mode='clip')
        # print("dec_emb shape after lookup:", dec_emb.shape, "min:", jnp.min(dec_emb), "max:", jnp.max(dec_emb), "NaN:", jnp.any(jnp.isnan(dec_emb)))
        if jnp.any(jnp.isnan(dec_emb)):
            print("NaN in dec_emb after lookup! W_emb NaN:", jnp.any(jnp.isnan(params['embedding']['W_emb'])))
        pos_emb = positional_embeddings(max_len=dec_input.shape[1], d_model=d_model)
        # print("pos_emb shape:", pos_emb.shape, "min:", jnp.min(pos_emb), "max:", jnp.max(pos_emb), "NaN:", jnp.any(jnp.isnan(pos_emb)))
        dec_emb = dec_emb + pos_emb
        # print("dec_emb shape after pos_emb:", dec_emb.shape, "min:", jnp.min(dec_emb), "max:", jnp.max(dec_emb), "NaN:", jnp.any(jnp.isnan(dec_emb)))
        if jnp.any(jnp.isnan(dec_emb)):
            print("NaN in dec_emb after positional embedding!")
        dec_emb = jnp.where(jnp.isnan(dec_emb), 0.0, dec_emb)
        # print("dec_emb exiting forward:", dec_emb.shape, "min:", jnp.min(dec_emb), "max:", jnp.max(dec_emb), "NaN:", jnp.any(jnp.isnan(dec_emb)))
        
        # Encoder and decoder pass
        encoded = transformer_encoder(enc_emb, params['encoder'], enc_keys, n_layers=6, d_model=d_model, training=training, dropout_rate=dropout_rate)
        decoded = transformer_decoder(dec_emb, encoded, params['decoder'], dec_keys, n_layers=6, d_model=d_model, training=training, dropout_rate=dropout_rate)

        if jnp.any(jnp.isnan(decoded)):
            print("Warning: NaN in decoded!")
        logits = jnp.matmul(decoded, params['output']['W_out']) + params['output']['b_out']
        if jnp.any(jnp.isnan(logits)):
            print("Warning: NaN detected in logits!")
        predictions = jnp.argmax(jax.nn.softmax(logits, axis=-1), axis=-1)
        return logits, predictions
    except Exception as e:
        print(f"Error in forward: {e}")
        raise

def loss_fn(params, inputs, targets, batch_size, vocab, vocab_size=10000, dropout_rate=0.0, key=jax.random.PRNGKey(0), training=False, d_model=64):
    """
    Corrected loss function.
    - Takes original targets.
    - Creates the shifted decoder input internally.
    - Computes loss on the full sequence, using a mask for padding.
    """
    current_batch_size = targets.shape[0]
    sos_id = vocab['<SOS>']
    
    # 1. Create the decoder input by shifting the original targets right.
    dec_input = jnp.concatenate([jnp.full((current_batch_size, 1), sos_id), targets[:, :-1]], axis=1)

    # 2. Get the model's predictions (logits).
    logits, _ = forward(params, inputs, dec_input, vocab_size=vocab_size, dropout_rate=dropout_rate, key=key, training=training, d_model=d_model)

    # 3. Compute loss between the full logits and the original targets.
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    # 4. Create a mask to ignore padding tokens (usually token ID 0) in the loss calculation.
    mask = (targets != 0)
    
    return jnp.sum(loss * mask) / jnp.sum(mask)

def init_params(key, vocab_size=10000, d_model=64, d_ff=256, n_heads=8, n_layers=6, dropout_rate=0.0):
    """Initialize model parameters with Xavier normal initialization."""
    # Split into enough keys for all parameters
    total_keys = 2 + n_layers * (6 + 10)  # 2 for embedding/output, 6 per encoder layer, 10 per decoder layer
    total_keys = total_keys + 10  # Add 10 extra keys as buffer
    keys = jax.random.split(key, total_keys)
    key_idx = 0

    # Embedding params
    embedding_params = {
        'W_emb': jax.nn.initializers.xavier_normal()(keys[key_idx], (vocab_size, d_model))
    }
    if jnp.any(jnp.isnan(embedding_params['W_emb'])):
        print("Warning: NaN in W_emb initialization! Reinitializing...")
        embedding_params['W_emb'] = jax.nn.initializers.xavier_normal()(jax.random.PRNGKey(key_idx + 1), (vocab_size, d_model))
    # print("W_emb shape:", embedding_params['W_emb'].shape, "NaN:", jnp.any(jnp.isnan(embedding_params['W_emb'])))
    key_idx += 1

    # Encoder params
    encoder_params = {
        'layers': [
            dict(
                self_attention=dict(
                    W_q=jax.nn.initializers.xavier_normal()(keys[key_idx], (d_model, d_model)),
                    W_k=jax.nn.initializers.xavier_normal()(keys[key_idx + 1], (d_model, d_model)),
                    W_v=jax.nn.initializers.xavier_normal()(keys[key_idx + 2], (d_model, d_model)),
                    W_o=jax.nn.initializers.xavier_normal()(keys[key_idx + 3], (d_model, d_model)),
                    b_q=jnp.zeros((d_model,)),
                    b_k=jnp.zeros((d_model,)),
                    b_v=jnp.zeros((d_model,)),
                    b_o=jnp.zeros((d_model,)),
                ),
                feed_forward=dict(
                    W_ff1=jax.nn.initializers.he_normal()(keys[key_idx + 4], (d_model, d_ff)),
                    W_ff2=jax.nn.initializers.xavier_normal()(keys[key_idx + 5], (d_ff, d_model)),
                    b_ff1=jnp.zeros((d_ff,)),
                    b_ff2=jnp.zeros((d_model,)),
                )
            ) for i in range(n_layers)
        ]
    }
    key_idx += 6 * n_layers  # Move index past all encoder layers

    # Decoder params
    decoder_params = {
        'layers': [
            dict(
                self_attention=dict(
                    W_q=jax.nn.initializers.xavier_normal()(keys[key_idx], (d_model, d_model)),
                    W_k=jax.nn.initializers.xavier_normal()(keys[key_idx + 1], (d_model, d_model)),
                    W_v=jax.nn.initializers.xavier_normal()(keys[key_idx + 2], (d_model, d_model)),
                    W_o=jax.nn.initializers.xavier_normal()(keys[key_idx + 3], (d_model, d_model)),
                    b_q=jnp.zeros((d_model,)),
                    b_k=jnp.zeros((d_model,)),
                    b_v=jnp.zeros((d_model,)),
                    b_o=jnp.zeros((d_model,)),
                ),
                cross_attention=dict(
                    W_q=jax.nn.initializers.xavier_normal()(keys[key_idx + 4], (d_model, d_model)),
                    W_k=jax.nn.initializers.xavier_normal()(keys[key_idx + 5], (d_model, d_model)),
                    W_v=jax.nn.initializers.xavier_normal()(keys[key_idx + 6], (d_model, d_model)),
                    W_o=jax.nn.initializers.xavier_normal()(keys[key_idx + 7], (d_model, d_model)),
                    b_q=jnp.zeros((d_model,)),
                    b_k=jnp.zeros((d_model,)),
                    b_v=jnp.zeros((d_model,)),
                    b_o=jnp.zeros((d_model,)),
                ),
                feed_forward=dict(
                    W_ff1=jax.nn.initializers.he_normal()(keys[key_idx + 8], (d_model, d_ff)),
                    W_ff2=jax.nn.initializers.xavier_normal()(keys[key_idx + 9], (d_ff, d_model)),
                    b_ff1=jnp.zeros((d_ff,)),
                    b_ff2=jnp.zeros((d_model,)),
                )
            ) for i in range(n_layers)
        ]
    }
    key_idx += 10 * n_layers  # Move index past all decoder layers

    # Output params
    output_params = {
        'W_out': jax.nn.initializers.xavier_normal()(keys[key_idx], (d_model, vocab_size)),
        'b_out': jnp.zeros((vocab_size,))
    }
    key_idx += 1

    assert key_idx <= total_keys, f"Used {key_idx} keys, exceeded {total_keys} allocated"
    params = {
        'embedding': embedding_params,
        'encoder': encoder_params,
        'decoder': decoder_params,
        'output': output_params,
    }
    return params

def compute_accuracy(predictions, targets, pad_token=0):
    """Compute accuracy, ignoring padding tokens and aligning shifted predictions."""
    mask = targets[:, 1:] != pad_token  # Ignore padding after first token
    correct_predictions = jnp.sum((predictions[:, 1:] == targets[:, 1:]) & mask)
    total_valid_tokens = jnp.sum(mask)
    accuracy = correct_predictions / total_valid_tokens if total_valid_tokens > 0 else 0.0
    return accuracy.item()

def train(params, optimizer, opt_state, data_iter, val_en_tokenized, val_ru_targets, train_losses, val_losses, train_accuracies, val_accuracies, max_steps, dropout_rate=0.0, key=jax.random.PRNGKey(0), d_model=64, vocab_size=10000, batch_size=16):
    step_count = 0
    best_val_loss = float('inf')
    early_stopping = False
    sos_id = vocab['<SOS>']
    
    # lr_scheduler_state = ReduceLROnPlateauState(
    #     learning_rate=opt_state[1].hyperparams['learning_rate'],
    #     best_loss=float('inf'),
    #     patience=0
    # )

    ## FIX 3: Increase epochs to allow training to be controlled by max_steps
    for epoch in range(34):
        for step, (batch_input, batch_targets) in enumerate(data_iter):
            step_key = jax.random.fold_in(key, step_count)
            
            current_batch_size = batch_targets.shape[0]
            shifted_targets = jnp.concatenate([jnp.full((current_batch_size, 1), sos_id), batch_targets[:, :-1]], axis=1)

            ## FIX 1: Pass the correct `current_batch_size` to loss_fn, not the fixed `batch_size`
            # THE FIX: Pass the original `batch_targets`, not a shifted version.
            loss, grads = jax.value_and_grad(loss_fn)(
                params, batch_input, batch_targets, current_batch_size, vocab, # <-- Pass original targets
                vocab_size=vocab_size, dropout_rate=dropout_rate, key=step_key, training=True, d_model=d_model
            )

            # ---> ADD THIS DEBUGGING BLOCK <---
            grads_flat, _ = jax.tree_util.tree_flatten(grads)
            has_nan = any(jnp.any(jnp.isnan(g)) for g in grads_flat)
            if has_nan:
                print(f"!!! NaN detected in gradients at step {step_count}. Stopping. !!!")
                break
            # ---> END DEBUGGING BLOCK <---

            train_losses.append(loss)
            
            if step_count % 50 == 0:

                # --- START: ADDED CODE FOR TRAIN ACCURACY ---
                # Get predictions for the current training batch to calculate accuracy
                _, train_predictions = forward(
                    params, batch_input, shifted_targets, 
                    dropout_rate=0.0,  # Use dropout=0.0 for a stable accuracy metric
                    key=step_key, 
                    d_model=d_model, 
                    training=False,  # Set training to False for evaluation
                    vocab_size=vocab_size
                )
                
                # Calculate and append the training accuracy
                train_accuracy = compute_accuracy(train_predictions, batch_targets)
                train_accuracies.append(train_accuracy)
                # --- END: ADDED CODE ---

                val_loss = 0.0
                val_accuracy = 0.0
                val_steps = 0
                val_iter = get_data_iterator(val_en_tokenized, val_ru_targets, batch_size=batch_size)
                for val_step, (val_batch_input, val_batch_targets) in enumerate(val_iter):
                    val_key = jax.random.fold_in(step_key, val_step)
                    current_val_batch_size = val_batch_targets.shape[0]
                    val_shifted_targets = jnp.concatenate([jnp.full((current_val_batch_size, 1), sos_id), val_batch_targets[:, :-1]], axis=1)
                    val_logits, val_predictions = forward(params, val_batch_input, val_shifted_targets, dropout_rate=0.0, key=val_key, d_model=d_model, training=False, vocab_size=vocab_size)
                    
                    ## FIX 2: Pass the correct `current_val_batch_size` to loss_fn here as well
                    val_loss += loss_fn(
                    params, val_batch_input, val_batch_targets, current_val_batch_size, vocab, # <-- Pass original targets
                    vocab_size=vocab_size, dropout_rate=0.0, key=val_key, training=False, d_model=d_model
                    ).item()
                    val_accuracy += compute_accuracy(val_predictions, val_batch_targets)
                    val_steps += 1
                    if val_steps >= 10: break
                val_loss /= val_steps
                val_accuracy /= val_steps
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                # lr_scheduler_state = update_lr_on_plateau(lr_scheduler_state, val_loss)
                # opt_state[1].hyperparams['learning_rate'] = lr_scheduler_state.learning_rate
                
                print(f"Step {step_count}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Train accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
                # print(f"Current Learning Rate: {lr_scheduler_state.learning_rate:.6f}")

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            if step_count >= max_steps:
                print(f"Max steps reached. Breaking at step {step_count}")
                early_stopping = True
                break
            step_count += 1
        if early_stopping:
            break
            
    return params, opt_state, train_losses, val_losses, train_accuracies, val_accuracies, early_stopping, step_count

def check_params_for_nan(params):
    for k, v in params.items():
        if isinstance(v, dict):
            check_params_for_nan(v)  # Recurse into nested dictionaries
        elif isinstance(v, jnp.ndarray):
            if jnp.any(jnp.isnan(v)):
                print(f"NaN detected in {k} parameters!")

def noam_schedule(step: int, d_model: int, warmup_steps: int = 4000, scale_factor: float = 1.0):
    # ... (the existing formula remains the same)
    arg1 = step ** -0.5
    arg2 = step * (warmup_steps ** -1.5)
    
    # Add a scaling factor to control the overall magnitude
    return (d_model ** -0.5) * jnp.minimum(arg1, arg2) * scale_factor

if __name__ == "__main__":
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    MAX_STEPS = 1000
    D_MODEL = 128
    D_FF = D_MODEL*4
    DROPOUT_RATE = 0.2
    N_LAYERS = 6
    N_HEADS = 8

    print(jax.devices())
    
    print(f"""
    --- Model Hyperparameters ---
    Max Steps:              {MAX_STEPS}
    Model Dim (d_model):    {D_MODEL}
    FFN Dim (d_ff):         {D_FF}
    Dropout Rate:           {DROPOUT_RATE}
    Encoder/Decoder Layers: {N_LAYERS}
    Attention Heads:        {N_HEADS}
    ---------------------------
    """)

    key = jax.random.PRNGKey(1)
    vocab, en_sentences, ru_sentences, vocab_size = load_dataset_and_vocab(split="train", max_vocab_size=20000)
    print(f"Vocab size: {vocab_size}")
    print(f"Dataset size: {len(en_sentences)}")

    params = init_params(key, vocab_size=vocab_size, d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS, n_layers=N_LAYERS, dropout_rate=DROPOUT_RATE)

    train_size = int(0.8 * len(en_sentences))
    train_sentences, val_sentences = en_sentences[:train_size], en_sentences[train_size:]
    train_ru, val_ru = ru_sentences[:train_size], ru_sentences[train_size:]
    print(f"Train size: {len(train_sentences)}, Val size: {len(val_sentences)}")

    en_tokenized = tokenize_dataset(train_sentences, vocab, vocab_size=vocab_size)
    ru_targets = generate_targets(tokenize_dataset(train_ru, vocab, vocab_size=vocab_size), vocab)
    val_en_tokenized = tokenize_dataset(val_sentences, vocab, vocab_size=vocab_size)
    val_ru_targets = generate_targets(tokenize_dataset(val_ru, vocab, vocab_size=vocab_size), vocab)
    
    # max_len = 32 # Define max sequence length
    # # Use the single, correct function for both source and target
    # en_tokenized = tokenize_and_pad(train_sentences, vocab, max_len=max_len)
    # ru_targets = tokenize_and_pad(train_ru, vocab, max_len=max_len)
    # val_en_tokenized = tokenize_and_pad(val_sentences, vocab, max_len=max_len)
    # val_ru_targets = tokenize_and_pad(val_ru, vocab, max_len=max_len)

    print(f"len(en_tokenized): {len(en_tokenized)}, len(val_en_tokenized): {len(val_en_tokenized)}")
    
    batch_size = 32

    train_batch_count = len(en_tokenized) // batch_size
    val_batch_count = len(val_en_tokenized) // batch_size
    print(f"Number of train batches per cycle: {train_batch_count}")
    print(f"Number of val batches per cycle: {val_batch_count}")

    data_iter = get_data_iterator(en_tokenized, ru_targets, batch_size=batch_size)
    # --- Start: New Learning Rate Schedule ---

    # 1. Define the hyperparameters for the schedule
    # warmup_steps = 400
    # peak_learning_rate = 3e-4  # A common starting point for Transformers
    # # Ensure total_steps is defined (it's your `max_steps` variable)
    # total_steps = max_steps 

    # # 2. Define the linear warmup and exponential decay phases
    # warmup_fn = optax.linear_schedule(
    #     init_value=0.0, 
    #     end_value=peak_learning_rate, 
    #     transition_steps=warmup_steps
    # )
    
    # decay_fn = optax.exponential_decay(
    #     init_value=peak_learning_rate,
    #     transition_steps=total_steps - warmup_steps,
    #     decay_rate=0.99
    # )

    # # 3. Combine the two schedules
    # lr_schedule = optax.join_schedules(
    #     schedules=[warmup_fn, decay_fn], 
    #     boundaries=[warmup_steps]
    # )

    # # 4. Create the final optimizer
    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.adam(learning_rate=lr_schedule)
    # )
    
    # opt_state = optimizer.init(params)

    # 3. Create the optimizer with the new schedule function
    #    (Note: we are now passing a function, not a pre-built schedule object)
    # d_model = 512
    # warmup_steps = 400 # The paper used 4000, a common default
    # # When creating your optimizer, pass in a smaller scale_factor
    # # Start with 0.5 and be prepared to go lower if instability persists.
    # learning_rate_fn = lambda step: noam_schedule(
    #     step, D_MODEL, warmup_steps, scale_factor=0.5
    # )

    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.adam(learning_rate=learning_rate_fn)
    # )

    optimizer = optax.adam(learning_rate=5e-5)

    opt_state = optimizer.init(params)

    params, opt_state, losses_post_train, losses_post_val, train_accuracies, val_accuracies, early_stopped, final_steps = train(params, optimizer, opt_state, data_iter, val_en_tokenized, val_ru_targets, train_losses, val_losses, train_accuracies, val_accuracies, MAX_STEPS, dropout_rate=DROPOUT_RATE, key=key, d_model=D_MODEL, vocab_size=vocab_size, batch_size=batch_size)
    if not early_stopped:
        assert len(losses_post_train) == final_steps + 1, f"Train losses length: {len(losses_post_train)}, expected {final_steps + 1}"
        assert len(losses_post_val) == (final_steps // 10) + 1, f"Val losses length: {len(losses_post_val)}, expected {(final_steps // 10) + 1}"

    print(f"final_steps: {final_steps}, len(losses_post_train): {len(losses_post_train)}, len(losses_post_val): {len(losses_post_val)}")


    # After the train() function returns the final trained_params...
    print("\nTraining complete. Saving model weights...")
    byte_data = to_bytes(params)

    with open("transformer_weights.msgpack", "wb") as f:
        f.write(byte_data)

    print("✅ Model weights saved to transformer_weights.msgpack")

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Loss
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Average Loss per Step', color=color)
    ax1.plot(range(len(losses_post_train)), losses_post_train, label='Train Loss', color=color)
    ax1.plot(range(0, len(losses_post_val) * 50, 50), losses_post_val, label='Val Loss', color='tab:orange', marker='o')
    # ax1.axvline(x=100, color='gray', linestyle='--', label='Flatten Point')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Instantiate a second y-axis for Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(0, len(train_accuracies) * 50, 50), train_accuracies, label='Train Accuracy', color='tab:green', linestyle='--')
    ax2.plot(range(0, len(val_accuracies) * 50, 50), val_accuracies, label='Val Accuracy', color='tab:purple', marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # Title and layout
    plt.title(f'Training Metrics Over {final_steps} Steps')
    fig.tight_layout()
    
    import datetime
    now = datetime.datetime.now()

    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    filename = f'metrics_plot_{final_steps}_steps_{timestamp_str}.png'
    full_path = os.path.join(plot_dir, filename)

    print(f"Saving plot to: {full_path}")
    plt.savefig(full_path)
    plt.show()

    print(f"Train loss at step 100: {losses_post_train[100] if len(losses_post_train) > 100 else losses_post_train[-1]}")
    print(f"Val loss at step 100: {losses_post_val[10] if len(losses_post_val) > 11 else losses_post_val[-1]}")
    print(f"Train accuracy at step 100: {train_accuracies[100] if len(train_accuracies) > 100 else train_accuracies[-1]:.4f}")
    print(f"Val accuracy at step 100: {val_accuracies[10] if len(val_accuracies) > 11 else val_accuracies[-1]:.4f}")
    print(f"Train loss at step {final_steps}: {losses_post_train[final_steps - 1]}")
    print(f"Val loss at step {final_steps}: {losses_post_val[-1]}")
    print(f"Train accuracy at step {final_steps}: {train_accuracies[- 1]:.4f}")
    print(f"Val accuracy at step {final_steps}: {val_accuracies[-1]:.4f}")