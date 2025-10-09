# ----------------------------------------------------------------------------
# FILE: model.py
# Core Transformer model: init_params, encoder/decoder, forward, greedy decode
# ----------------------------------------------------------------------------
import jax
import jax.numpy as jnp
from typing import Dict, Any

# ------------------------ Positional embeddings (precompute allowed) -----------------

def sinusoidal_positional_embeddings(max_len: int, d_model: int) -> jnp.ndarray:
    pos = jnp.arange(max_len)[:, None]
    i = jnp.arange(d_model // 2)[None, :]
    arg = pos / jnp.power(10000.0, 2 * i / d_model)
    pe = jnp.zeros((max_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(arg))
    pe = pe.at[:, 1::2].set(jnp.cos(arg))
    return pe[None, :]


def layer_norm(x, scale, offset, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + offset


# ------------------------ Single-layer primitives ----------------------------------

def split_heads(x, n_heads):
    # x: [B, T, D] -> [B, n_heads, T, D_head]
    B, T, D = x.shape
    D_head = D // n_heads
    return x.reshape(B, T, n_heads, D_head).transpose(0, 2, 1, 3)


def combine_heads(x):
    # x: [B, n_heads, T, D_head] -> [B, T, D]
    B, n_heads, T, D_head = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, n_heads * D_head)


# ------------------------ Encoder / Decoder layer implementations -------------------

def multi_head_attention(q, k, v, W_o, mask=None, dropout_rate=0.0, key=None, training=False):
    # q,k,v: [B, T, D] already projected
    # split
    n_heads = q.shape[-1] // (q.shape[-1] // (q.shape[-1]))  # placeholder to keep signatures similar
    # We expect q,k,v already in shape [B, n_heads, T, D_head]
    # compute scaled dot-product
    scale = jnp.sqrt(q.shape[-1]).astype(q.dtype)
    logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / scale
    if mask is not None:
        logits = jnp.where(mask, logits, -1e9)
    A = jax.nn.softmax(logits, axis=-1)
    if training and dropout_rate > 0.0 and key is not None:
        key, subkey = jax.random.split(key)
        A = A * jax.random.bernoulli(subkey, 1 - dropout_rate, A.shape) / (1 - dropout_rate)
    out = jnp.matmul(A, v)
    return out, A


# NOTE: to keep this file compact we will implement encoder/decoder layer functions that
# expect their projection matrices W_q,k,v,o to be applied outside or provided in param dicts.


def encoder_layer(x, params, key, n_heads, d_model, d_ff, dropout_rate, training, enc_input=None, vocab=None):
    B, T, D = x.shape
    # layer norm
    x_norm = layer_norm(x, params['norm1']['scale'], params['norm1']['offset'])

    # project
    W_q, W_k, W_v, W_o = params['self_attention']['W_q'], params['self_attention']['W_k'], params['self_attention']['W_v'], params['self_attention']['W_o']
    Q = split_heads(jnp.dot(x_norm, W_q), n_heads)
    K = split_heads(jnp.dot(x_norm, W_k), n_heads)
    V = split_heads(jnp.dot(x_norm, W_v), n_heads)

    # mask pad
    mask = None
    if enc_input is not None and vocab is not None:
        pad_mask = (enc_input != vocab['<PAD>'])[:, None, None, :]
        mask = pad_mask

    attn_out, _ = multi_head_attention(Q, K, V, W_o, mask=mask, dropout_rate=dropout_rate, key=key, training=training)
    attn_out = combine_heads(attn_out)
    attn_out = jnp.dot(attn_out, W_o)
    res1 = x + attn_out

    # FFN
    x2 = layer_norm(res1, params['norm2']['scale'], params['norm2']['offset'])
    W1, b1, W2, b2 = params['feed_forward']['W1'], params['feed_forward']['b1'], params['feed_forward']['W2'], params['feed_forward']['b2']
    hidden = jax.nn.relu(jnp.dot(x2, W1) + b1[None, None, :])
    ff = jnp.dot(hidden, W2) + b2[None, None, :]
    if training and dropout_rate > 0.0:
        key, subkey = jax.random.split(key)
        ff = ff * jax.random.bernoulli(subkey, 1 - dropout_rate, ff.shape) / (1 - dropout_rate)
    out = res1 + ff
    return out


def decoder_layer(x, enc_output, params, key, n_heads, d_model, d_ff, dropout_rate, training, dec_input=None, enc_input=None, vocab=None, debug=False):
    B, T, D = x.shape
    # self-attn
    x_norm = layer_norm(x, params['norm1']['scale'], params['norm1']['offset'])
    W_q, W_k, W_v, W_o = params['self_attention']['W_q'], params['self_attention']['W_k'], params['self_attention']['W_v'], params['self_attention']['W_o']
    Q = split_heads(jnp.dot(x_norm, W_q), n_heads)
    K = split_heads(jnp.dot(x_norm, W_k), n_heads)
    V = split_heads(jnp.dot(x_norm, W_v), n_heads)

    # prepare masks: pad + lookahead
    if dec_input is None or vocab is None:
        raise ValueError("decoder_layer requires dec_input and vocab for masking")
    pad_mask = (dec_input != vocab['<PAD>'])[:, None, None, :]
    lookahead = jnp.tril(jnp.ones((1, 1, T, T), dtype=bool))
    mask = pad_mask & lookahead

    attn_out, A = multi_head_attention(Q, K, V, W_o, mask=mask, dropout_rate=dropout_rate, key=key, training=training)
    attn_out = combine_heads(attn_out)
    attn_out = jnp.dot(attn_out, W_o)
    res1 = x + attn_out

    # cross-attn
    x_norm2 = layer_norm(res1, params['norm2']['scale'], params['norm2']['offset'])
    W_cq, W_ck, W_cv, W_co = params['cross_attention']['W_q'], params['cross_attention']['W_k'], params['cross_attention']['W_v'], params['cross_attention']['W_o']
    Qc = split_heads(jnp.dot(x_norm2, W_cq), n_heads)
    Kc = split_heads(jnp.dot(enc_output, W_ck), n_heads)
    Vc = split_heads(jnp.dot(enc_output, W_cv), n_heads)

    enc_mask = None
    if enc_input is not None and vocab is not None:
        enc_mask = (enc_input != vocab['<PAD>'])[:, None, None, :]

    cross_out, A2 = multi_head_attention(Qc, Kc, Vc, W_co, mask=enc_mask, dropout_rate=dropout_rate, key=key, training=training)
    cross_out = combine_heads(cross_out)
    cross_out = jnp.dot(cross_out, W_co)
    res2 = res1 + cross_out

    # FFN
    x3 = layer_norm(res2, params['norm3']['scale'], params['norm3']['offset'])
    W1, b1, W2, b2 = params['feed_forward']['W1'], params['feed_forward']['b1'], params['feed_forward']['W2'], params['feed_forward']['b2']
    hidden = jax.nn.relu(jnp.dot(x3, W1) + b1[None, None, :])
    ff = jnp.dot(hidden, W2) + b2[None, None, :]
    if training and dropout_rate > 0.0:
        key, subkey = jax.random.split(key)
        ff = ff * jax.random.bernoulli(subkey, 1 - dropout_rate, ff.shape) / (1 - dropout_rate)
    out = res2 + ff

    if debug:
        try:
            print(f"Decoder cross-attn mean={float(jnp.mean(A2)):.6f}")
        except Exception:
            pass
    return out


# ------------------------ Encoder / Decoder stacks ---------------------------------

def transformer_encoder(emb, params, keys, n_layers, n_heads, d_model, d_ff, dropout_rate, training, enc_input=None, vocab=None):
    x = emb
    for i in range(n_layers):
        x = encoder_layer(x, params['layers'][i], keys[i], n_heads, d_model, d_ff, dropout_rate, training, enc_input=enc_input, vocab=vocab)
    return layer_norm(x, params['final_norm']['scale'], params['final_norm']['offset'])


def transformer_decoder(dec_emb, enc_output, params, keys, n_layers, n_heads, d_model, d_ff, dropout_rate, training, dec_input=None, enc_input=None, vocab=None, debug=False):
    x = dec_emb
    for i in range(n_layers):
        x = decoder_layer(x, enc_output, params['layers'][i], keys[i], n_heads, d_model, d_ff, dropout_rate, training, dec_input=dec_input, enc_input=enc_input, vocab=vocab, debug=debug)
    return layer_norm(x, params['final_norm']['scale'], params['final_norm']['offset'])


# ------------------------ High-level forward & decode --------------------------------

def forward(params: Dict[str, Any], enc_input, dec_input,
            d_model, n_layers, n_heads, d_ff,
            dropout_rate, training, key=None, vocab=None, positional_emb=None, keys=None, debug=False):
    if key is None and keys is None:
        key = jax.random.PRNGKey(0)

    if keys is None:
        n_keys = int(n_layers * 2)
        keys = jax.random.split(key, n_keys)

    enc_keys, dec_keys = keys[:n_layers], keys[n_layers:]

    # Embedding lookup + positional embedding (positional_emb can be precomputed)
    enc_emb = params['embedding']['W_emb'][enc_input]
    dec_emb = params['embedding']['W_emb'][dec_input]
    if positional_emb is not None:
        enc_emb = enc_emb + positional_emb[:, :enc_input.shape[1], :]
        dec_emb = dec_emb + positional_emb[:, :dec_input.shape[1], :]
    else:
        enc_emb = enc_emb + sinusoidal_positional_embeddings(enc_input.shape[1], d_model)
        dec_emb = dec_emb + sinusoidal_positional_embeddings(dec_input.shape[1], d_model)

    enc_output = transformer_encoder(enc_emb, params['encoder'], enc_keys, n_layers, n_heads, d_model, d_ff, dropout_rate, training, enc_input=enc_input, vocab=vocab)
    dec_output = transformer_decoder(dec_emb, enc_output, params['decoder'], dec_keys, n_layers, n_heads, d_model, d_ff, dropout_rate, training, dec_input=dec_input, enc_input=enc_input, vocab=vocab, debug=debug)

    logits = jnp.dot(dec_output, params['output']['W_out'])
    preds = jnp.argmax(logits, axis=-1)
    return logits, preds


def greedy_decode(params, enc_input, vocab, max_len=32, d_model=256, n_layers=4, n_heads=8, d_ff=1024, dropout_rate=0.0, positional_emb=None):
    sos = vocab['<SOS>']
    eos = vocab['<EOS>']
    dec_input = jnp.full((enc_input.shape[0], 1), sos, dtype=jnp.int32)

    for _ in range(max_len - 1):
        logits, preds = forward(params, enc_input, dec_input, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate, training=False, key=None, vocab=vocab, positional_emb=positional_emb)
        next_token = preds[:, -1:]
        dec_input = jnp.concatenate([dec_input, next_token], axis=1)
        # early stopping when all sequences produced EOS
        if jnp.all(next_token == eos):
            break
    return dec_input
