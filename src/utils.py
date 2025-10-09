# ----------------------------------------------------------------------------
# FILE: utils.py
# Loss, accuracy, init params, save/load
# ----------------------------------------------------------------------------
import jax
import jax.numpy as jnp
from flax.serialization import to_bytes, from_bytes
import jax.nn.initializers as init
from typing import Dict, Any
import optax


def smoothed_loss(logits, targets, vocab, smoothing=0.1):
    num_classes = logits.shape[-1]
    confidence = 1.0 - smoothing
    low_conf = smoothing / num_classes
    targets_onehot = jax.nn.one_hot(targets, num_classes)
    soft_targets = confidence * targets_onehot + low_conf
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(soft_targets * log_probs, axis=-1)
    mask = (targets != vocab['<PAD>'])
    return jnp.sum(loss * mask) / (jnp.sum(mask) + 1e-8)


def compute_accuracy(predictions, targets, vocab):
    mask = (targets != vocab['<PAD>'])
    acc = jnp.sum((predictions == targets) * mask) / (jnp.sum(mask) + 1e-8)
    return float(acc)


def init_params(key, vocab_size, d_model, d_ff, n_heads, n_layers):
    keys = jax.random.split(key, 1000)
    k_idx = 0
    def _init(initializer, shape):
        nonlocal k_idx
        k = keys[k_idx]
        k_idx += 1
        return initializer(k, shape)

    def create_norm_params():
        return {'scale': _init(init.ones, (d_model,)), 'offset': _init(init.zeros, (d_model,))}

    def create_attention_params():
        return {
            'W_q': _init(init.xavier_normal(), (d_model, d_model)),
            'W_k': _init(init.xavier_normal(), (d_model, d_model)),
            'W_v': _init(init.xavier_normal(), (d_model, d_model)),
            'W_o': _init(init.xavier_normal(), (d_model, d_model)),
        }

    def create_ffn_params():
        return {
            'W1': _init(init.he_normal(), (d_model, d_ff)),
            'b1': _init(init.zeros, (d_ff,)),
            'W2': _init(init.xavier_normal(), (d_ff, d_model)),
            'b2': _init(init.zeros, (d_model,)),
        }

    params = {
        'embedding': {'W_emb': _init(init.xavier_normal(), (vocab_size, d_model))},
        'encoder': {'layers': [], 'final_norm': create_norm_params()},
        'decoder': {'layers': [], 'final_norm': create_norm_params()},
        'output': {'W_out': _init(init.xavier_normal(), (d_model, vocab_size))}
    }
    for _ in range(n_layers):
        params['encoder']['layers'].append({'self_attention': create_attention_params(), 'feed_forward': create_ffn_params(), 'norm1': create_norm_params(), 'norm2': create_norm_params()})
        params['decoder']['layers'].append({'self_attention': create_attention_params(), 'cross_attention': create_attention_params(), 'feed_forward': create_ffn_params(), 'norm1': create_norm_params(), 'norm2': create_norm_params(), 'norm3': create_norm_params()})
    return params


def save_params(params, path: str):
    with open(path, 'wb') as f:
        f.write(to_bytes(params))


def load_params(path: str, target):
    with open(path, 'rb') as f:
        data = f.read()
    return from_bytes(target, data)
