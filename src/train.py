# ----------------------------------------------------------------------------
# FILE: train.py
# Training loop and CLI entrypoint to run experiments
# ----------------------------------------------------------------------------
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import time
from typing import Dict

# We'll import from the other modules
from model import forward, greedy_decode, sinusoidal_positional_embeddings
from data import load_dataset_and_vocab, tokenize_and_pad, get_data_iterator
from utils import init_params, smoothed_loss, compute_accuracy, save_params


def make_optimizer(warmup_steps=1000, base_lr=3e-4, weight_decay=1e-4, max_steps=10000):
    schedule = optax.join_schedules(
        schedules=[optax.linear_schedule(init_value=0.0, end_value=base_lr, transition_steps=warmup_steps),
                   optax.cosine_decay_schedule(init_value=base_lr, decay_steps=max_steps - warmup_steps)],
        boundaries=[warmup_steps]
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=weight_decay)
    )
    return opt, schedule

# @jax.jit
def loss_and_grads(params, batch_input, batch_targets, vocab, model_args, dropout_rate, key):
    """Compute loss and gradients with a single forward pass (no jit)."""
    def _loss(p):
        sos = vocab['<SOS>']
        dec_input = jnp.concatenate(
            [jnp.full((batch_targets.shape[0], 1), sos), batch_targets[:, :-1]], axis=1
        )

        logits, _ = forward(
            p,
            batch_input,
            dec_input,
            d_model=model_args['d_model'],
            n_layers=model_args['n_layers'],
            n_heads=model_args['n_heads'],
            d_ff=model_args['d_ff'],
            dropout_rate=dropout_rate,
            training=True,
            key=key,
            vocab=vocab,
            positional_emb=model_args.get('positional_emb'),
        )

        return smoothed_loss(logits, batch_targets, vocab, smoothing=0.1)

    # Compute both loss and grads together (one forward pass)
    loss, grads = jax.value_and_grad(_loss)(params)
    return loss, grads

def train(max_steps=3000, batch_size=32, d_model=256, n_layers=4, n_heads=8, d_ff=1024, max_len=32):
    key = jax.random.PRNGKey(42)
    key, data_key, params_key = jax.random.split(key, 3)

    vocab, en_sentences, ru_sentences, vocab_size = load_dataset_and_vocab(subset_pct="train[:5%]", max_vocab_size=40000)
    ru_sentences = en_sentences.copy()

    train_size = int(0.8 * len(en_sentences))
    train_en = en_sentences[:train_size]
    val_en = en_sentences[train_size:]
    train_ru = ru_sentences[:train_size]
    val_ru = ru_sentences[train_size:]

    # small subset for faster runs (tweak as needed)
    # train_en = train_en[:1000]; train_ru = train_ru[:1000]
    # val_en = val_en[:200]; val_ru = val_ru[:200]

    train_tok = tokenize_and_pad(train_en, vocab, max_len)
    train_tgt = tokenize_and_pad(train_ru, vocab, max_len)
    val_tok = tokenize_and_pad(val_en, vocab, max_len)
    val_tgt = tokenize_and_pad(val_ru, vocab, max_len)

    params = init_params(params_key, vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers)

    positional_emb = sinusoidal_positional_embeddings(max_len, d_model)
    model_args = {'d_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads, 'd_ff': d_ff, 'positional_emb': positional_emb}

    optimizer, schedule = make_optimizer(warmup_steps=1000, base_lr=3e-4, weight_decay=1e-4, max_steps=max_steps)
    opt_state = optimizer.init(params)

    data_iter = get_data_iterator(train_tok, train_tgt, batch_size, data_key)

    train_losses = []
    val_accs = []

    pbar = tqdm(range(max_steps), desc='Training Steps')
    global_step = 0
    for step in pbar:
        key, subkey = jax.random.split(key)
        batch_input, batch_targets = next(data_iter)
        loss, grads = loss_and_grads(params, batch_input, batch_targets, vocab, model_args, dropout_rate=0.1, key=subkey)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        lr = schedule(global_step)
        global_step += 1

        train_losses.append(float(loss))

        if (step + 1) % 50 == 0:
            # validation
            val_iter = get_data_iterator(val_tok, val_tgt, batch_size, data_key)
            num_val_batches = max(1, len(val_tok) // batch_size)
            accs = []
            for _ in range(num_val_batches):
                v_in, v_tgt = next(val_iter)
                sos = vocab['<SOS>']
                dec_input = jnp.concatenate([jnp.full((v_tgt.shape[0], 1), sos), v_tgt[:, :-1]], axis=1)
                logits, preds = forward(params, v_in, dec_input,
                        training=False, key=None, vocab=vocab, dropout_rate=0.0, **model_args)
                accs.append(compute_accuracy(preds, v_tgt, vocab))
            avg_acc = sum(accs) / len(accs)
            val_accs.append(avg_acc)
            pbar.set_postfix({'Step': step+1, 'LR': float(lr), 'Train Loss': float(loss), 'Val Acc': float(avg_acc)})

    save_params(params, 'transformer_weights.msgpack')
    print('Training complete. Weights saved to transformer_weights.msgpack')
    return params, vocab, positional_emb


if __name__ == '__main__':
    # quick entrypoint
    train(max_steps=2000, batch_size=32)