import jax
import jax.numpy as jnp
import optax
from flax.serialization import to_bytes
import matplotlib.pyplot as plt
import datetime
import os
import re
from collections import Counter
from datasets import load_dataset
from typing import List, Dict
from transliterate import translit
from tqdm import tqdm

# ==============================================================================
# SECTION 1: DATA PROCESSING FUNCTIONS
# ==============================================================================

# def normalize_text(text: str) -> str:
#     """
#     Normalizes text by transliterating, converting to lowercase, and removing punctuation.
#     """
#     normalized = translit(text, 'ru', reversed=True).lower()
#     normalized = re.sub(r"[^a-zA-Zа-яА-Я0-9']+", " ", normalized)
#     return normalized.strip()

def normalize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9']+", " ", text.lower()).strip()

def load_dataset_and_vocab(split: str = "train", max_vocab_size: int = 20000):
    """
    Loads dataset from Hugging Face and builds a stable, correctly-sized vocabulary.
    """
    # dataset = load_dataset("opus_books", "en-ru", split=split)
    # print(f"Loaded {dataset.num_rows} samples for split: {split}")

    dataset = load_dataset("wmt14", "ru-en", split="train[:1%]")  # start with small subset
    print(f"Loaded {dataset.num_rows} samples for split: train[:1%]")

    en_sentences = [normalize_text(pair['en']) for pair in dataset['translation']]
    ru_sentences = [normalize_text(pair['ru']) for pair in dataset['translation']]

    special_tokens = ['<PAD>', '<UNK>', '<EOS>', '<SOS>']
    vocab = {token: i for i, token in enumerate(special_tokens)}

    word_counts = Counter(" ".join(en_sentences + ru_sentences).split())
    
    for word, _ in word_counts.most_common(max_vocab_size - len(special_tokens)):
        if word not in vocab:
            vocab[word] = len(vocab)
            
    final_vocab_size = len(vocab)
    print(f"Vocabulary built successfully. Final size: {final_vocab_size} tokens.")
    
    return vocab, en_sentences, ru_sentences, final_vocab_size

def tokenize_and_pad(sentences: List[str], vocab: Dict[str, int], max_len: int) -> jnp.ndarray:
    """A single, unified function to tokenize and pad sentences."""
    unk_id = vocab['<UNK>']
    pad_id = vocab['<PAD>']
    eos_id = vocab['<EOS>']
    
    all_token_ids = []
    for sentence in sentences:
        words = sentence.split()
        token_ids = [vocab.get(word, unk_id) for word in words]
        token_ids.append(eos_id)
        
        if len(token_ids) >= max_len:
            padded_ids = token_ids[:max_len]
        else:
            padded_ids = token_ids + [pad_id] * (max_len - len(token_ids))
        
        all_token_ids.append(padded_ids)
            
    return jnp.array(all_token_ids, dtype=jnp.int32)

def get_data_iterator(en_tokenized, ru_targets, batch_size, key=None):
    """Yields batches of data, shuffling each time."""
    if key is None:
        key = jax.random.PRNGKey(0)
    
    dataset_size = len(en_tokenized)
    assert dataset_size == len(ru_targets)
    
    while True:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            batch_indices = perm[i:i+batch_size]
            yield en_tokenized[batch_indices], ru_targets[batch_indices]

# ==============================================================================
# SECTION 2: TRANSFORMER MODEL ARCHITECTURE
# ==============================================================================

def positional_embeddings(max_len: int, d_model: int) -> jnp.ndarray:
    pos = jnp.arange(max_len)[:, None]
    i = jnp.arange(d_model // 2)[None, :]
    arg = pos / jnp.power(10000, 2 * i / d_model)
    pe = jnp.zeros((max_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(arg))
    pe = pe.at[:, 1::2].set(jnp.cos(arg))
    return pe[None, ...]

def layer_norm(x, scale, offset, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + offset

def encoder_layer(emb, params, key, n_heads, d_model, d_ff, dropout_rate, training, **kwargs):
    batch_size, max_len, _ = emb.shape
    
    norm1_params = params['norm1']
    norm1 = layer_norm(emb, scale=norm1_params['scale'], offset=norm1_params['offset'])
    
    W_q, W_k, W_v, W_o = params['self_attention'].values()
    Q = jnp.dot(norm1, W_q).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    K = jnp.dot(norm1, W_k).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    V = jnp.dot(norm1, W_v).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    
    E = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_model // n_heads)

    # Get pad mask based on encoder input token IDs
    enc_input = kwargs.get("enc_input")
    if enc_input is not None:
        vocab = kwargs["vocab"]
        pad_mask = (enc_input != vocab["<PAD>"])[:, None, None, :]
        E = jnp.where(pad_mask, E, -1e9)
    
    A = jax.nn.softmax(E, axis=-1)
    
    Y = jnp.matmul(A, V).transpose(0, 2, 1, 3).reshape(batch_size, max_len, d_model)
    attention_out = jnp.dot(Y, W_o)
    residual = emb + attention_out
    
    norm2_params = params['norm2']
    norm2 = layer_norm(residual, scale=norm2_params['scale'], offset=norm2_params['offset'])
    
    # W1, b1, W2, b2 = params['feed_forward'].values()

    ffn_params = params['feed_forward']
    W1, b1, W2, b2 = ffn_params['W1'], ffn_params['b1'], ffn_params['W2'], ffn_params['b2']

    ffn_out = jnp.dot(jax.nn.relu(jnp.dot(norm2, W1) + b1), W2) + b2
    
    if training:
        key, subkey = jax.random.split(key)
        ffn_out = ffn_out * jax.random.bernoulli(subkey, 1 - dropout_rate, ffn_out.shape) / (1 - dropout_rate)
        
    final_out = residual + ffn_out
    return final_out

def transformer_encoder(emb, params, keys, n_layers, **kwargs):
    x = emb
    for i in range(n_layers):
        x = encoder_layer(x, params['layers'][i], keys[i], **kwargs)
    final_norm_params = params['final_norm']
    return layer_norm(x, scale=final_norm_params['scale'], offset=final_norm_params['offset'])

def decoder_layer(dec_emb, enc_output, params, key, n_heads, d_model, d_ff, dropout_rate, training, debug=False, **kwargs):
    """
    Decoder layer: self-attention -> cross-attention -> feed-forward.
    - dec_emb: [B, T_dec, D]
    - enc_output: [B, T_enc, D]
    - kwargs may contain 'dec_input' (token ids) and 'enc_input' (token ids) and 'vocab'
    """
    batch_size, max_len, _ = dec_emb.shape

    # -------------------------
    # 1) Decoder self-attention
    # -------------------------
    norm1_params = params['norm1']
    norm1 = layer_norm(dec_emb, scale=norm1_params['scale'], offset=norm1_params['offset'])

    W_q, W_k, W_v, W_o = params['self_attention'].values()
    Q = jnp.dot(norm1, W_q).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    K = jnp.dot(norm1, W_k).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    V = jnp.dot(norm1, W_v).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)

    # Scaled dot-product
    E = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_model // n_heads)

    # masks: pad_mask (B,1,1,T_dec) & look-ahead (1,1,T_dec,T_dec) -> broadcast to (B, n_heads, T_dec, T_dec)
    dec_input = kwargs.get("dec_input")
    vocab = kwargs.get("vocab")
    if dec_input is None or vocab is None:
        raise ValueError("decoder_layer requires dec_input and vocab in kwargs for masking")

    pad_mask = (dec_input != vocab["<PAD>"])[:, None, None, :]   # [B,1,1,T_dec]
    look_ahead_mask = jnp.tril(jnp.ones((1, 1, max_len, max_len), dtype=bool))  # [1,1,T_dec,T_dec]
    mask = pad_mask & look_ahead_mask
    E = jnp.where(mask, E, -1e9)

    A = jax.nn.softmax(E, axis=-1)

    # optional attention dropout
    if training and dropout_rate > 0.0:
        key, subkey = jax.random.split(key)
        A = A * jax.random.bernoulli(subkey, 1 - dropout_rate, A.shape) / (1 - dropout_rate)

    Y = jnp.matmul(A, V).transpose(0, 2, 1, 3).reshape(batch_size, max_len, d_model)
    self_attention_out = jnp.dot(Y, W_o)
    residual1 = dec_emb + self_attention_out

    # -------------------------
    # 2) Cross-attention (decoder -> encoder)
    # -------------------------
    norm2_params = params['norm2']
    norm2 = layer_norm(residual1, scale=norm2_params['scale'], offset=norm2_params['offset'])

    W_cq, W_ck, W_cv, W_co = params['cross_attention'].values()
    Q = jnp.dot(norm2, W_cq).reshape(batch_size, max_len, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    K = jnp.dot(enc_output, W_ck).reshape(batch_size, -1, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)
    V = jnp.dot(enc_output, W_cv).reshape(batch_size, -1, n_heads, d_model // n_heads).transpose(0, 2, 1, 3)

    E = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_model // n_heads)

    # Mask encoder padding tokens if enc_input provided
    enc_input = kwargs.get("enc_input")
    if enc_input is not None:
        enc_pad_mask = (enc_input != vocab["<PAD>"])[:, None, None, :]   # [B,1,1,T_enc]
        E = jnp.where(enc_pad_mask, E, -1e9)

    A = jax.nn.softmax(E, axis=-1)

    # optional attention dropout
    if training and dropout_rate > 0.0:
        key, subkey = jax.random.split(key)
        A = A * jax.random.bernoulli(subkey, 1 - dropout_rate, A.shape) / (1 - dropout_rate)

    Y = jnp.matmul(A, V).transpose(0, 2, 1, 3).reshape(batch_size, max_len, d_model)
    cross_attention_out = jnp.dot(Y, W_co)
    residual2 = residual1 + cross_attention_out

    # -------------------------
    # 3) Feed-forward
    # -------------------------
    norm3_params = params['norm3']
    norm3 = layer_norm(residual2, scale=norm3_params['scale'], offset=norm3_params['offset'])

    ffn_params = params['feed_forward']
    W1, b1, W2, b2 = ffn_params['W1'], ffn_params['b1'], ffn_params['W2'], ffn_params['b2']

    ffn_hidden = jax.nn.relu(jnp.dot(norm3, W1) + b1[None, None, :])
    ffn_out = jnp.dot(ffn_hidden, W2) + b2[None, None, :]

    if training and dropout_rate > 0.0:
        key, subkey = jax.random.split(key)
        ffn_out = ffn_out * jax.random.bernoulli(subkey, 1 - dropout_rate, ffn_out.shape) / (1 - dropout_rate)

    final_out = residual2 + ffn_out

    # Optional debug prints (per-layer)
    if debug:
        try:
            mean_attn = float(jnp.mean(A))
            max_attn = float(jnp.max(A))
            print(f"Decoder layer cross-attn mean={mean_attn:.6f} max={max_attn:.6f}")
        except Exception:
            pass

    return final_out

def transformer_decoder(dec_emb, enc_output, params, keys, n_layers, debug=False, **kwargs):
    x = dec_emb
    for i in range(n_layers):
        x = decoder_layer(x, enc_output, params['layers'][i], keys[i], debug=debug, **kwargs)
    final_norm_params = params['final_norm']
    return layer_norm(x, scale=final_norm_params['scale'], offset=final_norm_params['offset'])

def forward(params, enc_input, dec_input,
            vocab_size, d_model, n_layers, n_heads, d_ff,
            dropout_rate, training, key=None, vocab=None, debug=False):
    if key is None:
        key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_layers * 2)
    enc_keys, dec_keys = keys[:n_layers], keys[n_layers:]

    enc_emb = params['embedding']['W_emb'][enc_input]
    enc_emb += positional_embeddings(max_len=enc_input.shape[1], d_model=d_model)
    
    dec_emb = params['embedding']['W_emb'][dec_input]
    dec_emb += positional_embeddings(max_len=dec_input.shape[1], d_model=d_model)
    
    enc_output = transformer_encoder(enc_emb, params['encoder'], enc_keys, n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, training=training, enc_input=enc_input, vocab=vocab)
    dec_output = transformer_decoder(dec_emb, enc_output, params['decoder'], dec_keys, n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, training=training, debug=debug, enc_input=enc_input, dec_input=dec_input, vocab=vocab)
    
    logits = jnp.dot(dec_output, params['output']['W_out'])
    predictions = jnp.argmax(logits, axis=-1)

    if debug:
        probs = jax.nn.softmax(logits[0, -1])
        topk_idx = jnp.argsort(probs)[-5:][::-1]
        topk_words = [list(vocab.keys())[int(i)] for i in topk_idx]
        topk_probs = [float(probs[i]) for i in topk_idx]
        print(f"[DEBUG] Top-5 next tokens: {list(zip(topk_words, topk_probs))}")

    return logits, predictions

def init_params(key, vocab_size, d_model, d_ff, n_heads, n_layers):
    """Initializes a full set of learnable parameters for the Transformer."""
    keys = jax.random.split(key, 200) # Ample keys for all layers
    k_idx = 0
    
    def _init(initializer, shape):
        nonlocal k_idx
        k, k_idx = keys[k_idx], k_idx + 1
        return initializer(k, shape)

    def create_norm_params():
        return {'scale': _init(jax.nn.initializers.ones, (d_model,)), 'offset': _init(jax.nn.initializers.zeros, (d_model,))}
    def create_attention_params():
        return {
            'W_q': _init(jax.nn.initializers.xavier_normal(), (d_model, d_model)),
            'W_k': _init(jax.nn.initializers.xavier_normal(), (d_model, d_model)),
            'W_v': _init(jax.nn.initializers.xavier_normal(), (d_model, d_model)),
            'W_o': _init(jax.nn.initializers.xavier_normal(), (d_model, d_model)),
        }
    def create_ffn_params():
        return {
            'W1': _init(jax.nn.initializers.he_normal(), (d_model, d_ff)),
            'b1': _init(jax.nn.initializers.zeros, (d_ff,)),
            'W2': _init(jax.nn.initializers.xavier_normal(), (d_ff, d_model)),
            'b2': _init(jax.nn.initializers.zeros, (d_model,)),
        }
    
    return {
        'embedding': {'W_emb': _init(jax.nn.initializers.xavier_normal(), (vocab_size, d_model))},
        'encoder': {
            'layers': [{'self_attention': create_attention_params(), 'feed_forward': create_ffn_params(), 'norm1': create_norm_params(), 'norm2': create_norm_params()} for _ in range(n_layers)],
            'final_norm': create_norm_params()
        },
        'decoder': {
            'layers': [{'self_attention': create_attention_params(), 'cross_attention': create_attention_params(), 'feed_forward': create_ffn_params(), 'norm1': create_norm_params(), 'norm2': create_norm_params(), 'norm3': create_norm_params()} for _ in range(n_layers)],
            'final_norm': create_norm_params()
        },
        'output': {'W_out': _init(jax.nn.initializers.xavier_normal(), (d_model, vocab_size))}
    }

# --- TRAINING & UTILS ---

def text_to_token_ids(sentences: List[str], vocab: Dict[str, int], max_len: int = 32) -> jnp.ndarray:
    """Correctly tokenizes sentences for inference, handling unknown words."""
    unk_id = vocab['<UNK>']
    pad_id = vocab['<PAD>']
    
    all_token_ids = []
    for sentence in sentences:
        # Use the same normalization as training
        words = normalize_text(sentence).split()
        token_ids = [vocab.get(word, unk_id) for word in words]
        
        if len(token_ids) >= max_len:
            padded_ids = token_ids[:max_len]
        else:
            padded_ids = token_ids + [pad_id] * (max_len - len(token_ids))
        all_token_ids.append(padded_ids)
            
    return jnp.array(all_token_ids, dtype=jnp.int32)

def loss_fn(params, batch_input, batch_targets, vocab, dropout_rate, key, training, **kwargs):
    sos_id = vocab['<SOS>']
    dec_input = jnp.concatenate([jnp.full((batch_targets.shape[0], 1), sos_id), batch_targets[:, :-1]], axis=1)
    dec_target = batch_targets

    logits, _ = forward(params, batch_input, dec_input,
                    training=training,
                    dropout_rate=dropout_rate,
                    key=key,
                    vocab=vocab,
                    **kwargs)
    
    # loss = optax.softmax_cross_entropy_with_integer_labels(logits, dec_target)
    loss = smoothed_loss(logits, dec_target, vocab)
    mask = (batch_targets != vocab['<PAD>'])
    
    return jnp.sum((loss * mask) / (jnp.sum(mask) + 1e-8))

def compute_accuracy(predictions, targets, vocab):
    mask = (targets != vocab['<PAD>'])
    accuracy = jnp.sum((predictions == targets) * mask) / (jnp.sum(mask) + 1e-8)
    return accuracy.item()

def smoothed_loss(logits, targets, vocab, smoothing=0.1):
    num_classes = logits.shape[-1]
    confidence = 1.0 - smoothing
    low_conf = smoothing / num_classes
    targets_onehot = jax.nn.one_hot(targets, num_classes)
    soft_targets = confidence * targets_onehot + low_conf
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(soft_targets * log_probs, axis=-1)
    mask = (targets != vocab['<PAD>'])
    return jnp.sum(loss * mask) / jnp.sum(mask)

def quick_eval(n=5):
    print("\n📊 Quick Eval on random samples\n")
    it = get_data_iterator(train_en_tok, train_ru_tok, 1, jax.random.PRNGKey(123))
    for _ in range(n):
        x, y = next(it)
        sos_id = vocab["<SOS>"]
        dec_input = jnp.concatenate([jnp.full((1, 1), sos_id), y[:, :-1]], axis=1)
        logits, preds = forward(params, x, dec_input, training=False, dropout_rate=0.0, **model_args)
        inp = " ".join([id2tok.get(int(i), "") for i in x[0] if id2tok.get(int(i)) not in ("<PAD>", "<EOS>")])
        out = " ".join([id2tok.get(int(i), "") for i in preds[0] if id2tok.get(int(i)) not in ("<PAD>", "<EOS>")])
        print(f"🟦 {inp}\n🟨 {out}\n{'─'*60}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Workaround for JAX Metal initialization error
    try:
        print(f"JAX backend targeting: {jax.default_backend()}")
        _ = jax.devices()
        print(f"JAX devices initialized: {jax.devices()}")
    except Exception as e:
        print(f"Could not initialize JAX GPU backend: {e}. Falling back to CPU.")

    # Hyperparameters
    MAX_STEPS = 2000
    D_MODEL = 256
    D_FF = D_MODEL * 4
    DROPOUT_RATE = 0.1
    N_LAYERS = 4
    N_HEADS = 8
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    MAX_LEN = 32

    # Testing Hyperparameters
    # MAX_STEPS = 5           # ✅ only 5 steps
    # BATCH_SIZE = 4          # ✅ tiny batch
    # MAX_LEN = 12            # ✅ shorter sentences
    # D_MODEL = 64            # ✅ smaller model for faster compile
    # N_LAYERS = 1            # ✅ one encoder, one decoder
    # N_HEADS = 4             # ✅ simpler attention

    
    print(f"""
    --- Model Hyperparameters ---
    Max Steps:              {MAX_STEPS}
    Model Dim (d_model):    {D_MODEL}
    FFN Dim (d_ff):         {D_FF}
    Dropout Rate:           {DROPOUT_RATE}
    Encoder/Decoder Layers: {N_LAYERS}
    Attention Heads:        {N_HEADS}
    Learning Rate:          {LEARNING_RATE}
    Batch Size:             {BATCH_SIZE}
    ---------------------------
    """)

    # --- Setup ---
    key = jax.random.PRNGKey(42)
    key, data_key, params_key = jax.random.split(key, 3)
    
    vocab, en_sentences, ru_sentences, vocab_size = load_dataset_and_vocab(max_vocab_size=20000)
    ru_sentences = en_sentences.copy()
    
    train_size = int(0.8 * len(en_sentences))
    train_en_sents, val_en_sents = en_sentences[:train_size], en_sentences[train_size:]
    train_ru_sents, val_ru_sents = ru_sentences[:train_size], ru_sentences[train_size:]

    # testing
    # MAX_STEPS = 3000
    train_en_sents = train_en_sents[:1000]
    train_ru_sents = train_ru_sents[:1000]
    val_en_sents = val_en_sents[:200]
    val_ru_sents = val_ru_sents[:200]
    
    train_en_tok = tokenize_and_pad(train_en_sents, vocab, MAX_LEN)
    train_ru_tok = tokenize_and_pad(train_ru_sents, vocab, MAX_LEN)
    val_en_tok = tokenize_and_pad(val_en_sents, vocab, MAX_LEN)
    val_ru_tok = tokenize_and_pad(val_ru_sents, vocab, MAX_LEN)

    # # --- Synthetic Copy Task ---
    # toy_sentences = ["i am happy", "he is good", "they are here", "we are fine"]
    # train_en_sents = toy_sentences
    # train_ru_sents = toy_sentences  # identical targets

    # vocab = {'<PAD>':0, '<UNK>':1, '<EOS>':2, '<SOS>':3,
    #         'i':4, 'am':5, 'happy':6, 'he':7, 'is':8, 'good':9,
    #         'they':10, 'are':11, 'here':12, 'we':13, 'fine':14}
    # vocab_inv = {v: k for k, v in vocab.items()}
    # vocab_size = len(vocab)

    # train_en_tok = tokenize_and_pad(train_en_sents, vocab, max_len=8)
    # train_ru_tok = tokenize_and_pad(train_ru_sents, vocab, max_len=8)
    # val_en_tok, val_ru_tok = train_en_tok, train_ru_tok

    # import numpy as np

    # VOCAB_SIZE = 50
    # SEQ_LEN = 8
    # N_SAMPLES = 2000

    # np.random.seed(0)
    # inputs = np.random.randint(4, VOCAB_SIZE, size=(N_SAMPLES, SEQ_LEN))
    # targets = inputs.copy()

    # # Replace a few with <EOS>/<PAD>
    # targets[:, -1] = 2  # EOS
    # vocab = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2, '<SOS>': 3}
    # vocab.update({str(i): i for i in range(4, VOCAB_SIZE)})

    # train_en_tok = jnp.array(inputs[:1600])
    # train_ru_tok = jnp.array(targets[:1600])
    # val_en_tok = jnp.array(inputs[1600:])
    # val_ru_tok = jnp.array(targets[1600:])
    
    
    def lr_schedule(step, base_lr=3e-4, warmup=1000):
        step = jnp.maximum(1, step)
        return base_lr * jnp.minimum(step / warmup, 1.0)

    # --- Model and Optimizer ---
    model_args = {'vocab_size': vocab_size, 'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_heads': N_HEADS, 'd_ff': D_FF}
    params = init_params(params_key, vocab_size=vocab_size, d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS, n_layers=N_LAYERS)
    
    # ---------- Replace your optimizer init with this ----------
    warmup_steps = 1000  # choose warmup length (tuneable)
    base_lr = 3e-4       # recommended base LR

    # create an optax schedule and optimizer ONCE
    # schedule = optax.linear_schedule(init_value=0.0, end_value=base_lr, transition_steps=warmup_steps)
    # schedule = optax.cosine_decay_schedule(init_value=3e-4, decay_steps=10000)
    schedule = optax.linear_schedule(3e-4, 1e-5, 5000)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    # ---------- Replace your train_step with this ----------
    @jax.jit
    def train_step(params, opt_state, batch_input, batch_targets, key, global_step, vocab):
        """
        Single training step using the pre-created `optimizer` (with schedule).
        Returns (params, opt_state, loss, new_key, global_step+1, lr)
        """
        step_key, new_key = jax.random.split(key)

        # compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(
            params, batch_input, batch_targets, vocab, DROPOUT_RATE, step_key, training=True, **model_args
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # compute lr for logging: schedule returns a jnp scalar
        lr = schedule(opt_state.count)

        return new_params, new_opt_state, loss, new_key, global_step + 1, lr
    
    # @jax.jit
    def eval_step(params, batch_input, batch_targets):
        sos_id = vocab['<SOS>']
        dec_input = jnp.concatenate([jnp.full((batch_targets.shape[0], 1), sos_id), batch_targets[:, :-1]], axis=1)
        
        logits, predictions = forward(
            params, batch_input, dec_input, vocab=vocab, training=False, dropout_rate=0.0, **model_args, key=None
        )
        accuracy = compute_accuracy(predictions, batch_targets, vocab)
        return accuracy

    # --- Training Loop ---
    global_step = 0
    train_losses, val_accuracies = [], []
    key = jax.random.PRNGKey(42)
    train_key = key
    data_iter = get_data_iterator(train_en_tok, train_ru_tok, BATCH_SIZE, data_key)

    pbar = tqdm(range(MAX_STEPS), desc="Training Steps")
    for step in pbar:
        key, train_key = jax.random.split(key)
        batch_input, batch_targets = next(data_iter)
        # batch_targets = batch_targets.at[:, -1].set(vocab["<EOS>"])

        # print("Sample target:", batch_targets[0])

        params, opt_state, loss, key, global_step, lr = train_step(
            params, opt_state, batch_input, batch_targets, train_key, global_step, vocab
        )

        if step == 0:
            print("\nSample input:", batch_input[0])
            print("Sample target:", batch_targets[0])
            print(f"Initial loss: {float(loss):.4f}")

        # if (step + 1) % 1 == 0:
        #     print(f"Step {step+1}: loss={float(loss):.4f}, lr={float(lr):.6f}")

        train_losses.append(float(loss))

        # ---- Every 50 steps: run validation ----
        if (step + 1) % 50 == 0:
            val_accuracies_batch = []
            val_iter = get_data_iterator(val_en_tok, val_ru_tok, BATCH_SIZE, data_key)
            num_val_batches = len(val_en_tok) // BATCH_SIZE

            if num_val_batches == 0:
                print("⚠️ No validation batches available — skipping eval")
                continue

            for _ in range(num_val_batches):
                val_batch_input, val_batch_targets = next(val_iter)
                v_acc = eval_step(params, val_batch_input, val_batch_targets)
                val_accuracies_batch.append(v_acc)

            avg_val_accuracy = jnp.mean(jnp.array(val_accuracies_batch))
            val_accuracies.append(float(avg_val_accuracy))

            pbar.set_postfix({
                "Step": step + 1,
                "LR": f"{float(lr):.6f}",
                "Train Loss": f"{float(loss):.4f}",
                "Val Accuracy": f"{float(avg_val_accuracy):.4f}"
            })

    print("\n--- Training Complete ---")
    
    # --- Save and Plot ---
    byte_data = to_bytes(params)
    with open("transformer_weights.msgpack", "wb") as f:
        f.write(byte_data)
    print("✅ Model weights saved to transformer_weights.msgpack")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(train_losses, label='Train Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('Val Accuracy', color=color)
    ax2.plot(range(50, MAX_STEPS + 1, 50), val_accuracies, label='Val Accuracy', color='tab:purple', marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    plt.title(f'Training Metrics Over {MAX_STEPS} Steps')
    fig.tight_layout()

    import textwrap

    def pretty_print_example(input_ids, target_ids, pred_ids, id2tok, max_len=20):
        """Nicely format a single example."""
        def decode(ids):
            toks = [id2tok.get(int(i), "<UNK>") for i in ids if id2tok.get(int(i)) not in ("<PAD>", "<SOS>", "<EOS>")]
            return " ".join(toks[:max_len])

        input_txt  = decode(input_ids)
        target_txt = decode(target_ids)
        pred_txt   = decode(pred_ids)

        print("─" * 80)
        print(f"🟦 INPUT :  {textwrap.fill(input_txt,  70)}")
        print(f"🟩 TARGET:  {textwrap.fill(target_txt, 70)}")
        print(f"🟨 PRED  :  {textwrap.fill(pred_txt,   70)}")
        print("─" * 80)
        print()

    # ------------------------------------------------------------------------------
    # 1️⃣ Run a few random training examples
    # ------------------------------------------------------------------------------
    data_iter = get_data_iterator(train_en_tok, train_ru_tok, 1)
    id2tok = {v: k for k, v in vocab.items()}

    print("\n📘 RANDOM TRAINING EXAMPLES (Input → Target → Prediction)\n")

    for _ in range(5):
        x, y = next(data_iter)
        sos_id = vocab["<SOS>"]
        dec_input = jnp.concatenate([jnp.full((1, 1), sos_id), y[:, :-1]], axis=1)

        logits, preds = forward(params, x, dec_input, training=False, dropout_rate=0.0, key=None, vocab=vocab, **model_args)
        pretty_print_example(x[0], y[0], preds[0], id2tok)

    # ------------------------------------------------------------------------------
    # 2️⃣ Custom test sentences
    # ------------------------------------------------------------------------------
    print("\n📗 CUSTOM TEST SENTENCES\n")

    test_sentences = ["i am", "she said", "he went to the city", "they are coming home"]
    test_tokens = text_to_token_ids(test_sentences, vocab, max_len=MAX_LEN)

    for sent, toks in zip(test_sentences, test_tokens):
        dec_input = jnp.full((1, 1), vocab["<SOS>"], dtype=jnp.int32)
        logits, preds = forward(params, toks[None, :], dec_input,
                                training=False, dropout_rate=0.0,
                                key=None, vocab=vocab, **model_args)
        # decoded = " ".join([id2tok.get(int(i), "<UNK>") for i in preds[0]
        #                     if id2tok.get(int(i)) not in ("<PAD>", "<SOS>", "<EOS>")])
        decoded = []
        for t in preds[0]:
            word = id2tok[int(t)]
            if word == "<EOS>":
                break
            if word not in ("<PAD>", "<SOS>"):
                decoded.append(word)
        print(" ".join(decoded))
        print(f"💬 {sent:<30} →  {decoded}")


    # quick_eval()

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, f'metrics_plot_{MAX_STEPS}_steps_{timestamp_str}.png')
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()

