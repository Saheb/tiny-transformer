# ----------------------------------------------------------------------------
# FILE: data.py
# Data loading, normalization, tokenization, iterators
# ----------------------------------------------------------------------------
import re
from collections import Counter
from datasets import load_dataset
from typing import List, Dict
import jax
import jax.numpy as jnp


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9']+", " ", text.lower()).strip()


def load_dataset_and_vocab(split: str = "train", max_vocab_size: int = 20000, subset_pct: str = "train[:1%]"):
    dataset = load_dataset("wmt14", "ru-en", split=subset_pct)
    print(f"Loaded {dataset.num_rows} samples for split: {subset_pct}")

    en_sentences = [normalize_text(p['en']) for p in dataset['translation']]
    ru_sentences = [normalize_text(p['ru']) for p in dataset['translation']]

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
    unk_id = vocab['<UNK>']
    pad_id = vocab['<PAD>']
    eos_id = vocab['<EOS>']
    out = []
    for s in sentences:
        toks = [vocab.get(w, unk_id) for w in s.split()]
        toks.append(eos_id)
        if len(toks) >= max_len:
            toks = toks[:max_len]
        else:
            toks = toks + [pad_id] * (max_len - len(toks))
        out.append(toks)
    return jnp.array(out, dtype=jnp.int32)


def get_data_iterator(en_tokenized, ru_targets, batch_size, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    dataset_size = len(en_tokenized)
    assert dataset_size == len(ru_targets)
    while True:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, dataset_size)
        for i in range(0, dataset_size, batch_size):
            batch_idx = perm[i:i+batch_size]
            yield en_tokenized[batch_idx], ru_targets[batch_idx]


def text_to_token_ids(sentences: List[str], vocab: Dict[str, int], max_len: int = 32) -> jnp.ndarray:
    unk = vocab['<UNK>']; pad = vocab['<PAD>']; eos = vocab['<EOS>']
    out = []
    for s in sentences:
        toks = [vocab.get(w, unk) for w in normalize_text(s).split()]
        if len(toks) >= max_len:
            toks = toks[:max_len]
        else:
            toks = toks + [pad] * (max_len - len(toks))
        out.append(toks)
    return jnp.array(out, dtype=jnp.int32)
