import numpy as np
import jax.numpy as jnp
import jax
import re
from collections import Counter
from datasets import load_dataset
from typing import List, Dict
from transliterate import translit

def normalize_text(text):
    """
    Normalizes text by transliterating, converting to lowercase, and removing punctuation.
    """
    normalized = translit(text, 'ru', reversed=True).lower()
    normalized = re.sub(r"[^a-zA-Zа-яА-Я0-9']+", " ", normalized)
    return normalized.strip()

def load_dataset_and_vocab(split="train", max_vocab_size=20000):
    """
    Loads dataset from Hugging Face and builds a stable, correctly-sized vocabulary.
    """
    dataset = load_dataset("opus_books", "en-ru", split=split)
    print(f"Loaded {dataset.num_rows} samples for split: {split}")

    en_sentences = [normalize_text(pair['en']) for pair in dataset['translation']]
    ru_sentences = [normalize_text(pair['ru']) for pair in dataset['translation']]

    # Define special tokens with consistent, uppercase names
    special_tokens = ['<PAD>', '<UNK>', '<EOS>', '<SOS>']
    vocab = {token: i for i, token in enumerate(special_tokens)}
    
    # Use Counter for a stable, frequency-based vocabulary
    all_tokens = ' '.join(en_sentences + ru_sentences).split()
    word_counts = Counter(all_tokens)
    
    # Correctly cap the vocabulary by adding the most common words
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
        words = sentence.split() # Sentences are already normalized
        token_ids = [vocab.get(word, unk_id) for word in words]
        
        # Add End-of-Sequence token
        token_ids.append(eos_id)
        
        # Pad or truncate
        if len(token_ids) >= max_len:
            padded_ids = token_ids[:max_len]
        else:
            padded_ids = token_ids + [pad_id] * (max_len - len(token_ids))
        
        all_token_ids.append(padded_ids)
            
    return jnp.array(all_token_ids, dtype=jnp.int32)

def get_data_iterator(en_tokenized, ru_targets, batch_size=16, key=None):
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