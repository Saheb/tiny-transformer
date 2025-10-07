from typing import Dict, List
import jax.numpy as jnp

def build_vocab(en_sentences: List[str], ru_sentences: List[str]) -> Dict[str, int]:
    """Builds a vocabulary from English and Russian sentence pairs.
    
    Args:
        en_sentences: List of English sentences from the dataset.
        ru_sentences: List of Russian sentences paired with English.
    
    Returns:
        Dict[str, int]: Vocabulary mapping tokens to unique IDs, including special tokens.
    """
    # Add special tokens: <pad>, <unk>, <sos>, <eos>
    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    counter = 3
    # Extract unique tokens from both languages
    sentences = en_sentences + ru_sentences
    for sentence in sentences:
        words = sentence.split(' ')
        for word in words:
            if word != '' and word not in vocab:
                vocab[word] = counter + 1
                counter = counter + 1

    print(vocab)
    return vocab

from datasets import load_dataset
import numpy as np
from transliterate import translit

def normalize_text(text):
    """Normalize text by transliterating Cyrillic to Latin and converting to lowercase."""
    return translit(text, 'ru', reversed=True).lower()

# def normalize_text(text):
#     """
#     Normalize text by transliterating, converting to lowercase, 
#     and removing punctuation.
#     """
#     # Transliterate and convert to lowercase
#     normalized = translit(text, 'ru', reversed=True).lower()
#     # Remove any characters that are not letters, numbers, or apostrophes
#     normalized = re.sub(r"[^a-zA-Zа-яА-Я0-9']+", " ", normalized)
#     return normalized.strip()

# from datasets import load_dataset

def load_dataset_and_vocab(split="train", max_vocab_size=20000):
    """Load dataset and build a shared vocabulary from en-ru pairs."""
    dataset = load_dataset("opus_books", "en-ru", split=split)
    print(f"Loaded {dataset.num_rows} samples")
    print("Sample pair:", dataset[0]['translation'])

    # Extract and normalize sentences
    en_sentences = [normalize_text(pair['translation']['en']) for pair in dataset]
    ru_sentences = [normalize_text(pair['translation']['ru']) for pair in dataset]

    # Build shared vocab with <SOS> as a special token
    all_tokens = ' '.join(en_sentences + ru_sentences).split()
    vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3}
    vocab.update({token: i + 4 for i, token in enumerate(set(all_tokens))})
    vocab_size = min(len(vocab), max_vocab_size)
    print(f"Vocab size: {vocab_size} (capped at {max_vocab_size})")
    return vocab, en_sentences, ru_sentences, vocab_size

# In data.py

import re
from collections import Counter
from datasets import load_dataset

# def load_dataset_and_vocab(split="train", max_vocab_size=20000):
#     """
#     Loads dataset from Hugging Face and builds a stable, correctly-sized vocabulary.
#     """
#     dataset = load_dataset("opus_books", "en-ru", split=split)
#     en_sentences = [normalize_text(pair['en']) for pair in dataset['translation']]
#     ru_sentences = [normalize_text(pair['ru']) for pair in dataset['translation']]

#     # FIX: Using consistent lowercase special tokens
#     special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
#     vocab = {token: i for i, token in enumerate(special_tokens)}
    
#     # ... (rest of the function is the same)
#     word_counts = Counter(' '.join(en_sentences + ru_sentences).split())
#     for word, _ in word_counts.most_common(max_vocab_size - len(special_tokens)):
#         if word not in vocab:
#             vocab[word] = len(vocab)
            
#     return vocab, en_sentences, ru_sentences, len(vocab)

def tokenize_dataset(sentences, vocab, batch_size=16, max_len=32, pad_token=0, vocab_size=10000):
    """Tokenize and pad all sentences into an array with index capping."""
    tokenized = []
    for sent in sentences:  # Process all sentences, not just batch_size
        tokens = normalize_text(sent).split()[:max_len - 1]
        tokens.append('<eos>')
        padded = [min(vocab.get(token, vocab['<unk>']), vocab_size - 1) for token in tokens] + [pad_token] * (max_len - len(tokens))
        tokenized.append(padded)
    return np.array(tokenized, dtype=np.int32)

# def tokenize_and_pad(sentences: List[str], vocab: Dict[str, int], max_len: int) -> jnp.ndarray:
#     """A single, unified function to tokenize and pad sentences."""
#     # FIX: Using consistent lowercase special tokens
#     unk_id = vocab['<unk>']
#     pad_id = vocab['<pad>']
#     eos_id = vocab['<eos>']
    
#     all_token_ids = []
#     for sentence in sentences:
#         words = sentence.split()
#         token_ids = [vocab.get(word, unk_id) for word in words]
#         token_ids.append(eos_id)
        
#         if len(token_ids) >= max_len:
#             padded_ids = token_ids[:max_len]
#         else:
#             padded_ids = token_ids + [pad_id] * (max_len - len(token_ids))
        
#         all_token_ids.append(padded_ids)
            
#     return jnp.array(all_token_ids, dtype=jnp.int32)

def generate_targets(tokenized, vocab, pad_token=0):
    """Generate shifted targets for autoregressive prediction."""
    targets = np.zeros_like(tokenized)
    targets[:, :-1] = tokenized[:, 1:]  # Shift right to predict next token
    targets[:, -1] = pad_token  # Last token as <pad>
    return targets

def get_data_iterator(en_tokenized, ru_targets, batch_size=16):
    while True:
        for i in range(0, len(en_tokenized), batch_size):
            batch_input = en_tokenized[i:i + batch_size]
            batch_targets = ru_targets[i:i + batch_size]
            # print(f"Yielding batch at index {i}, input shape: {batch_input.shape}, target shape: {batch_targets.shape}")
            yield batch_input, batch_targets

# Example usage (can be removed or kept for testing)
if __name__ == "__main__":
    vocab, en_sentences, ru_sentences, vocab_size = load_dataset_and_vocab()
    en_tokenized = tokenize_dataset(en_sentences, vocab)
    ru_tokenized = tokenize_dataset(ru_sentences, vocab)
    ru_targets = generate_targets(ru_tokenized, vocab)
    data_iter = get_data_iterator(en_tokenized, ru_targets)

    print("en_tokenized shape:", en_tokenized.shape)
    print("Sample en:", en_tokenized[0])
    print("ru_targets shape:", ru_targets.shape)
    print("Sample ru_targets:", ru_targets[0])