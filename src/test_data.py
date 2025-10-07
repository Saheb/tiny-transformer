import jax.numpy as jnp
from typing import Dict, List
from data import build_vocab

def test_build_vocab():
    """Runs all unit tests for build_vocab."""
    
    # Test 1: Basic pair with special tokens
    en1 = ["the cat runs"]
    ru1 = ["кот бежит"]
    vocab1 = build_vocab(en1, ru1)
    test1_pass = (
        len(vocab1) == 9  # 4 special + 5 unique tokens ("the", "cat", "runs", "кот", "бежит")
        and vocab1["<pad>"] == 0
        and vocab1["<unk>"] == 1
        and vocab1["<sos>"] == 2
        and vocab1["<eos>"] == 3
        and "the" in vocab1 and "кот" in vocab1
        and all(isinstance(id, int) for id in vocab1.values())
    )
    
    # Test 2: Multiple pairs with duplicates
    en2 = ["the cat runs", "a dog"]
    ru2 = ["кот бежит", "собака"]
    vocab2 = build_vocab(en2, ru2)
    test2_pass = (
        len(vocab2) == 12  # 4 special + 8 unique tokens ("the", "cat", "runs", "a", "dog", "кот", "бежит", "собака")
        and vocab2["<pad>"] == 0
        and "a" in vocab2 and "собака" in vocab2
        and len(set(vocab2.values()) - {0, 1, 2, 3}) == 8  # 8 unique token IDs
    )
    
    # Test 3: Empty inputs
    en3 = []
    ru3 = []
    vocab3 = build_vocab(en3, ru3)
    test3_pass = (
        len(vocab3) == 4  # Only special tokens
        and vocab3["<pad>"] == 0
        and vocab3["<unk>"] == 1
        and vocab3["<sos>"] == 2
        and vocab3["<eos>"] == 3
    )
    
    # Test 4: Single Russian sentence with Cyrillic
    en4 = [""]
    ru4 = ["котик маленький"]
    vocab4 = build_vocab(en4, ru4)
    test4_pass = (
        len(vocab4) == 6  # 4 special + 2 tokens ("котик", "маленький")
        and vocab4["<pad>"] == 0
        and "котик" in vocab4 and "маленький" in vocab4
        and vocab4["котик"] != vocab4["маленький"]  # Distinct IDs
    )
    
    # Aggregate results
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    print(f"All tests passed: {all_pass}")
    print(f"Test 1 (basic pair): {test1_pass}")
    print(f"Test 2 (multiple pairs): {test2_pass}")
    print(f"Test 3 (empty inputs): {test3_pass}")
    print(f"Test 4 (Russian only): {test4_pass}")
    
    return all_pass

# Run the tests
if __name__ == "__main__":
    test_build_vocab()