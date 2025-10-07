import jax
import jax.numpy as jnp
from typing import List, Dict
from transformer import text_to_token_ids, token_embeddings, positional_embeddings, combine_embeddings

def test_text_to_token_ids() -> bool:
    """Tester function for text_to_token_ids. Returns True if all tests pass."""
    
    # Test case 1: Simple batch with short sentences
    vocab1 = {'the': 1, 'cat': 2, 'runs': 3, 'dog': 4, 'sleeps': 5}
    sentences1 = ['the cat runs', 'the dog sleeps']
    result1 = text_to_token_ids(sentences1, vocab1)
    expected_shape1 = (2, 32)
    test1_pass = (
        result1.shape == expected_shape1
        and result1.dtype == jnp.int32
        and jnp.all(result1[0, :3] == jnp.array([1, 2, 3]))
        and jnp.all(result1[1, :3] == jnp.array([1, 4, 5]))
        and jnp.all(result1[:, 3:] == 0)
    )
    
    # Test case 2: Batch size 16 with varying lengths
    vocab2 = {'hello': 10, 'world': 11, 'test': 12, 'longer': 13, 'sentence': 14}
    sentences2 = ['hello world'] * 15 + ['hello world test longer sentence']
    result2 = text_to_token_ids(sentences2, vocab2)
    expected_shape2 = (16, 32)
    test2_pass = (
        result2.shape == expected_shape2
        and jnp.all(result2[0, :2] == jnp.array([10, 11]))
        and jnp.all(result2[15, :5] == jnp.array([10, 11, 12, 13, 14]))
        and jnp.all(result2[:, 5:] == 0)  # Pads after 5 tokens in last sentence
    )
    
    # Test case 3: Empty sentence (all padding)
    vocab3 = {'pad': 0}  # Padding token
    sentences3 = ['']
    result3 = text_to_token_ids(sentences3, vocab3)
    test3_pass = (
        result3.shape == (1, 32)
        and jnp.all(result3 == 0)
    )
    
    # Test case 4: Sentence longer than max_len (tests full tokenization + pad; add slicing for truncation if needed)
    vocab4 = {'a': 1, 'b': 2, 'c': 3}
    long_sentence = 'a b c a b c ' * 20  # >32 tokens
    sentences4 = [long_sentence]
    result4 = text_to_token_ids(sentences4, vocab4, max_len=5)  # Small max_len for test
    test4_pass = (
        result4.shape == (1, 5)
        and jnp.all(result4[0, :5] == jnp.array([1, 2, 3, 1, 2]))  # First 5 tokens
    )
    
    # Aggregate results
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    print(f"All tests passed: {all_pass}")
    print(f"Test 1 (simple batch): {test1_pass}")
    print(f"Test 2 (batch size 16): {test2_pass}")
    print(f"Test 3 (empty sentence): {test3_pass}")
    print(f"Test 4 (long sentence, max_len=5): {test4_pass}")
    
    return all_pass


def test_token_embeddings():
    vocab_size, d_model = 100, 64
    token_ids = jnp.zeros((16, 32), dtype=jnp.int32)  # Dummy input
    embed_matrix = jnp.ones((vocab_size, d_model), dtype=jnp.float32)
    result = token_embeddings(token_ids, embed_matrix)
    return (
        result.shape == (16, 32, d_model)
        and result.dtype == jnp.float32
    )

def test_token_embeddings_varied():
    token_ids = jnp.array([[0, 1], [2, 3]])
    embed_matrix = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    result = token_embeddings(token_ids, embed_matrix, d_model=2)
    expected = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    return result.shape == (2, 2, 2) and jnp.all(result == expected)

def test_positional_embeddings():
    pos_emb = positional_embeddings()
    return (
        pos_emb.shape in [(32, 64), (1, 32, 64)]
        and pos_emb.dtype == jnp.float32
    )

def test_combine_embeddings():
    """Tests combine_embeddings with dummy inputs."""
    batch_size, max_len, d_model = 2, 4, 8  # Small sizes for easy checking
    # Dummy token embeddings (all 1s)
    token_emb = jnp.ones((batch_size, max_len, d_model), dtype=jnp.float32)
    # Positional embeddings from your function
    pos_emb = positional_embeddings(max_len, d_model)
    
    # Combine
    combined = combine_embeddings(token_emb, pos_emb)
    
    # Checks
    shape_check = combined.shape == (batch_size, max_len, d_model)
    dtype_check = combined.dtype == jnp.float32
    # Sample addition: first element should be 1 + pos_emb[0, 0, 0]
    addition_check = combined[0, 0, 0] == token_emb[0, 0, 0] + pos_emb[0, 0, 0]
    
    print(f"token_emb shape: {token_emb.shape}")
    print(f"pos_emb shape: {pos_emb.shape}")
    print(f"combined shape: {combined.shape}")
    print(f"combined dtype: {combined.dtype}")
    print(f"Shape check: {shape_check}")
    print(f"Dtype check: {dtype_check}")
    print(f"Addition check (first element): {addition_check}")
    
    return shape_check and dtype_check and addition_check

# Run the tester
if __name__ == "__main__":
    # test_text_to_token_ids()
    print(test_token_embeddings())
    print(test_token_embeddings_varied())
    print(test_positional_embeddings())
    test_combine_embeddings()