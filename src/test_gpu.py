import jax
import jax.numpy as jnp
import time

print("--- JAX GPU Test ---")

# 1. ADD THIS BLOCK to explicitly initialize and verify the JAX backend
try:
    print(f"JAX backend targeting: {jax.default_backend()}")
    devices = jax.devices()
    print(f"JAX devices found: {devices}")
    
    if not any('METAL' in str(d).upper() or 'GPU' in str(d).upper() for d in devices):
        print("\n⚠️ WARNING: JAX is not using the GPU. It's using the CPU.")
        exit()
    else:
        print(f"\n✅ Success! JAX is configured to use the device: {devices[0]}")

except Exception as e:
    print(f"An error occurred during JAX initialization: {e}")
    exit()
# --- END of added block ---


# 2. Define a large computation (matrix multiplication)
MATRIX_SIZE = 4096
# Now that the backend is warm, this line should work without error
key = jax.random.PRNGKey(0)

# Create two large random matrices on the device
print(f"\nCreating two {MATRIX_SIZE}x{MATRIX_SIZE} matrices...")
x = jax.random.normal(key, (MATRIX_SIZE, MATRIX_SIZE))
y = jax.random.normal(key, (MATRIX_SIZE, MATRIX_SIZE))

# 3. Define a JIT-compiled function for the computation
@jax.jit
def multiply_matrices(a, b):
    return jnp.dot(a, b)

# 4. Run and time the computation
print("Warming up the JIT compiler (first run will be slower)...")
result = multiply_matrices(x, y)
result.block_until_ready()

print("\nRunning benchmark...")
start_time = time.time()

result = multiply_matrices(x, y)
result.block_until_ready()

end_time = time.time()
duration = end_time - start_time

print(f"✅ Large matrix multiplication ({MATRIX_SIZE}x{MATRIX_SIZE}) completed in: {duration:.4f} seconds.")
print("\nIf this time is very short (typically < 1 second), your GPU is working correctly.")