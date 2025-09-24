import numpy as np
import aie.iron as iron
from ml_dtypes import bfloat16
from aie.iron.functional import plus

# Set seeds for reproducibility
np.random.seed(42)

def test_transform_addition(size=64, dtype=np.int16):
    """Test iron.transform with addition operation."""
    
    print("Testing iron.transform with Addition")
    print("=" * 40)
    
    # Create test tensors
    device = "npu"
    
    # Create two input tensors
    tensor1 = iron.tensor(
        np.arange(1, size + 1, dtype=dtype),
        dtype=dtype,
        device=device
    )
    
    tensor2 = iron.tensor(
        np.arange(1, size + 1, dtype=dtype),
        dtype=dtype,
        device=device
    )   
    
    # Create output tensor
    output = iron.tensor(
        np.zeros(size, dtype=dtype),
        dtype=dtype,
        device=device
    )
    
    print(f"Input tensor 1: {tensor1.numpy()}")
    print(f"Input tensor 2: {tensor2.numpy()}")
    print(f"Output tensor (before): {output.numpy()}")
    
    # Apply transform with addition
    iron.transform(tensor1, tensor2, output, plus)
    
    print(f"Output tensor (after): {output.numpy()}")
    
    # Verify results
    expected = np.arange(1, size + 1, dtype=dtype) + np.arange(1, size + 1, dtype=dtype)
    actual = output.numpy()
    
    print(f"Expected result: {expected}")
    print(f"Actual result:   {actual}")
    
    # Check if results match
    if np.allclose(actual, expected):
        print("✅ Test PASSED: Results match expected values!")
    else:
        print("❌ Test FAILED: Results don't match expected values!")
    
    return np.allclose(actual, expected)


if __name__ == "__main__":
    # Run tests
    test1_passed = test_transform_addition(size=32, dtype=np.int16)
    print(f"\nTest Summary:")
    print(f"test: {'PASSED' if test1_passed else 'FAILED'}")
    
    if test1_passed:
        print("🎉 All tests PASSED!")
    else:
        print("💥 Some tests FAILED!")
