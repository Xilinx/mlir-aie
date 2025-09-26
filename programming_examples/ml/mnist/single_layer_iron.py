import numpy as np
import aie.iron as iron
from ml_dtypes import bfloat16
from aie.iron.functional import identity, relu
from aie.iron.algorithms import for_each
from aie.iron.graph import capture_graph
import time
from utilities import do_bench

# Set seeds for reproducibility
np.random.seed(42)

class SingleLayerMNIST:
    """
    Single-layer classifier following PyTorch-style abstraction.
    
    Architecture: Flatten -> Dense -> Output
    Dimensions adjusted to work with iron.matmul constraints.
    """
    
    def __init__(self, input_size=768, output_size=32, dtype=bfloat16, device="npu"):
        """
        Initialize the single-layer model.
        
        Args:
            input_size (int): Number of input features (must be multiple of 64)
            output_size (int): Number of output classes (must be multiple of 32)
            dtype: Data type for weights and computations (default: bfloat16)
            device (str): Device to run on ('npu' or 'cpu')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype
        self.device = device
        
        # Initialize weights and bias
        self._initialize_parameters()
        
        print(f"SingleLayerMNIST initialized:")
        print(f"  Input size: {self.input_size}")
        print(f"  Output size: {self.output_size}")
        print(f"  Data type: {self.dtype}")
        print(f"  Device: {self.device}")
        print(f"  Weight shape: {self.weight.shape}")
        print(f"  Bias shape: {self.bias.shape}")
    
    def _initialize_parameters(self):
        """Initialize model parameters with random weights."""
        # Create weight matrix: (input_size, output_size)
        self.weight = iron.tensor(
            np.random.randn(self.input_size, self.output_size).astype(self.dtype) * 0.01,
            dtype=self.dtype,
            device=self.device
        )
        
        # Create bias vector: (output_size,)
        self.bias = iron.tensor(
            np.zeros(self.output_size, dtype=self.dtype),
            dtype=self.dtype,
            device=self.device
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (iron.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            iron.Tensor: Output logits (raw scores) of shape (batch_size, output_size)
        """
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        batch_size = x.shape[0]
        
        # Create output tensor for matmul result
        logits = iron.tensor(
            np.zeros((batch_size, self.output_size), dtype=self.dtype),
            dtype=self.dtype,
            device=self.device
        )
        
        # Matrix multiplication: x @ weight
        # (batch_size, input_size) @ (input_size, output_size) -> (batch_size, output_size)
        iron.matmul(x, self.weight, out=logits)
        
        # Apply ReLU activation to each element
        relu_func = relu(self.dtype)
        for_each(logits.view(-1), relu_func, out=logits.view(-1))
        
        return logits
    
    def parameters(self):
        """Return model parameters (PyTorch-style)."""
        return [self.weight, self.bias]
    
    def to(self, device):
        """Move model to specified device (PyTorch-style)."""
        self.device = device
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode (PyTorch-style)."""
        # For this simple model, no difference between train/eval modes
        pass
    
    def train(self):
        """Set model to training mode (PyTorch-style)."""
        # For this simple model, no difference between train/eval modes
        pass

def test_forward_pass(use_graph=False):
    """Simple forward pass test with benchmarking using do_bench."""
    
    mode = "Graph" if use_graph else "Direct"
    print(f"Single Layer Forward Pass Test ({mode} Mode) with do_bench")
    print("=" * 55)
    
    # Create model with iron.matmul compatible dimensions
    model = SingleLayerMNIST(input_size=768, output_size=32, dtype=bfloat16, device="npu")
    
    # Create random input (batch_size must be multiple of 64 for iron.matmul)
    batch_size = 128  # Must be multiple of 64
    input_data = iron.tensor(
        np.random.randn(batch_size, 768).astype(bfloat16),
        dtype=bfloat16,
        device="npu"
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if use_graph:
        # Capture graph once
        print("Capturing graph...")
        with capture_graph() as graph:
            result = model.forward(input_data)
        
        # Define forward pass function for do_bench using captured graph
        def forward_pass():
            return graph.replay()
    else:
        # Define forward pass function for do_bench using direct operations
        def forward_pass():
            return model.forward(input_data)
    
    # Run benchmark using do_bench
    results = do_bench(
        forward_pass,
        n_warmup=5,
        n_repeat=20,
        return_mode="all"
    )
    
    # Show results
    output = model.forward(input_data)  # Get output for display
    print(f"\nOutput shape: {output.shape}")
    print(f"Average time taken: {results['mean']*1000:.2f}ms")
    print(f"Sample output logits (first 5): {output.numpy()[0][:5]}")
    print(f"Predicted class for first sample: {np.argmax(output.numpy()[0])}")
    
    return results

if __name__ == "__main__":
    # Test both direct and graph modes
    #print("Testing Direct Mode:")
    #test_forward_pass(use_graph=False)
    
    #print("\n" + "="*60 + "\n")
    
    print("Testing Graph Mode:")
    test_forward_pass(use_graph=False)
