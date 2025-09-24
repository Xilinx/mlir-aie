import numpy as np
import aie.iron as iron
from ml_dtypes import bfloat16
from aie.iron.functional import relu
from aie.iron.algorithms import for_each
from aie.iron.graph import capture_graph
from utilities import do_bench

class Linear:
    """Linear layer implementation using Iron operations."""
    
    def __init__(self, in_features, out_features, dtype=bfloat16, device="npu"):
        """
        Initialize linear layer.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            dtype: Data type for weights and computations
            device (str): Device to run on
        """
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device
        
        # Initialize weights and bias
        self.weight = iron.tensor(
            np.random.randn(in_features, out_features).astype(dtype) * 0.1,
            dtype=dtype,
            device=device
        )
        
        self.bias = iron.tensor(
            np.zeros(out_features, dtype=dtype),
            dtype=dtype,
            device=device
        )
    
    def forward(self, x):
        """
        Forward pass through linear layer.
        
        Args:
            x (iron.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            iron.Tensor: Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # Create output tensor
        output = iron.tensor(
            np.zeros((batch_size, self.out_features), dtype=self.dtype),
            dtype=self.dtype,
            device=self.device
        )
        
        # Matrix multiplication: x @ weight
        iron.matmul(x, self.weight, out=output)
        
        # Add bias to each row
        # TODO: Use iron.add with broadcasting
        #for i in range(batch_size):
        #    output[i] = output[i] + self.bias
        
        return output
    
    def parameters(self):
        """Return layer parameters."""
        return [self.weight, self.bias]

class ReLU:
    """ReLU activation layer implementation using Iron operations."""
    
    def __init__(self, dtype=bfloat16, device="npu"):
        """
        Initialize ReLU layer.
        
        Args:
            dtype: Data type for computations
            device (str): Device to run on
        """
        self.dtype = dtype
        self.device = device
        self.relu_func = relu(dtype)
    
    def forward(self, x):
        """
        Forward pass through ReLU layer.
        
        Args:
            x (iron.Tensor): Input tensor
            
        Returns:
            iron.Tensor: Output tensor with ReLU applied
        """
        # Apply ReLU activation to each element
        for_each(x.view(-1), self.relu_func)
        return x
    
    def parameters(self):
        """Return layer parameters (ReLU has no parameters)."""
        return []

class MNISTModel:
    """MNIST model implementation using Iron operations."""
    
    def __init__(self, dtype=bfloat16, device="npu"):
        """
        Initialize MNIST model.
        
        Args:
            dtype: Data type for weights and computations
            device (str): Device to run on
        """
        self.dtype = dtype
        self.device = device
        
        # Dense layers with ReLU activation - shapes adjusted for Iron matmul constraints
        # Iron matmul requires: M%64=0, K%64=0, N%32=0
        self.fc1 = Linear(768, 128, dtype=dtype, device=device)        # First dense layer: 768 -> 128 (768%64=0, 128%32=0)
        self.relu1 = ReLU(dtype=dtype, device=device)                 # ReLU activation after fc1
        self.fc2 = Linear(128, 64, dtype=dtype, device=device)        # Second dense layer: 128 -> 64 (128%64=0, 64%32=0)
        self.relu2 = ReLU(dtype=dtype, device=device)                 # ReLU activation after fc2
        self.fc3 = Linear(64, 32, dtype=dtype, device=device)        # Output layer: 64 -> 32 (64%64=0, 32%32=0)
        
        print(f"MNISTModel initialized:")
        print(f"  Data type: {self.dtype}")
        print(f"  Device: {self.device}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def parameters(self):
        """Return all model parameters."""
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.relu1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.relu2.parameters())
        params.extend(self.fc3.parameters())
        return params
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        self.fc1.device = device
        self.relu1.device = device
        self.fc2.device = device
        self.relu2.device = device
        self.fc3.device = device
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        pass
    
    def train(self):
        """Set model to training mode."""
        pass
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (iron.Tensor): Input tensor of shape (batch_size, 768) - must be Iron compatible
            
        Returns:
            iron.Tensor: Output logits of shape (batch_size, 32)
        """
        # First dense layer with ReLU activation
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        
        # Second dense layer with ReLU activation
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        
        # Output layer (no softmax)
        x = self.fc3.forward(x)
        
        return x

def benchmark_model(model, input_data, use_graph=False):
    """Benchmark the model using do_bench."""
    mode = "Graph" if use_graph else "Direct"
    print(f"\nBenchmarking MNIST Model ({mode} Mode)")
    print("=" * 40)
    
    if use_graph:
        # Capture graph once
        print("Capturing graph...")
        with capture_graph() as graph:
            result = model.forward(input_data)
        
        # Define forward pass function for do_bench using captured graph
        def forward_pass():
            return graph.replay(input_data)
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
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average time taken: {results['mean']*1000:.2f}ms")
    print(f"Min time: {results['min']*1000:.2f}ms")
    print(f"Max time: {results['max']*1000:.2f}ms")
    print(f"Std deviation: {results['std']*1000:.2f}ms")
    
    return results

def inference_example():
    """Example inference with random input using Iron."""
    # Create model
    model = MNISTModel(dtype=bfloat16, device="npu")
    model.eval()  # Set to evaluation mode
    
    print("\nModel Architecture:")
    print(f"  Layer 1: Linear(768 -> 128) with ReLU")
    print(f"  Layer 2: Linear(128 -> 64) with ReLU")
    print(f"  Layer 3: Linear(64 -> 32)")
    print(f"  Note: Input must be (batch_size, 768) where batch_size % 128 = 0")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Example inference with random input
    # Create random input (batch_size=256, features=768) - batch must be divisible by 128
    random_input = iron.tensor(
        np.random.randn(256, 768).astype(bfloat16),
        dtype=bfloat16,
        device="npu"
    )
    
    # Run inference and show results
    output = model.forward(random_input)
    
    # TBD:
    # iron.export(output, "mnist_model")
    # iron.import("mnist_model")
    
    # Get predicted class for first sample
    output_np = output.numpy()
    predicted_class = np.argmax(output_np[0])
    confidence = np.max(output_np[0])
    
    print(f"\nInference Results:")
    print(f"Random input shape: {random_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits (first sample): {output_np[0]}")
    print(f"Predicted class (first sample): {predicted_class}")
    print(f"Max logit value (first sample): {float(confidence):.4f}")
    
    # Run benchmarks
    print("\n" + "="*60)
    benchmark_model(model, random_input, use_graph=False)
    
    #print("\n" + "="*60)
    #benchmark_model(model, random_input, use_graph=True)

if __name__ == "__main__":
    inference_example()
