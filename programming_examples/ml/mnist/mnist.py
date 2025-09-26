import numpy as np
import aie.iron as iron
from ml_dtypes import bfloat16
from aie.iron.functional import relu
from aie.iron.algorithms import for_each
from aie.iron.graph import capture_graph
from utilities import do_bench
import argparse
import os



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
        
        # Initialize weights with random values
        self.weight = iron.tensor(
            np.random.randn(in_features, out_features).astype(dtype) * 0.1,
            dtype=dtype,
            device=device
        )
        
        # No bias (as per your requirement)
        self.bias = None
    
    def load_state_dict(self, state_dict):
        """Load weights from state dictionary (PyTorch-style)."""
        if 'weight' in state_dict:
            weight_data = state_dict['weight']
            
            # PyTorch stores weights as (out_features, in_features)
            # Iron expects (in_features, out_features)
            if weight_data.shape == (self.out_features, self.in_features):
                # Transpose to match Iron format
                weight_data = weight_data.T
            elif weight_data.shape != (self.in_features, self.out_features):
                raise ValueError(f"Weight shape {weight_data.shape} doesn't match expected ({self.in_features}, {self.out_features}) or ({self.out_features}, {self.in_features})")
            
            self.weight = iron.tensor(
                weight_data.astype(self.dtype),
                dtype=self.dtype,
                device=self.device
            )
    
    def state_dict(self):
        """Return state dictionary (PyTorch-style)."""
        return {'weight': self.weight.numpy()}
    
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
        
        # No bias addition (as per your requirement)
        
        return output
    
    def parameters(self):
        """Return layer parameters."""
        return [self.weight] if self.bias is None else [self.weight, self.bias]

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
        
        print("MNISTModel initialized:")
        print(f"  Data type: {self.dtype}")
        print(f"  Device: {self.device}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def load_state_dict(self, state_dict):
        """Load weights from state dictionary (PyTorch-style)."""
        self.fc1.load_state_dict(state_dict['fc1'])
        self.fc2.load_state_dict(state_dict['fc2'])
        self.fc3.load_state_dict(state_dict['fc3'])
        print("✅ Model weights loaded successfully!")
    
    def state_dict(self):
        """Return state dictionary (PyTorch-style)."""
        return {
            'fc1': self.fc1.state_dict(),
            'fc2': self.fc2.state_dict(),
            'fc3': self.fc3.state_dict()
        }
    
    def load_weights_from_dir(self, weights_dir):
        """Load weights from directory (convenience method)."""
        import os
        import pickle
        
        # Load weight files
        fc1_path = os.path.join(weights_dir, "fc1_weight.npy")
        fc2_path = os.path.join(weights_dir, "fc2_weight.npy")
        fc3_path = os.path.join(weights_dir, "fc3_weight.npy")
        
        if not all(os.path.exists(p) for p in [fc1_path, fc2_path, fc3_path]):
            raise FileNotFoundError(f"Weight files not found in {weights_dir}")
        
        # Load weights
        fc1_weight = np.load(fc1_path)
        fc2_weight = np.load(fc2_path)
        fc3_weight = np.load(fc3_path)
        
        # Create state dict and load
        state_dict = {
            'fc1': {'weight': fc1_weight},
            'fc2': {'weight': fc2_weight},
            'fc3': {'weight': fc3_weight}
        }
        
        self.load_state_dict(state_dict)
        print(f"✅ Loaded weights from: {weights_dir}")
        print(f"  fc1: {fc1_weight.shape}")
        print(f"  fc2: {fc2_weight.shape}")
        print(f"  fc3: {fc3_weight.shape}")
    
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

def inference_example(weights_dir=None, test_image_file=None):
    """Example inference with test image using Iron."""
    # Create model
    model = MNISTModel(dtype=bfloat16, device="npu")
    
    # Load weights if provided (PyTorch-style)
    if weights_dir:
        model.load_weights_from_dir(weights_dir)
    
    model.eval()  # Set to evaluation mode
    
    print("\nModel Architecture:")
    print("  Layer 1: Linear(768 -> 128) with ReLU")
    print("  Layer 2: Linear(128 -> 64) with ReLU")
    print("  Layer 3: Linear(64 -> 32)")
    print("  Note: Input must be (batch_size, 768) where batch_size % 128 = 0")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Load test image if provided
    if test_image_file and os.path.exists(test_image_file):
        print(f"\nLoading test image from: {test_image_file}")
        
        # Load the test image (should be 768 dimensions already)
        test_image = np.load(test_image_file)
        print(f"Loaded image shape: {test_image.shape}")
        
        # Ensure it's the right shape for batch processing
        if test_image.ndim == 1:
            # Single image, pad batch dimension to 128 (Iron matmul requirement)
            test_input = test_image.reshape(1, 768)
            # Pad batch dimension to 128
            batch_padded = np.zeros((128, 768), dtype=bfloat16)
            batch_padded[0] = test_input[0]  # Put the actual image in first position
            test_input = batch_padded
        else:
            test_input = test_image
        
        # Convert to Iron tensor
        input_tensor = iron.tensor(
            test_input.astype(bfloat16),
            dtype=bfloat16,
            device="npu"
        )
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Run inference
        print("\nRunning inference...")
        total_time = 0
        num_runs = 100
        import time
        with capture_graph() as graph:
            _ = model.forward(input_tensor)
            graph.compile()

            for i in range(num_runs):
                start_time = time.time()
                output = graph.replay()
                end_time = time.time()
                total_time += end_time - start_time
        total_time /= num_runs
        print(f"Average time taken: {total_time*1000:.2f} ms")
        # Get prediction (no softmax - just raw logits)
        output_np = output.numpy()
        # Only use the first result since we padded the batch
        predicted_class = np.argmax(output_np[0])
        max_logit = float(np.max(output_np[0]))
        
        print("\n" + "="*50)
        print("INFERENCE RESULTS:")
        print("="*50)
        print(f"Input shape: {input_tensor.shape} (padded to batch size 128)")
        print(f"Output shape: {output.shape}")
        print(f"Raw logits (first sample): {output_np[0]}")
        print(f"Predicted class: {predicted_class}")
        print(f"Max logit value: {max_logit:.4f}")
        print("="*50)
        
    else:
        print(f"\nNo test image file provided or file not found: {test_image_file}")
        print("Usage: python mnist.py [weights_dir] [test_image_file]")
        print("Example: python mnist.py mnist_weights test_image.npy")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    weights_dir = None
    test_image_file = None
    
    args = argparse.ArgumentParser()
    args.add_argument("-w", "--weights_dir", type=str, default="mnist_weights")
    args.add_argument("-t", "--test_image_file", type=str, default="data/test_image_5.npy")
    args = args.parse_args()
    
    weights_dir = args.weights_dir
    test_image_file = args.test_image_file
    
    print(f"Loading pre-trained weights from: {weights_dir}")
    print(f"Test image file: {test_image_file}")
    
    inference_example(weights_dir=weights_dir, test_image_file=test_image_file)
