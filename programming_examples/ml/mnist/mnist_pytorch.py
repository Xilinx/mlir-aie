import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print("GPU is available")
else:
    device = "cpu"
    print("GPU not available, using CPU")

class MNISTModel(nn.Module):
    """MNIST model implementation using PyTorch."""
    
    def __init__(self, input_size=768, hidden1=128, hidden2=64, output_size=32):
        """
        Initialize MNIST model.
        
        Args:
            input_size (int): Number of input features
            hidden1 (int): Number of hidden units in first layer
            hidden2 (int): Number of hidden units in second layer
            output_size (int): Number of output classes
        """
        super(MNISTModel, self).__init__()
        
        # Dense layers with ReLU activation
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"MNISTModel (PyTorch {device.upper()}) initialized:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden1: {hidden1}")
        print(f"  Hidden2: {hidden2}")
        print(f"  Output size: {output_size}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_weights(self):
        """Initialize weights with random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_size)
        """
        # First dense layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second dense layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Output layer (no softmax as requested)
        x = self.fc3(x)
        
        return x

def benchmark_model(model, input_data, device_name="cpu", n_warmup=5, n_repeat=20):
    """Benchmark the model using PyTorch timing."""
    print(f"\nBenchmarking MNIST Model PyTorch {device_name.upper()}")
    print("=" * 40)
    
    model.eval()
    model.to(device_name)
    input_data = input_data.to(device_name)
    
    # Warmup runs
    print("Warming up...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_data)
    
    # Benchmark runs
    print("Running benchmark...")
    times = []
    
    with torch.no_grad():
        for _ in range(n_repeat):
            start_time = time.time()
            output = model(input_data)
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    times_ms = [t * 1000 for t in times]
    mean_time = np.mean(times_ms)
    min_time = np.min(times_ms)
    max_time = np.max(times_ms)
    std_time = np.std(times_ms)
    
    # Show results
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average time taken: {mean_time:.2f}ms")
    print(f"Min time: {min_time:.2f}ms")
    print(f"Max time: {max_time:.2f}ms")
    print(f"Std deviation: {std_time:.2f}ms")
    
    return {
        'mean': mean_time / 1000,
        'min': min_time / 1000,
        'max': max_time / 1000,
        'std': std_time / 1000
    }

def main():
    """Main function to run PyTorch benchmarking."""
    print("MNIST Model PyTorch Benchmarking")
    print("=" * 40)
    
    print(f"Using device: {device}")
    print()
    
    # Create model
    model = MNISTModel(input_size=768, hidden1=128, hidden2=64, output_size=32)
    
    print("\nModel Architecture:")
    print(f"  Layer 1: Linear(768 -> 128) with ReLU")
    print(f"  Layer 2: Linear(128 -> 64) with ReLU")
    print(f"  Layer 3: Linear(64 -> 32)")
    print(f"  Note: Input must be (batch_size, 768)")
    
    # Create random input data
    batch_size = 256
    input_data = torch.randn(batch_size, 768, dtype=torch.float32)
    
    print(f"\nInput data shape: {input_data.shape}")
    
    # Run inference and show results
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    
    # Get predicted class for first sample
    predicted_class = torch.argmax(output[0]).item()
    confidence = torch.max(output[0]).item()
    
    print(f"\nInference Results:")
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits (first sample): {output[0].numpy()}")
    print(f"Predicted class (first sample): {predicted_class}")
    print(f"Max logit value (first sample): {confidence:.4f}")
    
    # Run benchmark
    print("\n" + "="*60)
    benchmark_model(model, input_data, device_name=device)

if __name__ == "__main__":
    main()
