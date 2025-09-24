import numpy as np
import aie.iron as iron

class IronDenseLayer:
    """
    A dense (fully connected) layer implemented using iron.matmul.
    
    This layer performs: output = activation(input @ weight + bias)
    where @ is matrix multiplication using iron.matmul.
    """
    
    def __init__(self, input_size, output_size, activation=None, dtype=np.int16, device="npu"):
        """
        Initialize the dense layer.
        
        Args:
            input_size (int): Number of input features
            output_size (int): Number of output features  
            activation (str, optional): Activation function ('relu', 'softmax', None)
            dtype (np.dtype): Data type for weights and computations
            device (str): Device to run on ('npu' or 'cpu')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.dtype = dtype
        self.device = device
        
        # Initialize weights and bias with random values
        self.weight = iron.tensor(
            np.random.randn(input_size, output_size).astype(dtype) * 0.1,
            dtype=dtype,
            device=device
        )
        
        self.bias = iron.tensor(
            np.zeros(output_size, dtype=dtype),
            dtype=dtype, 
            device=device
        )
    
    def forward(self, x):
        """
        Forward pass through the dense layer.
        
        Args:
            x (iron.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            iron.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Ensure input is on the same device as weights
        if x.device != self.device:
            x = x.to(self.device)
        
        # Matrix multiplication: x @ weight
        # x: (batch_size, input_size), weight: (input_size, output_size)
        # result: (batch_size, output_size)
        output = iron.matmul(x, self.weight)
        
        # Add bias (broadcasting)
        # For now, we'll do this element-wise since iron doesn't have broadcast add
        # In a real implementation, you might want to tile the bias appropriately
        batch_size = x.shape[0]
        bias_expanded = iron.tensor(
            np.tile(self.bias.numpy(), (batch_size, 1)),
            dtype=self.dtype,
            device=self.device
        )
        
        # Apply activation function
        if self.activation == 'relu':
            output = self._relu(output)
        elif self.activation == 'softmax':
            output = self._softmax(output)
        
        return output
    
    def _relu(self, x):
        """ReLU activation function."""
        # For now, we'll use numpy operations since iron doesn't have element-wise ops yet
        # In a real implementation, you'd use iron's element-wise operations
        x_np = x.numpy()
        x_np = np.maximum(0, x_np)
        return iron.tensor(x_np, dtype=self.dtype, device=self.device)
    
    def _softmax(self, x):
        """Softmax activation function."""
        # For now, we'll use numpy operations since iron doesn't have element-wise ops yet
        x_np = x.numpy()
        # Subtract max for numerical stability
        x_np = x_np - np.max(x_np, axis=1, keepdims=True)
        exp_x = np.exp(x_np)
        x_np = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return iron.tensor(x_np, dtype=self.dtype, device=self.device)

class IronNeuralNetwork:
    """
    A simple neural network using iron.matmul for matrix operations.
    
    Architecture: Flatten -> Dense(128, ReLU) -> Dense(32, ReLU) -> Dense(10, Softmax)
    """
    
    def __init__(self, input_shape=(28, 28), dtype=np.int16, device="npu"):
        """
        Initialize the neural network.
        
        Args:
            input_shape (tuple): Input image shape (height, width)
            dtype (np.dtype): Data type for computations
            device (str): Device to run on ('npu' or 'cpu')
        """
        self.input_shape = input_shape
        self.input_size = input_shape[0] * input_shape[1]  # 28 * 28 = 784
        self.dtype = dtype
        self.device = device
        
        # Create layers
        self.flatten_size = self.input_size
        self.layer1 = IronDenseLayer(self.input_size, 128, activation='relu', dtype=dtype, device=device)
        self.layer2 = IronDenseLayer(128, 32, activation='relu', dtype=dtype, device=device)
        self.layer3 = IronDenseLayer(32, 10, activation='softmax', dtype=dtype, device=device)
    
    def forward(self, x):
        """
        Forward pass through the entire network.
        
        Args:
            x (iron.Tensor): Input tensor of shape (batch_size, height, width)
            
        Returns:
            iron.Tensor: Output tensor of shape (batch_size, 10)
        """
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        # Flatten input: (batch_size, 28, 28) -> (batch_size, 784)
        batch_size = x.shape[0]
        x_flat = iron.tensor(
            x.numpy().reshape(batch_size, -1),
            dtype=self.dtype,
            device=self.device
        )
        
        # Forward pass through layers
        x = self.layer1.forward(x_flat)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        
        return x

def test_iron_neural_network():
    """Test the iron neural network implementation."""
    print("Testing Iron Neural Network Implementation")
    print("=" * 50)
    
    # Create network
    model = IronNeuralNetwork(input_shape=(28, 28), dtype=np.int16, device="npu")
    
    print(f"Model created with device: {model.device}")
    print(f"Input size: {model.input_size}")
    print(f"Layer 1: {model.input_size} -> 128 (ReLU)")
    print(f"Layer 2: 128 -> 32 (ReLU)")
    print(f"Layer 3: 32 -> 10 (Softmax)")
    
    # Create random input
    batch_size = 2
    random_input = iron.tensor(
        np.random.randn(batch_size, 28, 28).astype(np.int16),
        dtype=np.int16,
        device="npu"
    )
    
    print(f"\nInput shape: {random_input.shape}")
    print(f"Input device: {random_input.device}")
    
    # Run forward pass
    with iron.no_grad():  # Disable gradient computation for inference
        output = model.forward(random_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    print(f"Output values: {output.numpy()}")
    
    # Check that output probabilities sum to 1 (for softmax)
    output_np = output.numpy()
    prob_sums = np.sum(output_np, axis=1)
    print(f"Probability sums: {prob_sums}")
    print(f"All probabilities sum to 1: {np.allclose(prob_sums, 1.0)}")
    
    # Get predicted classes
    predicted_classes = np.argmax(output_np, axis=1)
    print(f"Predicted classes: {predicted_classes}")

def test_individual_layer():
    """Test individual dense layer."""
    print("\nTesting Individual Dense Layer")
    print("=" * 30)
    
    # Create a dense layer
    layer = IronDenseLayer(input_size=784, output_size=128, activation='relu', dtype=np.int16, device="npu")
    
    # Create random input
    batch_size = 4
    input_data = iron.tensor(
        np.random.randn(batch_size, 784).astype(np.int16),
        dtype=np.int16,
        device="npu"
    )
    
    print(f"Input shape: {input_data.shape}")
    
    # Forward pass
    output = layer.forward(input_data)
    
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    print(f"Output sample: {output.numpy()[0][:5]}")  # First 5 values of first sample

if __name__ == "__main__":
    # Run tests
    test_individual_layer()
    test_iron_neural_network()
