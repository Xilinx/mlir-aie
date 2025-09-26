import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
import glob
import random

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print("GPU is available")
else:
    device = "cpu"
    print("GPU not available, using CPU")

class CustomDataset(Dataset):
    """Custom dataset with augmentation to expand from 50 to 5000 samples."""
    
    def __init__(self, data_dir, target_size=5000):
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "digit_*.npy")))
        self.target_size = target_size
        print(f"Loaded {len(self.data_files)} original custom samples")
        print(f"Will generate {target_size} augmented samples")
    
    def __len__(self):
        return self.target_size
    
    def augment_data(self, data):
        """Apply augmentation to 768-feature data."""
        # Your data is 768 features, so we need to pad it to 784 (28x28) first
        # Pad the 768 features to 784 (28x28)
        data_padded = np.zeros(784)
        data_padded[:768] = data
        
        # Reshape to 28x28 for augmentation
        img_28x28 = data_padded.reshape(28, 28)
        
        # Convert to PIL Image for augmentation
        from PIL import Image
        img_pil = Image.fromarray((img_28x28 * 255).astype(np.uint8))
        
        # Apply random augmentations
        if random.random() < 0.7:  # 70% chance of rotation
            angle = random.uniform(-15, 15)
            img_pil = img_pil.rotate(angle, fillcolor=0)
        
        if random.random() < 0.5:  # 50% chance of translation
            dx = random.uniform(-2, 2)
            dy = random.uniform(-2, 2)
            img_pil = img_pil.transform(img_pil.size, Image.AFFINE, 
                                      (1, 0, dx, 0, 1, dy), fillcolor=0)
        
        if random.random() < 0.3:  # 30% chance of scaling
            scale = random.uniform(0.9, 1.1)
            new_size = (int(28 * scale), int(28 * scale))
            img_pil = img_pil.resize(new_size, Image.LANCZOS)
            # Crop back to 28x28
            left = (new_size[0] - 28) // 2
            top = (new_size[1] - 28) // 2
            img_pil = img_pil.crop((left, top, left + 28, top + 28))
        
        # Convert back to numpy and normalize
        img_array = np.array(img_pil) / 255.0
        
        # Apply MNIST normalization
        mean = 0.1307
        std = 0.3081
        img_normalized = (img_array - mean) / std
        
        # Flatten to 784 features (28x28) to match MNIST
        img_flat = img_normalized.flatten()
        
        return img_flat
    
    def __getitem__(self, idx):
        # Select a random original sample
        original_idx = idx % len(self.data_files)
        data_path = self.data_files[original_idx]
        data = np.load(data_path)
        
        # Apply augmentation
        augmented_data = self.augment_data(data)
        
        # Convert to tensor and reshape to match MNIST format (1, 28, 28)
        data_tensor = torch.from_numpy(augmented_data).float()
        data_tensor = data_tensor.view(1, 28, 28)  # Reshape to (1, 28, 28) to match MNIST
        
        # Extract label from filename
        filename = os.path.basename(data_path)
        label = int(filename.split('_')[1])
        
        # Return label as int to match MNIST format
        return data_tensor, label

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
        
        # Dense layers with ReLU activation (no bias)
        self.fc1 = nn.Linear(input_size, hidden1, bias=False)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.fc3 = nn.Linear(hidden2, output_size, bias=False)
        
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

def get_augmented_transforms():
    """Get transforms with data augmentation for training."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=15),  # Rotate ±15 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Shift slightly
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Random erasing
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_test_transforms():
    """Get transforms for testing (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def load_mnist_data(batch_size=256, num_workers=2):
    """Load MNIST dataset with custom data combined."""
    # Load MNIST datasets with different transforms
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=get_augmented_transforms()
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=get_test_transforms()
    )
    
    # Load custom dataset
    custom_dataset = CustomDataset('custom_data')
    
    # Combine MNIST and custom data
    train_dataset = ConcatDataset([mnist_train, custom_dataset])
    test_dataset = mnist_test  # Keep original MNIST test set
    
    # Create data loaders with num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, test_loader

def train_model(model, train_loader, optimizer, criterion, device_name="cpu"):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device_name), target.to(device_name)
        
        # Reshape data to match model input size (28*28 = 784, but model expects 768)
        # We'll crop the data to 768 dimensions (not pad!)
        batch_size = data.size(0)
        data = data.view(batch_size, -1)  # Flatten to (batch_size, 784)
        data = data[:, :768]  # Crop to 768 dimensions (not pad!)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, criterion, device_name="cpu"):
    """Evaluate the model on test data."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device_name), target.to(device_name)
            
            # Reshape data to match model input size
            batch_size = data.size(0)
            data = data.view(batch_size, -1)  # Flatten to (batch_size, 784)
            data = data[:, :768]  # Crop to 768 dimensions (not pad!)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

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
    """Main function to run PyTorch training and benchmarking."""
    print("MNIST Model PyTorch Training & Benchmarking")
    print("=" * 50)
    
    print(f"Using device: {device}")
    print()
    
    # Create model
    model = MNISTModel(input_size=768, hidden1=128, hidden2=64, output_size=32)  # Changed to 32 classes for Iron compatibility
    
    print("\nModel Architecture:")
    print(f"  Layer 1: Linear(768 -> 128) with ReLU (no bias)")
    print(f"  Layer 2: Linear(128 -> 64) with ReLU (no bias)")
    print(f"  Layer 3: Linear(64 -> 32) (no bias)")
    print(f"  Note: Input will be cropped from 784 to 768 dimensions")
    print(f"  Data Augmentation: Rotation ±15°, Translation ±10%, Random Erasing 20%")
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=256)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Slightly higher LR for augmented data
    
    # Move model to device
    model.to(device)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50)
    
    num_epochs = 25  # More epochs for augmented data
    best_test_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Train
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # Learning rate scheduling
        if epoch > 15 and epoch % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
            print(f"Learning rate reduced to: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    
    # Save model weights
    print("\n" + "="*50)
    print("Saving Model Weights:")
    print("="*50)
    
    # Create weights directory
    import os
    weights_dir = "mnist_weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Save weights for each layer
    np.save(f"{weights_dir}/fc1_weight.npy", model.fc1.weight.detach().cpu().numpy())
    np.save(f"{weights_dir}/fc2_weight.npy", model.fc2.weight.detach().cpu().numpy())
    np.save(f"{weights_dir}/fc3_weight.npy", model.fc3.weight.detach().cpu().numpy())
    
    # Save model metadata
    model_info = {
        'input_size': 768,
        'hidden1': 128,
        'hidden2': 64,
        'output_size': 32,
        'best_test_acc': best_test_acc,
        'architecture': '3-layer MLP with ReLU, no bias',
        'augmentation': 'Rotation ±15°, Translation ±10%, Random Erasing 20%'
    }
    torch.save(model_info, f"{weights_dir}/model_info.pkl")
    
    print(f"Weights saved to: {weights_dir}/")
    print(f"  - fc1_weight.npy: {model.fc1.weight.shape}")
    print(f"  - fc2_weight.npy: {model.fc2.weight.shape}")
    print(f"  - fc3_weight.npy: {model.fc3.weight.shape}")
    print(f"  - model_info.pkl: Model metadata")
    
    # Run inference on a sample
    print("\n" + "="*50)
    print("Sample Inference:")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        # Get a sample from test set
        sample_data, sample_target = next(iter(test_loader))
        sample_data = sample_data[0:1].to(device)  # Take first sample
        sample_target = sample_target[0:1].to(device)
        
        # Reshape and crop data (match Iron model)
        batch_size = sample_data.size(0)
        sample_data = sample_data.view(batch_size, -1)  # Flatten to (batch_size, 784)
        sample_data = sample_data[:, :768]  # Crop to 768 dimensions (not pad!)
        
        output = model(sample_data)
        predicted_class = torch.argmax(output[0]).item()
        confidence = torch.max(output[0]).item()
        true_class = sample_target[0].item()
        
        print(f"True class: {true_class}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Correct: {'Yes' if predicted_class == true_class else 'No'}")
    
    # Run benchmark
    print("\n" + "="*50)
    print("Performance Benchmark:")
    print("="*50)
    
    # Create random input data for benchmarking
    batch_size = 256
    input_data = torch.randn(batch_size, 768, dtype=torch.float32)
    benchmark_model(model, input_data, device_name=device)

if __name__ == "__main__":
    main()
