#!/usr/bin/env python3
"""
MNIST Test Image Downloader & Processor

Downloads MNIST data, extracts it, and creates test images for the MNIST model.

Usage:
    python download_test_images.py
"""

import numpy as np
import struct
import os
import gzip
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch

def download_and_extract_mnist():
    """Download and extract MNIST data if not already present."""
    print("Checking MNIST data...")
    
    # Check if raw files exist
    raw_dir = "data/MNIST/raw"
    images_file = os.path.join(raw_dir, "t10k-images-idx3-ubyte")
    labels_file = os.path.join(raw_dir, "t10k-labels-idx1-ubyte")
    
    if os.path.exists(images_file) and os.path.exists(labels_file):
        print("✅ Raw MNIST files already exist")
        return images_file, labels_file
    
    print("📥 Downloading MNIST dataset...")
    
    # Download using torchvision (this will extract automatically)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    print("✅ MNIST data downloaded and extracted")
    return images_file, labels_file

def read_mnist_images(filename):
    """Read MNIST images from raw data file."""
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        print(f"📊 MNIST Images:")
        print(f"   Count: {num_images}")
        print(f"   Size: {rows}x{cols}")
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        
        return images

def read_mnist_labels(filename):
    """Read MNIST labels from raw data file."""
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        print(f"📊 MNIST Labels:")
        print(f"   Count: {num_labels}")
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return labels

def process_and_save_test_images():
    """Process MNIST data and create test images."""
    print("MNIST Test Image Downloader & Processor")
    print("=" * 50)
    
    # Download/extract MNIST data
    images_file, labels_file = download_and_extract_mnist()
    
    # Read MNIST data
    images = read_mnist_images(images_file)
    labels = read_mnist_labels(labels_file)
    
    print(f"\n✅ Loaded {len(images)} test images")
    
    # Process first 5 images (one from each digit 0-4)
    test_images = []
    test_labels = []
    
    for digit in range(5):
        # Find first image of this digit
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) > 0:
            idx = digit_indices[0]
            test_images.append(images[idx])
            test_labels.append(labels[idx])
            print(f"   Digit {digit}: Using image {idx}")
    
    print(f"\n🔄 Processing {len(test_images)} test images...")
    
    # Process each test image
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        print(f"\nProcessing image {i+1} (digit {label}):")
        
        # Normalize image (convert to float and normalize like PyTorch)
        img_normalized = image.astype(np.float32) / 255.0
        img_normalized = (img_normalized - 0.1307) / 0.3081
        
        print(f"   Original shape: {image.shape}")
        print(f"   Pixel range: [{image.min()}, {image.max()}]")
        print(f"   Normalized range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
        
        # Flatten and crop to 768 dimensions
        img_flat = img_normalized.flatten()  # (784,)
        img_cropped = img_flat[:768]  # (768,)
        
        print(f"   Flattened: {img_flat.shape}")
        print(f"   Cropped: {img_cropped.shape}")
        
        # Save the test image in the data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.join(data_dir, f"test_image_{i+1}.npy")
        np.save(filename, img_cropped)
        print(f"   ✅ Saved as: {filename}")
        
        # Create visual version
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap='gray')
        plt.title(f'Test Image {i+1} (Digit: {label})')
        plt.axis('off')
        visual_filename = os.path.join(data_dir, f"test_image_{i+1}_visual.png")
        plt.savefig(visual_filename, bbox_inches='tight')
        plt.close()
        print(f"   📸 Visual saved as: {visual_filename}")
    
    print(f"\n🎯 All test images ready!")
    print(f"   Use with: python mnist.py mnist_weights test_image_1.npy")
    print(f"   Or test all: python mnist.py mnist_weights test_image_2.npy")
    print(f"   etc...")

if __name__ == "__main__":
    process_and_save_test_images()