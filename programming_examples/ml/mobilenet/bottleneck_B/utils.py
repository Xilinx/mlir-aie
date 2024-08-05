from PIL import Image
import torch
import json
import numpy as np
def convert_to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        raise TypeError("Unsupported array type")
    
class ExpandChannels(object):
    def __init__(self, target_channels):
        self.target_channels = target_channels

    def __call__(self, img):
        img = torch.Tensor(img)
        channels, height, width = img.shape
        if channels == self.target_channels:
            return img
        elif channels > self.target_channels:
            return img[:self.target_channels, :, :]
        else:
            # Repeat the existing channels to match the target_channels
            repeats = self.target_channels // channels
            remainder = self.target_channels % channels
            img = img.repeat(repeats, 1, 1)
            if remainder > 0:
                img = torch.cat([img, img[:remainder, :, :]], dim=0)
            return img