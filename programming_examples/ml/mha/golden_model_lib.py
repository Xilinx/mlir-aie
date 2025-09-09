### A library to generated C headers with golden values from PyTorch tensors
### Author: Victor Jung
###

import os 
from typing import Dict

import torch
import numpy as np

# Map data types to C++ types
CPP_DTYPE_MAP = {
    "bf16": "std::bfloat16_t",
    "f32": "float",
    "i8": "int8_t",
    "i16": "int16_t", 
    "i32": "int32_t",
}

HEADER_STR = """// Generated golden reference values for {header_name}

#ifndef GOLDEN_REFERENCE_H
#define GOLDEN_REFERENCE_H

#include <array>
#include <cstdint>
#include <stdfloat>

namespace golden_reference {{

"""

CLOSING_STR = """} // namespace golden_reference\n#endif // GOLDEN_REFERENCE_H\n"""

def tensor_to_header(array: np.ndarray, cpp_dtype: str, name: str) -> str:
    
    ret = "\n"
    ret += f"// Array {name} {array.shape} of type {cpp_dtype}\n"
    ret += f"constexpr std::array<{cpp_dtype}, {np.prod(array.shape)}> {name} = {{\n"
        
    array_flat = array.flatten().astype(np.float32) if cpp_dtype == "std::bfloat16_t" else array.flatten()
    
    for i, val in enumerate(array_flat):
        if cpp_dtype == "std::bfloat16_t":
            ret += f"    {cpp_dtype}({float(val):.6f}f)"
        elif cpp_dtype == "float":
            ret += f"    {float(val):.6f}f"
        else:
            ret += f"    {int(val)}"
        
        if i < len(array_flat) - 1:
            ret += ","
        if (i + 1) % 8 == 0:
            ret += "\n"
        
    ret += "\n};"
    return ret

def export_to_header(tensor_dict: Dict[str, np.ndarray], dtype: str, header_path: str, name: str = "golden_reference"):
    """Export matrices to C++ header file."""
    
    cpp_dtype = CPP_DTYPE_MAP[dtype]
    header_dir = os.path.dirname(header_path)
    
    if header_dir and not os.path.exists(header_dir):
        os.makedirs(header_dir, exist_ok=True)

    with open(header_path, 'w') as f:
        f.write(HEADER_STR.format(header_name=name))

        for name, tensor in tensor_dict.items():
            f.write(tensor_to_header(tensor, cpp_dtype, name))
            f.write("\n\n")
        
        f.write(CLOSING_STR)
        
def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()