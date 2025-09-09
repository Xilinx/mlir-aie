import os
import sys
import argparse

import torch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from golden_model_lib import export_to_header, torch_to_numpy


# Map data types to PyTorch types
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
}


def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch golden reference for matrix multiplication")
    parser.add_argument("--dtype", type=str, choices=["bf16", "f32"], default="bf16", help="Input data type")
    parser.add_argument("--output", type=str, default="golden_reference.h", help="Output header file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    parser.add_argument("--heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--S_q", type=int, default=256, help="Sequence length for query (Q)")
    parser.add_argument("--S_kv", type=int, default=256, help="Sequence length for key/value (KV)")
    parser.add_argument("-d", type=int, default=256, help="Embedding dimension (d)")
    parser.add_argument("--num_KV_heads", type=int, default=2, help="Number of heads for Key-Value pairs")
    
    args = parser.parse_args()
    
    num_kv_heads = args.num_KV_heads
    if args.num_KV_heads == 0:
        num_kv_heads = args.heads
    number_of_groups = args.heads // num_kv_heads

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    val_range = 4
    
    Q = torch.rand(args.heads, args.S_q, args.d, dtype=torch.float32) * val_range
    K = torch.rand(num_kv_heads, args.S_kv, args.d, dtype=torch.float32) * val_range
    V = torch.rand(num_kv_heads, args.S_kv, args.d, dtype=torch.float32) * val_range
    
    K = K.repeat_interleave(number_of_groups, dim=0)
    V = V.repeat_interleave(number_of_groups, dim=0)
    
    # MHA from PyTorch
    inv_scale = 1 / np.sqrt(K.shape[-1])
    O = torch.nn.functional.scaled_dot_product_attention(
        Q.to(torch.bfloat16),
        K.to(torch.bfloat16),
        V.to(torch.bfloat16),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=inv_scale
    ).to(torch.float32)
    
    tensor_dict = {
        "Q": torch_to_numpy(Q),
        "K": torch_to_numpy(K),
        "V": torch_to_numpy(V),
        "O": torch_to_numpy(O)
    }
    
    export_to_header(tensor_dict, args.dtype, args.output, name="Multi-Head Attention")

if __name__ == "__main__":
    main()
