#!/usr/bin/env python3
import sys
import re

# Configuration: buffer layout
# Each entry: (buffer_name, size_expr)
BUFFER_CONFIG = [
    ("input", "embedding_dim"),
    ("weights_1", "embedding_dim * hidden_dim"),
    ("weights_2", "embedding_dim * hidden_dim"),
    ("weights_3", "hidden_dim * embedding_dim"),
    ("left", "hidden_dim"),
    ("left_swished", "hidden_dim"),
    ("right", "hidden_dim"),
    ("intermediate", "hidden_dim"),
    ("output", "embedding_dim"),
]

# Configuration: operations to execute
# Each entry: (device_name, buffer_args)
# NOTE: Consecutive operations on the same device will be grouped into a single
# aiex.configure block to minimize reconfiguration overhead.
OPERATIONS_CONFIG = [
    ("gemv_1", ["weights_1", "input", "left"]),
    ("gemv_1", ["weights_2", "input", "right"]),  # Grouped with previous gemv_1
    ("silu", ["left", "left_swished"]),
    ("eltwise_mul", ["left_swished", "right", "intermediate"]),
    ("gemv_2", ["weights_3", "intermediate", "output"]),
]

# Map device names to their indices (for referencing the @device)
DEVICE_INDEX_MAP = {
    "gemv_1": 0,
    "silu": 1,
    "eltwise_mul": 2,
    "gemv_2": 3,
}

def extract_device_content(mlir_file, device_name):
    with open(mlir_file, 'r') as f:
        content = f.read()
    
    # Find the start of the aie.device
    device_start = content.find('aie.device(npu2) {')
    if device_start == -1:
        raise ValueError(f"No aie.device found in {mlir_file}")
    
    # Find matching closing brace by counting braces
    brace_count = 0
    i = device_start + len('aie.device(npu2) ')
    start_content = i + 1  # Start after the opening brace
    
    for idx in range(i, len(content)):
        if content[idx] == '{':
            brace_count += 1
        elif content[idx] == '}':
            brace_count -= 1
            if brace_count == 0:
                # Found the matching closing brace
                device_content = content[start_content:idx].strip()
                return f"    aie.device(npu2) @{device_name} {{\n{device_content}\n    }}\n"
    
    raise ValueError(f"Could not find matching closing brace in {mlir_file}")

def calculate_buffer_offsets(embedding_dim, hidden_dim):
    """Calculate buffer offsets based on configuration."""
    offsets = {}
    current_offset = 0
    
    for buffer_name, size_expr in BUFFER_CONFIG:
        size = eval(size_expr, {"embedding_dim": embedding_dim, "hidden_dim": hidden_dim})
        offsets[buffer_name] = {
            "start": current_offset,
            "size": size,
            "end": current_offset + size,
        }
        current_offset += size
    
    offsets["total"] = current_offset
    return offsets

def create_main_device(buffer_offsets):
    """Generate the main device with runtime sequence.
    Optimizes by grouping consecutive operations on the same device.
    """
    lines = []
    lines.append("    aie.device(npu2) @main {")
    lines.append("")
    lines.append(f"        aiex.runtime_sequence @sequence(%arg : memref<{buffer_offsets['total']}xbf16>) {{")
    lines.append("")
    lines.append("            %c3_i32 = arith.constant 3 : i32")
    lines.append("")
    
    # Group consecutive operations by device to minimize reconfigurations
    i = 0
    while i < len(OPERATIONS_CONFIG):
        device_name, buffer_args = OPERATIONS_CONFIG[i]
        device_ref_name = list(DEVICE_INDEX_MAP.keys())[DEVICE_INDEX_MAP[device_name]]
        
        # Collect all consecutive operations for the same device
        device_operations = [(buffer_args, i)]
        j = i + 1
        while j < len(OPERATIONS_CONFIG) and OPERATIONS_CONFIG[j][0] == device_name:
            device_operations.append((OPERATIONS_CONFIG[j][1], j))
            j += 1
        
        # Generate a single aiex.configure block for all operations on this device
        lines.append(f"            aiex.configure @{device_ref_name} {{")
        
        # Track which buffers we've already declared in this configure block
        declared_buffers = set()
        
        for op_buffer_args, op_idx in device_operations:
            # Generate arg_slice declarations for buffers not yet declared
            for buf_name in op_buffer_args:
                if buf_name not in declared_buffers:
                    buf_info = buffer_offsets[buf_name]
                    lines.append(f"                %{buf_name} = aiex.arg_slice %arg[{buf_info['start']}:{buf_info['end']}] : memref<{buffer_offsets['total']}xbf16> -> memref<{buf_info['size']}xbf16>")
                    declared_buffers.add(buf_name)
            
            # Generate aiex.run call
            buffer_args_str = ", ".join([f"%{b}" for b in op_buffer_args])
            buffer_types_str = ", ".join([f"memref<{buffer_offsets[b]['size']}xbf16>" for b in op_buffer_args])
            lines.append(f"                aiex.run @sequence ({buffer_args_str}) : ({buffer_types_str})")
        
        lines.append("            }")
        lines.append("")
        
        # Move to the next ungrouped device
        i = j
    
    lines.append("        }")
    lines.append("")
    lines.append("    }")
    
    return "\n".join(lines) + "\n"

def main():
    if len(sys.argv) != 6:
        print("Usage: combine_mlir.py <gemv1.mlir> <silu.mlir> <eltwise.mlir> <gemv2.mlir> <output.mlir>")
        sys.exit(1)
    
    embedding_dim = 2048
    hidden_dim = 8192
    
    # Calculate buffer offsets from configuration
    buffer_offsets = calculate_buffer_offsets(embedding_dim, hidden_dim)
    
    # Get unique device names from operations config
    device_names = list(DEVICE_INDEX_MAP.keys())
    
    output = "module {\n\n"
    output += create_main_device(buffer_offsets)
    output += "\n"
    
    # Extract device content for each unique device
    for i, mlir_file in enumerate(sys.argv[1:5]):
        output += extract_device_content(mlir_file, device_names[i])
        output += "\n"
    
    output += "}\n"
    
    with open(sys.argv[5], 'w') as f:
        f.write(output)

if __name__ == "__main__":
    main()
