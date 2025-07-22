#! /usr/bin/env python3

import iron
import torch
import logging

def main():
    """
    Main function with inline graph capture
    """
    # Initialize some example tensors
    batch_size, seq_len, hidden_size = 2, 10, 512
    
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    up_weight = torch.randn(hidden_size, hidden_size * 4)  # Up projection weight
    gate_input = torch.randn(batch_size, seq_len, hidden_size)
    gate_weight = torch.randn(hidden_size, hidden_size * 4)  # Gate projection weight
    down_weight = torch.randn(hidden_size * 4, hidden_size)  # Down projection weight
    
    # Set verbose level to see operation logs
    iron.set_verbose(logging.DEBUG)
    
    # Graph capture context
    with iron.capture_graph() as graph:
        up_projection = iron.matmul(input_tensor, up_weight, device="npu")
        gate_projection = iron.matmul(gate_input, gate_weight, device="npu")

        gate_activated = iron.silu(gate_projection, device="npu")
        gated_output = iron.transform(up_projection, gate_activated, lambda a, b: a * b, device="npu")

        _ = iron.matmul(gated_output, down_weight, device="npu")
    
    # Execute the captured graph
    logging.info("Executing graph")
    final_output = graph.execute()
    
    # Generate DOT file and create PNG image
    graph.visualize("llama2_graph.png")
    
    # Show final output shape
    if final_output is not None:
        print(f"Final output shape: {final_output.shape}")
    else:
        print("Final output shape: None")
    return final_output

if __name__ == "__main__":
    result = main()
