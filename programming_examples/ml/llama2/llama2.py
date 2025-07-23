#! /usr/bin/env python3

import iron2
import torch
import logging
import aie.iron as iron


def main():
    """
    Main function with inline graph capture
    """
    # Initialize some example tensors
    batch_size, seq_len, hidden_size = 2, 10, 512

    # Alternative size configurations (uncomment to use):
    # batch_size, seq_len, hidden_size = 4, 20, 768  # Moderately larger
    batch_size, seq_len, hidden_size = 8, 50, 1024  # Significantly larger
    # batch_size, seq_len, hidden_size = 16, 128, 2048   # Much larger
    # batch_size, seq_len, hidden_size = 32, 256, 4096   # Large (smaller Llama2)
    # batch_size, seq_len, hidden_size = 64, 512, 8192  # Very large (full Llama2)

    input_tensor = iron.rand(batch_size, seq_len, hidden_size, device="npu")

    # Up projection weight
    up_weight = iron.rand(hidden_size, hidden_size * 4, device="npu")
    gate_input = iron.rand(batch_size, seq_len, hidden_size, device="npu")
    # Gate projection weight
    gate_weight = iron.rand(hidden_size, hidden_size * 4, device="npu")
    # Down projection weight
    down_weight = iron.rand(hidden_size * 4, hidden_size, device="npu")

    # Set verbose level to see operation logs
    iron2.set_verbose(logging.DEBUG)

    # Graph capture context
    with iron2.capture_graph() as graph:
        up_projection = iron2.matmul(input_tensor, up_weight, device="cpu")
        gate_projection = iron2.matmul(gate_input, gate_weight, device="cpu")

        gate_activated = iron2.silu(gate_projection, device="cpu")
        gated_output = iron2.binary_transform(
            up_projection, gate_activated, lambda a, b: a * b, device="npu"
        )

        _ = iron2.matmul(gated_output, down_weight, device="cpu")

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
