# EXPERIMENTAL: Multi-Head Attention (MHA) & Grouped-Query Attention (GQA)

Once MLIR-AIE and IRON are installed, you can run the design with the following command:
```
cmake -B build
cmake --build build --target run
```

The configuration of the MHA & GQA is at the top of the `CMakeLists.txt`. The current configuration is from the LLama3.2B model with 8 heads and a KV group size of 4.

Configuration parameters:
- `num_KV_heads`: The number of KV heads to group together, if 0 or 1, we perform a MHA
- `S_q`: Sequence length for Q
- `S_kv`: Sequence length for K and V
- `heads`: Number of heads
- `d`: The head dimension

Disclaimer: This is a WIP example and not all dimensions are supported!