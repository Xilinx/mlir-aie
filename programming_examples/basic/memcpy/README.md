---
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# **Memcpy**

> **Exercise Design:** The runtime sequence in `memcpy.py` is intentionally left unoptimized — drain operations run serially rather than in parallel, which limits measured bandwidth. Your task is to restructure the runtime sequence using `TaskGroup()` to achieve full concurrency across all columns and channels. See Step 4 below for guidance, and [getting_started/00_memcpy/memcpy.py](../../getting_started/00_memcpy/memcpy.py) for the reference solution.

The `memcpy.py` design is a highly parallel, parameterized design that uses shim DMAs in every NPU column. It enables both compute and bypass modes to help you analyze performance characteristics.

---

### **Exercise: Measure and Analyze Peak NPU Memory Bandwidth with `memcpy`**

In this exercise, you'll use the `memcpy` design to measure memory bandwidth across the full NPU, not just a single AI Engine tile. This is a practical example to evaluate how well a design utilizes available memory bandwidth across multiple columns and channels.

#### **Objective**

* Understand the structure of a full NPU dataflow application using IRON
* Measure and report the peak memory bandwidth achieved
* Experiment with parameters (columns, channels, data size, bypass mode) to study how architecture and design choices affect performance
* Adapt the runtime data movement code to optimize performance

---

## Step-by-Step Instructions

1. **Configure Your Run Using `make` Parameters:**

   Use the Makefile variables to control:

   * `length`: Size of the full transfer in `int32_t` (must be divisible by `columns * channels` and a multiple of 1024)
   * `cols`: Number of NPU columns (≤ 4 for `npu`, ≤ 8 for `npu2`)
   * `chans`: 1 or 2 DMA channels per column
   * `bypass`: Set to `True` to skip the AIE core (DMA-only mode)

   Example:

   ```bash
   make run length=16777216 cols=4 chans=2 bypass=True
   ```

2. **Explore the Two Modes:**

   * **Bypass Mode (`--bypass True`)**: Uses only shim DMA, bypassing the AIE core. This isolates raw memory movement capability.
   * **Passthrough Mode (`--bypass False`)**: Adds a minimal AIE kernel, mimicking a compute+transfer design. Helps understand potential core overhead.

3. **Calculate Effective Bandwidth:**

   In the C++ host code (`test.cpp`), a run is timed between `start` and `stop`:

   ```cpp
   const float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
   ```

   Add the following code to compute bandwidth:

   ```cpp
   double total_bytes = 2.0 * N * sizeof(int32_t); // input and output
   double bandwidth_GBps = total_bytes / (npu_time * 1e-6) / 1e9;
   std::cout << "Effective Bandwidth: " << bandwidth_GBps << " GB/s" << std::endl;
   ```

4. **Ensure Optimal Task Sequencing in the Runtime**

	To achieve full parallelism when draining data from all columns and channels, the `memcpy` design can use **task groups** in the IRON `Runtime` sequence body to group operations and start them together before waiting for completion.

	Modify your IRON runtime sequence body to optimize performance using **task groups**:

	* **Start workers (if not bypassing)** by passing them to the `Program` (they are launched for you — the body does not start them)
	* **Group drain tasks** using `TaskGroup()` so they all begin execution concurrently
	* **Use `tg.finish()`** to explicitly synchronize the completion of the group

	*Key Code Snippet:*

	```python
	def sequence(a_in, b_out, in_hs, out_hs):
	    tg_out = TaskGroup()  # Initialize a group for parallel drain tasks
	
	    # Fill the input FIFOs (these will start immediately)
	    for idx in range(len(in_hs)):
	        in_hs[idx].fill(a_in, taps[idx])
	
	    # Drain the outputs into host buffer and wait for all to finish
	    for idx in range(len(out_hs)):
	        out_hs[idx].drain(
	            b_out,
	            taps[idx],
	            wait=True,
	            group=tg_out,  # Add task to the group
	        )
	
	    tg_out.finish()  # Wait for all drain tasks together
	
	rt = Runtime(
	    sequence,
	    [transfer_type, transfer_type],
	    fn_args=[in_prods, out_conses],
	)
	# Workers (if not bypassing) are launched via the Program, not the body:
	Program(dev, rt, workers=my_workers).resolve_program()
	```

 	*Why This Matters:*

	Without grouping, each `drain(..., wait=True)` call would block serially, and you’d lose concurrency across channels. This would **underutilize the memory system** and give you lower bandwidth measurements.
	
	Using a `TaskGroup` ensures:
	
	* All drain tasks begin concurrently
	* Runtime waits for **all drains to complete together**, preserving parallel execution

	HINT: If you're stuck wondering where to use task groups, see [programming_examples/getting_started/00_memcpy/memcpy.py](../../getting_started/00_memcpy/memcpy.py).

5. **Report Your Findings:**

   * Run experiments across different `cols`, `chans`, and `bypass` settings
   * Record latency and bandwidth
   * Identify the configuration that delivers the highest bandwidth
   * Discuss why bypass mode may or may not outperform passthrough
   * Understand the runtime sequence operations in the generated `memcpy.mlir` file

## Expected Outcome

By the end of this exercise, you should have a solid understanding of:

* How to configure a scalable data movement design across a Ryzen AI NPU
* The impact of architectural parameters on dataflow and bandwidth
* How IRON APIs map to low-level hardware capabilities like DMA engines and ObjectFifos
* The importance of proper transfer scheduling and synchronization of runtime data movement operations
