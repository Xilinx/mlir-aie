--- 

# **Memcpy**

The `memcpy.py` design is a highly parallel, parameterized design that uses shim DMAs in every NPU column. It enables both compute and bypass modes to help you analyze performance charactaristics.

---

### **Exercise: Measure and Analyze Peak NPU Memory Bandwidth with `memcpy`**

In this exercise, you'll use the `memcpy` design to measure memory bandwidth across the full NPU, not just a single AI Engine tile. This is a practical example to evaluate how well a design utilizes available memory bandwidth across multiple columns and channels.

#### **Objective**

* Understand the structure of a full NPU dataflow application using IRON
* Measure and report the peak memory bandwidth achieved
* Experiment with parameters (columns, channels, data size, bypass mode) to study how architecture and design choices affect performance
* Adapt the runtime data movement code to optimize performance

---

### **Step-by-Step Instructions**

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

   * **Bypass Mode (`--bypass True`)**: Uses only shim DMA, bypassing the AIE core. This isolates raw memory movement capability
   * **Passthrough Mode (`--bypass False`)**: Adds a minimal AIE kernel, mimicking a compute+transfer design. Helps understand potential core overhead

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

	To achieve full parallelism when draining data from all 	columns and channels, the `memcpy` design can use **task groups** in the IRON 	`Runtime().sequence` to group operations and start them together before waiting for completion.

	Modify your IRON runtime sequences to optimize performance using **task groups**:

	* **Start workers (if not bypassing)** before enqueueing transfers
	* **Group drain tasks** using `task_group()` so they all begin execution concurrently
	* **Use `finish_task_group()`** to explicitly synchronize the completion of the group

	*Key Code Snippet:*

	```python
	rt = Runtime()
	with rt.sequence(transfer_type, transfer_type) as (a_in, b_out):
	    if not bypass:
	        rt.start(*my_workers)
	
	    tg_out = rt.task_group()  # Initialize a group for parallel drain tasks
	
	    # Fill the input FIFOs (these will start immediately)
	    for i in range(num_columns):
	        for j in range(num_channels):
	            rt.fill(
	                of_ins[i * num_channels + j].prod(),
	                a_in,
	                taps[i * num_channels + j],
	            )
	
	    # Drain the outputs into host buffer and wait for all to finish
	    for i in range(num_columns):
	        for j in range(num_channels):
	            rt.drain(
	                of_outs[i * num_channels + j].cons(),
	                b_out,
	                taps[i * num_channels + j],
	                wait=True,
	                task_group=tg_out,  # Add task to the group
	            )
	
	    rt.finish_task_group(tg_out)  # Wait for all drain tasks together
	```

 	*Why This Matters:*

	Without grouping, each `drain(..., wait=True)` call would block serially, and you’d lose concurrency across channels. This would **underutilize the memory system** and give you lower bandwidth measurements.
	
	Using a `task_group` ensures:
	
	* All drain tasks begin concurrently
	* Runtime waits for **all drains to complete together**, preserving parallel execution

5. **Report Your Findings:**

   * Run experiments across different `cols`, `chans`, and `bypass` settings
   * Record latency and bandwidth
   * Identify the configuration that delivers the highest bandwidth
   * Discuss why bypass mode may or may not outperform passthrough
   * Understand the runtme sequence operations in the generated `memcpy.mlir` file

---

### **Expected Outcome**

By the end of this exercise, you should have a solid understanding of:

* How to configure a scalable data movement design across a Ryzen AI NPU
* The impact of architectural parameters on dataflow and bandwidth
* How IRON APIs map to low-level hardware capabilities like DMA engines and object FIFOs
* The importance of proper transfer scheduling and synchronization of runtime data movement operations
