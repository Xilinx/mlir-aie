<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->
# <ins>Tutorial 2a - Host code configuration</ins>

From the previous introduction in [tutorial-2](..), the first step is to initialize and configure our design (whether it be for simulation or hardware implementaion). Our build utility `aiecc.py` had actually already done part of this for us since `aie.mlir` specifies a design for the entire AI Engine array. As a result, `aiecc.py` has already generated a directory with the host API libraries necessary to initialize, configure and interact with our design. These can be found in the `aie.mlir.prj` folder after you run `aiecc.py`. 

Of particular interest is the `aie.mlir.prj/aie_inc.cpp` file which contains many of the configuration API functions for our design. These functions can be invoked from host code to initialize, configure and control our AI Engine design. Additional common host code API functions for testing can also be found in the [runtime_lib/test_lib/test_library.h](../../../runtime_lib/test_lib/test_library.h).

## <ins>Tutorial 2a Lab</ins>

1. Run `make` in this directory. Then take a look at the file under `aie.mlir.prj/aie_inc.cpp`. You will see a number of helpful API functions for initializing and configuring our design from the host. We've built a host program that does this named [test.cpp](./test.cpp). Take a look at this file to see the common set of init/config functions which will be describing in more detail below:

### <ins>Host Code Configuration API</ins>
The set of functions described below are often called as a group to configure the AI Engine array and program the elf files for each individual tile. Detailed descriptions of these functions can be found in the common `test_library.cpp` and design specific `aie.mlir.prj/aie_inc.cpp`.

| Host Config API | Description |
|----------|-------------|
| aie_libxaie_ctx_t | Struct of AI Engine configuration information and data types use by configuration functions |
| mlir_aie_init_libxaie () | Allocates and initializes aie_libxaie_ctx_t struct |
| mlir_aie_init_device (_xaie) | Initializes our AI Engine array and reserves the entire array for configuration. |
| mlir_aie_configure_cores (_xaie) | Disables and resets all relevant tiles and loads elfs into relevant tiles. It also releases all locks to value 0. |
| mlir_aie_configure_switchboxes (_xaie) | Configures the switchboxes used to route stream connections. |
| mlir_aie_configure_dmas (_xaie) | Configures all tile DMAs |
| mlir_aie_initialize_locks (_xaie) | Configures initial lock values (placeholder). |
| mlir_aie_clear_tile_memory (_xaie, int col, int row) | Clear tile data memory for a given tile. Call for each tile you wish to clear the tile memory for. |


Instantiating the above as code block would look something like:
```
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);

  mlir_aie_clear_tile_memory(_xaie, 1, 4); // clear local data memory for tile(1,4)
```
Following this code block, the only components that are not yet configured are the shim DMAs because they require some parameters obtained at runtime. In the case of our example vck190 platform, to configure the shim_dma using the libxaiengine drivers, we pass in a virtual address to a block of memory in DDR . The general step order that we call our host config APIs for shim DMA configuration are:
1. Allocate buffers
2. Configure shim DMAs
3. Synchronize DDR cache (virtual address) with DDR physical memory as needed

The dynamic buffer allocation and shim DMA config function calls would look like:
```
  // Allocate buffers and return virtual pointers to these buffers
  ext_mem_model_t buf0, buf1;
  int *mem_ptr_in = mlir_aie_mem_alloc(_xaie, buf0, 256);
  int *mem_ptr_out = mlir_aie_mem_alloc(_xaie, buf1, 256);

  // Set virtual pointer used to configure shim_dma
  mlir_aie_external_set_addr_ddr_test_buffer_in(_xaie, (u64)mem_ptr_in);
  mlir_aie_external_set_addr_ddr_test_buffer_out(_xaie, (u64)mem_ptr_out);
  mlir_aie_configure_shimdma_70(_xaie); // configures 2 DMAs in shim DMA tile

  ... 
  mlir_aie_sync_mem_dev(_xaie, buf0); // Sync cache data with physical data (write)
  mlir_aie_sync_mem_cpu(_xaie, buf0); // Sync physical data with cache data (read)
```
| Host Config API | Description |
|----------|-------------|
| mlir_aie_mem_alloc (_xaie, ext_mem_model_t buf, int size) | Dynamic allocation of memory buffer associated with buffer handle (buf) and a size.  Size is defined in words (4 bytes). |
| mlir_aie_external_set_addr_< symbol name > (u64 addr) | Set the address (addr) for an MLIR-AIE external buffer used in configuring the shim DMA. This address is usually the virtual address when working with the backend linux kernel drivers. |
| mlir_aie_configure_shimdma_< location > (_xaie) | Complete shim DMA configuration given the virtual address value set by mlir_aie_external_set_addr_ for all DMAs belonging to this shim_dma tile (up to 4). |
| mlir_aie_sync_mem_dev (ext_mem_model_t buf)| Synchronize between DDR cache (virtual address) and DDR physical memory accessed by NOC/ shim_dma, i.e, flush the DDR cache. In simulation, we explicitly copy from host memory to the memory region accessed by shim DMA model. We call this after we update DDR data and want the shim DMA to see the new data. |
| mlir_aie_sync_mem_cpu (ext_mem_model_t buf)| Synchronize between DDR physical memory accessed by NOC/ shim_dma and DDR cache (virtual address), i.e., invalidate the DDR cache. In simulation, we explicitly copy from shim DMA model accessible memory to host memory. We call this before we read DDR data to make sure shim DMA written data is updated before being read by the host program. |

> **More information about "syncing" memory**: In a system with caches, the data in the cache can be out of date with the data in global memory.  If an accelerator has coherent access to the caches, then this inconsistency isn't a problem because the accelerator will be able to access data in the cache directly and the cache will do the right thing.  This is called "shared virtual memory".  However, in a system where the accelerator only sees global memory and does not have cache-coherent view of data in the caches, we need to make sure that the accelerator 'sees' the data in the cache, when it can only access global memory.  The solution to this is to explicitly flush or invalidate data in the caches.  Normally this would be taken care of in the operating system, but this is a high-latency operation.  In this case, we do it in userspace:  `mlir_aie_sync_mem_dev` is essentially "flush the processor caches so that the device can see data" and `mlir_aie_sync_mem_cpu` is essentially "invalidate the processor caches so that the processor will reread data from global memory".

Finally, we are ready to start the cores and poll and test values to ensure correct functionality. An example sequence of host commands might look like:
```
    mlir_aie_start_cores(_xaie);

    printf("\ntile(3,4) status\n");
    mlir_aie_print_tile_status(_xaie, 3, 4);

    printf("\ntile(7,0) status\n");
    mlir_aie_print_shimdma_status(_xaie, 7, 0);

    printf("Release ddr input/output locks(1) to enable them\n");
    mlir_aie_release_ddr_test_buffer_in_lock(_xaie, 1, 0);
    mlir_aie_release_ddr_test_buffer_out_lock(_xaie, 1, 0);

    // Wait for shim DMA output lock
    if(mlir_aie_acquire_ddr_test_buffer_out_lock(_xaie, 0, timeout) == XAIE_OK)
      printf("Acquired ddr output lock(0). Output shim dma done.\n");
    else
      printf("Timed out (%d) while trying to acquire ddr output lock (0).\n", timeout);

    mlir_aie_sync_mem_cpu(_xaie, buf1); // Sync output buffer back to DDR/cache

    // Check buffer at index 5 again for expected value of 114 for tile(3,4)    
    mlir_aie_check("After start cores:", mlir_aie_read_buffer_a34(_xaie, 5), 114,
                   errors);
    mlir_aie_check("After start cores:", mem_ptr_out[5], 114, errors);
```

| Host Config API | Description |
|----------|-------------|
| mlir_aie_start_cores (_xaie)| Unresets and enables all tiles in the design.|
| mlir_aie_print_tile_status (_xaie, int col, int row) | Prints out tile status to stdout |
| mlir_aie_print_shimdma_status (_xaie, int col, int row) | Prints out shim DMA tile status to stdout |
| mlir_aie_acquire_< symbolic lock name > (_xaie, int value, int timeout)| Acquire lock for a given lock associated with a symbolic lock name with the lock value (value, 0 or 1). Timeout is the amount of time in microseconds that the operation waits to succeed. This acquire is non-blocking. If timeout is set to 0, we check once and return. If set to a positive integer value, we poll the lock for that period of time and then return after the timeout value. |
| mlir_aie_release_< symbolic lock name > (_xaie, int value, int timeout) | Release lock for a given lock associated with a symbolic lock name to the lock value (value, 0 or 1). Timeout is the amount of time in microseconds that the operation waits to succeed. |
| mlir_aie_read_buffer_< symbolic buffer name > (_xaie, int index) | Read buffer from tile's local memory based on the symbolic buffer name at the index offset (index). |
| mlir_aie_write_buffer_< symbolic buffer name > (_xaie, int index, int value) | Write value to buffer in tile's local memory based on the symbolic buffer name with value (value) at the index offset (index). |
| mlir_aie_check (s, r, v, errors) | Macro definition to check between a value (v) and the expected reference value (r). If an inequality is found, the output error message include the string (s) and we increment (errors). |

You will notice that in our example, we use the `mlir_aie_acquire_< symbolic lock name >` to gate when the execution of our design is done. This is facilitated by having added locks around the kernel code. While this is not strictly necessary, it is a helpful technique for checking when a AI engine core is done. We will explore this in future tutorials as locks can be used in various ways to control data communication and gate operations of AI engines.

2. Look again in the main directory and you will see several key files including `core_1_4.elf` which is the elf program for tile(1,4). You also will see `tutorial-2a.exe` which is the cross-compiled host executable generated from the host code source `test.cpp`.
3. Take a look at [Makefile](./Makefile) and notice the additional compile arguments added to `aiecc.py`. We've included the `test_library.cpp` and the host code source [test.cpp](./test.cpp) and specified the final host executable output with a `-o tutorial-2a.exe`.

With all these host API calls, we can build a complete host program that configures and initializes our AI Engine design, enable it to run, and check for results. In [tutorial-2b](../tutorial-2b), we go through the steps of running a simulation of this design. In [tutorial-2c](../tutorial-2c), we run our design on the board and describe how to measure performance in hardware.

