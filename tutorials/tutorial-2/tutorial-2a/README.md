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

From the previous introduction in [tutorial-2](..), the first step is to initialize and configure our design (whether it be for simulation or hardware implementaion). Our build utility `aiecc.py` had actually already done part of this for us since `aie.mlir` specifies a design for the entire AI Engine array. As a result, `aiecc.py` has already generated a directory with the API libraries necessary to initialize, configure and interact with our design. These can be found in the `acdc_project` folder after you run `aiecc.py`. 

Of particular interest is the `acdc_project/aie_inc.cpp` file which contains many of the custom API functions for our design. These functions can be invoked from host code to initialize, configure and control our AI Engine design. Additional common host code API functions for testing can also be found in the [runtime_lib/test_library.cpp](../../../runtime_lib/test_library.cpp).

## <ins>Tutorial 2a Lab</ins>

1. Run `make` in this directory. Then take a look at the file under `acdc_project/aie_inc.cpp`. You will see a number of helpful API functions for initializing and configuring our design from the host. We've built a host program that does this named [test.cpp](./test.cpp). Take a look at this file to see the common set of init/config functions which will be describing in more detail below:

### <ins>Common Host API init/config functions</ins>
The next set of functions are often called as a group to configure the AI Engine array and program the elf files for each individual tile. 

| Host Config API | Description |
|----------|-------------|
| mlir_aie_init_libxaie | Instantiates a struct of configuration information and data types used by later configuration functions. |
| mlir_aie_init_device | Initializes our AI Engine array and reserves the entire array for configuration. |
| mlir_aie_configure_cores | Disables and resets all relevant tiles and loads elfs into relevant tiles. It also releases all locks to value 0. |
| mlir_aie_configure_switchboxes | Configures the switchboxes use to route stream connections. |
| mlir_aie_configure_dmas | Configures all tile DMAs |
| mlir_aie_initialize_locks | Configures initial lock values as applicable. |
| mlir_aie_clear_tile_memory | Clear tile data memory for a given tile. Call for each tile you wish to clear the tile memory for. |


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
Following this code block, the only components that are not yet configured are the shim DMAs because they require some parameters obtained at runtime. In the case of our example vck190 platform, to configure the shimDMA using the libxaiengine drivers, we pass in a virtual address to DDR. The general order that we call our host config APIs for shim DMA configuration are:
1. Allocate buffers
2. Configure shim DMAs
3. Synchronize DDR cache (virtual address) with DDR physical memory as needed

The dynamic buffer allocation and shim DMA config function calls could look like:
```
  // Set aside N DDR buffers (here, N=2)
  mlir_aie_init_mems(_xaie, 2);

  // Allocate buffer and return virtual pointer to memory
  int *mem_ptr_in = mlir_aie_mem_alloc(_xaie, 0, 256);  // buffer 0, 256 bytes
  int *mem_ptr_out = mlir_aie_mem_alloc(_xaie, 1, 256); // buffer 1, 256 bytes

  // Set virtual pointer used to configure shimDMA
  mlir_aie_external_set_addr_ddr_test_buffer_in((u64)mem_ptr_in);
  mlir_aie_external_set_addr_ddr_test_buffer_out((u64)mem_ptr_out);
  mlir_aie_configure_shimdma_70(_xaie);

  ... 
  mlir_aie_sync_mem_dev(_xaie, 0); // Sync cache data with physical data (write)
  mlir_aie_sync_mem_cpu(_xaie, 0); // Sync physical data with cache data (read)
```
| Host Config API | Description |
|----------|-------------|
| mlir_aie_init_mems | Initialize N DDR memory buffers. At the moment, we need to know the number of buffers we need up front. |
| mlir_aie_mem_alloc | Dynamic allocation of memory buffer associated with buffer ID number and a size. The ID is sequential starting from 0 and matches the description as defined in aie.mlir |
| mlir_aie_external_set_addr_< symbol name > | Set the address used in configuring the shim DMA (associated with an external buffer)|
| mlir_aie_configure_shimdma_< location > | Complete shim DMA configuration given the virtual address value set by mlir_aie_external_set_addr_. There is one of these for every shim DMA tile but can encompas up to 4x DMAs in that tile. |
| mlir_aie_sync_mem_dev | Synchronize between DDR cache (virtual address) and DDR physical memory accessed by NOC/ shimDMA. In simulation, we explicitly copy from host memory to memory region accessed by shim DMA model. We call this after we update DDR data and want the shim DMA to see the new data. |
| mlir_aie_sync_mem_cpu | Sycnhronize between DDR physical memory accessed by NOC/ shimDMA and DDR cache (virtual address). In simulation, we explicitly copy from shim DMA model accessible memory to host memory. we call this before we read DDR data to make sure shim DMA written data is updated. |

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

    mlir_aie_sync_mem_cpu(_xaie, 1); // Sync output buffer back to DDR/cache

    // Check buffer at index 5 again for expected value of 114 for tile(3,4)    
    mlir_aie_check("After start cores:", mlir_aie_read_buffer_a34(_xaie, 5), 114,
                   errors);
    mlir_aie_check("After start cores:", mem_ptr_out[5], 114, errors);
```

| Host Config API | Description |
|----------|-------------|
| mlir_aie_start_cores | Unresets and enables all tiles in the design.|
| mlir_aie_print_tile_status | Prints out tile status to stdout |
| mlir_aie_print_shimdma_status | Prints out shim DMA status to stdout |
| mlir_aie_release_< symbolic lock name > | Release lock for a given lock based on the symbolic lock name and lock value. |
| mlir_aie_acquire_< symbolic lock name > | Acquire lock for a given lock based on the symbolic lock name, lock value, and timeout value. |
| mlir_aie_read_buffer_< symbolic buffer name > | Read buffer from tile local memory based on the symbolic buffer name. |
| mlir_aie_write_buffer_< symbolic buffer name > | Write value to buffer in tile local memory based on the symbolic buffer name. |
| mlir_aie_check | Check between a value and the expected value and output the prepended error message and increment error variable if difference is found. |

You will notice that in our example, we use the `mlir_aie_acquire_< symbolic lock name >` to gate when the execution of our design is done. This is facilitated by having added locks around the kernel code. While this is not strictly necessary, it is a helpful technique for checking when a AI engine core is done. We will explore this in future tutorials as locks can be used in various ways to control data communication and gate operations of AI engines.

2. Look again in the main directory and you will see several key files including `core_1_4.elf` which is the elf program for tile(1,4). You also will see `tutorial-2a.exe` which is the cross-compiled host binary generated from the host code source `test.cpp`.
3. Take a look at [Makefile](./Makefile) and notice the additional arguments added to `aiecc.py`. We've included the `test_library.cpp` and the host code source [test.cpp](./test.cpp) and specified the final host executable output with a -o tutorial-2a.exe.

With all these host API calls, we can build a complete host program that configures and initializes our AI Engine design, enable it to run, and check for results. In [tutorial-2b](../tutorial-2b), we go through the steps of running a simulation of this design. In [tutorial-2c](../tutorial-2c), we run our design on the board and describe how to measure performance in hardware.

