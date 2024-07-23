<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Passthrough DMAs with PLIO</ins>

This reference design can be run on the VCK5000 Versal device. This design leverages the same data movement pattern as the [Passthrough DMAs](../passthrough-dmas) example design but it uses a soft DMA. Please see the [platforms repo](https://github.com/Xilinx/ROCm-air-platforms) for more information on how the programmable logic is integrated with the AIEs. This is meant to be an illustrative example to highlight how to integrate PL designs with AIE designs programmed using mlir-aie.

In the platform, tile (26, 0) has PLIO connected to a DMA implemented in the programmable logic. There are two designs, `aie2-input-plio.py` uses the soft DMA to push data from DRAM into the AIEs, wheras `aie2-output-plio.py` uses the soft DMA to receive data from the AIEs and push it to DRAM. The soft DMA is programmed using the same mechanism as the ShimDMAs.

In the [design](./aie2.py) data is brought from external memory to `ComputeTile2` and back, without modification from the tile, by using an implicit copy via the compute tile's Data Movement Accelerator (DMA). The data is read from and written to external memory through the Shim tile (`col`, 0).

The implicit copy is performed using the `object_fifo_link` operation that specifies how input data arriving via `of_in` should be sent further via `of_out` by specifically leveraging the compute tile's DMA. This operation and its functionality are described in more depth in [Section-2b](../../../programming_guide/section-2/section-2b/03_Link_Distribute_Join/README.md#object-fifo-link) of the programming guide.


To compile and run the design for VCK5000:
```
make all
./output.elf // To run the kernel which outputs over PLIO
./input.elf // To run the kernel which inputs over PLIO
```
