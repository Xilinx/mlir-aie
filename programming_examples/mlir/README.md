<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>MLIR Examples</ins>

These examples illustrate how AIE designs are expressed at the MLIR level, which is the intermediate representation that the Python IRON API compiles down to. Reading these files alongside the [programming guide](../../programming_guide/) can provide insight into what the higher-level abstractions generate.

## Examples

* [MM_2x2](./MM_2x2/) - Matrix multiplication mapped onto a 2×2 array of AIE cores, in circuit-switched, packet-switched, and ObjectFIFO variants. Targets Versal VCK5000.

* [horizontal_diffusion](./horizontal_diffusion/) - Implementation of the horizontal diffusion stencil computation from the COSMO atmospheric model, demonstrating multi-core data streaming across AIE tiles. Published at ICS 2023. Targets Versal hardware.

* [autocorrelation](./autocorrelation/) - Autocorrelation of a signal vector across AIE cores using explicit DMA programming.

* [idct](./idct/) - Inverse Discrete Cosine Transform (IDCT) kernel for image/video processing.

* [prime_sieve_large](./prime_sieve_large/) - Sieve of Eratosthenes for finding prime numbers, demonstrating a dataflow pipeline across multiple AIE tiles.
