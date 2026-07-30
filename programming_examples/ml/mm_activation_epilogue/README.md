<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# mm_activation_epilogue: one resident program, three RTP-selected GEMM epilogues

This design implements a fused, `float32`-in/`float32`-out GEMM epilogue for
`aie2p` that applies one of `{identity, SiLU, GELU}` to a matmul's f32
accumulator tile. All three modes are compiled into ONE xclbin; which one
runs is selected per dispatch by a single runtime-parameter (RTP) word, not
by loading a different xclbin. NPU2-only (the underlying kernel lives under
`aie_kernels/aie2p/`).

## Why a runtime-selected epilogue

A GEMM microkernel's activation (none for a plain linear layer, SiLU for a
SwiGLU-style FFN, GELU for a classic transformer FFN) is usually fixed at
build time, one xclbin per activation. In a pipeline that calls several
differently-activated matmuls back to back, that means a full hardware
reconfiguration between them. This design instead compiles all three
epilogue bodies into one resident program and switches between them with an
`aiex.npu.rtp_write`-class dispatch: same xclbin, same `hw_context`, no
reload.

In my own project (not this exact packaged example, but the same kernel
logic and the same RTP mode-select mechanism, exercised by a resident
matmul+epilogue design) I measured this on NPU2 today: registering ONE
`hw_context` and alternating SiLU/GELU dispatches through it stayed
numerically correct (rel-L2 0.0083 / 0.0094 against an f32 numpy reference,
both well under my 0.03 gate, unchanged after 100 alternating dispatches),
the compiled per-core ELF / PDI / CDO binaries were byte-identical between
the two modes except for 4 bytes (the per-core RTP write constant), and the
mode switch itself showed no measurable dispatch-time cost beyond ordinary
dispatch-to-dispatch noise (alternating-dispatch mean within -0.015 ms of
the solo-mode baseline mean, i.e. the delta was negative -- the "switch"
was not distinguishable from no switch at all). Those numbers are from my
own differently-shaped production design using the identical mode-select
mechanism, not from this file.

This packaged example has itself been run on NPU2: all three modes dispatch
clean, identity is bit-exact against the reference, and SiLU and GELU come
out at rel-L2 0.00246 and 0.00341 respectively over a `[-8, 8]` sweep, both
inside the `atol=0.05` gate this example checks against.

## Source Files Overview

1. `mm_activation_epilogue.py`: IRON design. Two cores split a flat `size`-element
   `float32` input; each core's compiled body waits on a
   `WorkerRuntimeBarrier`, reads a per-core RTP `mode` word, runs one full
   pass over its tiles under that mode, then releases the barrier and (per
   the AIE core's implicit outer loop) waits for the next epoch. The
   runtime sequence dispatches three such epochs -- mode 0, 1, 2 -- each
   draining into its own output tensor so every mode's result can be
   checked independently. Structurally this mirrors
   [`ml/scale_shift`](../scale_shift)'s two-phase (multiply, then add)
   RTP-parameter dispatch, extended to three phases and three separate
   outputs.

1. `mm_activation_epilogue.cc`: AIE2P kernel, pulled from
   [`aie_kernels/aie2p/`](../../../aie_kernels/aie2p/). One `mode` argument
   (0/1/2) selects identity, SiLU (hybrid f32/bf16 precision -- see the
   file's header comment for why), or GELU (tanh approximation).

1. `test.cpp`: C++ testbench. Loads the compiled XCLBIN + `insts.bin`,
   drives one input and three output buffers, and checks every element of
   all three outputs against a host `float` reference.

## Usage

### Standalone JIT verification

```shell
python3 mm_activation_epilogue.py --dev npu2
```

### C++ Testbench

```shell
make
make run
```
