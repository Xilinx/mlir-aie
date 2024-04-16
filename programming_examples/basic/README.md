<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Explaining the weight expand examples</ins>

The followning designs are different variations of upconverting 4-bit values to `bf16` values. In AIE2, 1024 `bf16` vales, takes 512 Clock Cycles (CC) to move to L1 and 64 CC to load from L1 to the core. A total of 576 CC. 

* [weight_expand_trace_bf16_int4](./weight_expand_trace_bf16_int4) - Every 32 `int4` values shares one `bf16` scaler. Values are unpacked to `int8` and then `int16` to be converted to `bf16` and output is the product of the scaler and the values in `bf16`.
The trace shows 293 CC for upconverting 1024 values. Total size of 1024x0.5B + 32x2B = 576 Bytes. It takes 144 CC to move to L1 and 18 CC to load from L1 to the core. In total: 293+144+18 = 455 CC

* [weight_expand_trace_int16_int4](./weight_expand_trace_int16_int4) - Every 32 `int4` values shares one `int16` scaler. Values are unpacked to `int8` and then `int16`. Output is the product of the scaler and the values in `int16`. The trace shows 117 CC for upconverting 1024 values. Total size of 1024x0.5B + 32x2B = 576 Bytes. It takes 144 CC to move to L1 and 18 CC to load from L1 to the core. In total: 117+144+18 = 279 CC

* [weight_expand_trace_bf16_u4_bf16](./weight_expand_trace_bf16_u4_bf16) - Every 32 `uint4` values shares one `bf16` scaler and one `bf16` minimum. Values are unpacked to `uint8` and then `uint16` to be casted to `int16` and then converted to `bf16` and output is the product of the scaler and the values in `bf16` plus the minimum. The trace shows 383 CC for upconverting 1024 values. Total size of 1024x0.5B + 32x4B = 640 Bytes. It takes 160 CC to move to L1 and 20 CC to load from L1 to the core. In total: 383+160+20 = 563 CC

* [weight_expand_trace_sb](./weight_expand_trace_sb) - Every 32 `uint4` values shares one `int8` scaler and one `int8` minimum. Every 1024 `uint4` values, share one `bf16` scaler and one `bf16` minimum. Values are unpacked to `uint8` and then casted to `int8` and then scaled and added to minimum to be converted to `bf16` and output is the product of the scaler and the values in `bf16` plus the minimum. The trace shows 410 CC for upconverting 1024 values. Total size of 1024x0.5B + 32x2B + 4B = 580 Bytes. It takes 145 CC to move to L1 and 18.125 CC to load from L1 to the core. In total: 410+145+18.125 = 573.125 CC

