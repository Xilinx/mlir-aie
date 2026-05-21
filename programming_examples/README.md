<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Programming Examples</ins>

These programming examples are provided so that application programmers can learn how to leverage the IRON design flow with mlir-aie python bindings, and the mlir-aie intermediate representation directly to build applications targeting AI Engines. 

Most examples are a single `<example_name>.py` design driven by `@iron.jit` — one file describes the AIE-array dataflow, JIT-compiles to xclbin/insts, and runs end-to-end (or feeds the prebuilt artifacts to a C++ host).  A few examples additionally provide an `<example_name>_placed.py` variant written against a lower-level form of IRON for the cases where explicit tile/core placement is the pedagogical point.

They are organized into the following directories:

## [getting_started](./getting_started)

Designs tailored to the new user experience that span from basic applications such as SAXPY to more complicated ones such as tiled matrix multiplication, for the NPU in Ryzen™ AI.

## [algorithms](./algorithms)

Higher-level algorithm templates (transform, for_each, and parallel variants) that handle Workers, ObjectFIFOs, and data movement automatically for common element-wise dataflow patterns on the NPU in Ryzen™ AI.

## [basic](./basic)

Basic building blocks to understand the NPU architecture and first steps towards building applications for the NPU in Ryzen™ AI. 

## [ml](./ml)

Machine learning building blocks, design components, and reference designs for the NPU in Ryzen™ AI. 

## [vision](./vision)

Computer vision processing pipeline designs for the NPU in Ryzen™ AI.

## [mlir](./mlir)

MLIR-based reference designs expressed in the MLIR-AIE dialect intermediate representation.

## [utils](./utils)

Utilty functions leveraged in the programming examples. 