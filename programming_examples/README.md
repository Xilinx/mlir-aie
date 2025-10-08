<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Programming Examples</ins>

These programming examples are provided so that application programmers can learn how to leverage the IRON design flow with mlir-aie python bindings, and the mlir-aie intermediate representation directly to build applications targeting AI Engines. 

Each IRON example has one or more implementations:
* `<example_name>.py` - These designs are generally written using a higher-level version of IRON
* `<example_name>_placed.py` - These designs are generally written using a lower-level verion of IRON

They are organized into the following directories:

## [getting_started](./getting_started) 

Designs tailored to the new user experience that span from basic applications such as SAXPY to more complicated ones such as tiled matrix multiplication, for the NPU in Ryzen™ AI.

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