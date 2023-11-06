# MLIR-based AI Engine toolchain

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Xilinx/mlir-aie/buildAndTest.yml?branch=main)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr-raw/Xilinx/mlir-aie)

![](https://mlir.llvm.org//mlir-logo.png)

This repository contains an [MLIR-based](https://mlir.llvm.org/) toolchain for AMD AI Engine-based devices, such as [AMD Versal](https://www.xilinx.com/products/technology/ai-engine.html) and [AMD Ryzen AI](https://www.amd.com/en/products/ryzen-ai).  This repository can be used to generate low-level configuration for the AI Engine portion of the device, including AIE processors, stream switches, TileDMA and ShimDMA blocks. Backend code generation is included, targetting the [aie-rt](https://github.com/Xilinx/aie-rt/tree/main-aie) library.  This project is primarily intended to support tool builders with convenient low-level access to devices and enable the development of a wide variety of programming models from higher level abstractions.  As such, although it contains some examples, this project is not intended to represent end-to-end compilation flows or to be particularly easy to use for application design.

[Full Documentation](https://xilinx.github.io/mlir-aie/)

-----
<p align="center">Copyright&copy; 2019-2021 Xilinx</p>
