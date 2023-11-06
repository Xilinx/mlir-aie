# MLIR-based AIEngine toolchain

![](https://mlir.llvm.org//mlir-logo.png)

This repository contains an [MLIR-based](https://mlir.llvm.org/) toolchain for AMD AI Engine-based devices, such as [AMD Versal](https://www.xilinx.com/products/technology/ai-engine.html) and [AMD Ryzen AI](https://www.amd.com/en/products/ryzen-ai).  This repository can be used to generate low-level configuration for the AI Engine portion of the device, including AIE processors, stream switches, TileDMA and ShimDMA blocks. Backend code generation is included, targetting the [aie-rt](https://github.com/Xilinx/aie-rt/tree/main-aie) library.  This project is primarily intended to support tool builders with convenient low-level access to devices and enable the development of a wide variety of programming models from higher level abstractions.  As such, although it contains some examples, this project is not intended to represent end-to-end compilation flows or to be particularly easy to use for application design.

![](dialects.png)

[Getting Started](Building.md)

[Running on a board](Platform.md)

[Github sources](https://github.com/Xilinx/mlir-aie)

Generated code documentation
- [AIE Dialect](AIEDialect.md) - [AIE Passes](AIEPasses.md)
- [AIEX Experimental Dialect](AIEXDialect.md) - [AIEX Experimental Passes](AIEXPasses.md)
- [AIEVec Dialect](AIEVecDialect.md) - [AIEVec Passes](AIEVecPasses.md)
- [ADF Dialect](ADFDialect.md) - [ADF Passes](ADFPasses.md)

Tutorials
- [Step-by-step Tutorial](../tutorials/README.md)
- [AIE Design Patterns](AIEDesignPatterns)
- [AIE Routing](AIERouting)
- [AIE Vectorization of Scalar Code](AIEVectorization)

-----

<p align="center">Copyright&copy; 2019-2023 Advanced Micro Devices, Inc</p>
