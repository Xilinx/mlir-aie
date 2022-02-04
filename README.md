# MLIR-based AIEngine toolchain

![](https://mlir.llvm.org//mlir-logo.png)

This repository contains an [MLIR-based](https://mlir.llvm.org/) toolchain for Xilinx Versal AIEngine-based devices.  This can be used to generate low-level configuration for the AIEngine portion of the device, including processors, stream switches, TileDMA and ShimDMA blocks. Backend code generation is included, targetting the LibXAIE library.  This project is primarily intended to support tool builders with convenient low-level access to devices and enable the development of a wide variety of programming models from higher level abstractions.  As such, although it contains some examples, this project is not intended to represent end-to-end compilation flows or to be particularly easy to use for system design.

[Getting Started](Building.md)

[Running on a board](Platform.md)

[Github sources](https://github.com/Xilinx/mlir-aie)

Generated code documentation
- [AIE Dialect](AIEDialect.md) [AIE Passes](AIEPasses.md)
- [AIEVec Dialect](AIEVecDialect.md) [AIEVec Passes](AIEVecPasses.md)
- [ADF Dialect](ADFDialect.md) [ADF Passes](ADFPasses.md)

Tutorials
- [AIE Design Patterns](AIEDesignPatterns)
- [AIE Routing](AIERouting)

-----

<p align="center">Copyright&copy; 2019-2021 Xilinx</p>
