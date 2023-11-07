# MLIR-based AIEngine toolchain

![](https://mlir.llvm.org//mlir-logo.png)

This repository contains an [MLIR-based](https://mlir.llvm.org/) toolchain for AMD AI Engine-enabled devices, such as [AMD Ryzen™ AI](https://www.amd.com/en/products/ryzen-ai) and [AMD Versal™ ](https://www.xilinx.com/products/technology/ai-engine.html).  This repository can be used to generate low-level configurations for the AI Engine portion of these devices. AI Engines are organized as a spatial array of tiles. Each tile contains AI Engine cores and/or memories. The spatial array is connected by stream switches that can be configured to route data between AI Engine tiles scheduled by their programmable Data Movement Accelerators (DMAs). This repository contains MLIR representations, with multiple levels of abstraction, to target AI Engine devices. This enables compilers and developers to program AI Engine cores, as well as describe data movements and array connectivity. A Python API is made available as a convenient interface for generating MLIR design descriptions. Backend code generation is also included, targetting the [aie-rt](https://github.com/Xilinx/aie-rt/tree/main-aie) library.  

This project is primarily intended to support the open-source community, particularily tool builders, with low-level access to AIE devices and enable the development of a wide variety of programming models from higher level abstractions.  As such, although it contains some examples, this project is not intended to represent an end-to-end compilation flow for application design. If you're looking for an out-of-the-box experience for highly efficient machine learning, check out the [AMD Ryzen™ AI Software Platform](https://github.com/amd/RyzenAI-SW/)

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
