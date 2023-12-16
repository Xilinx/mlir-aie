# Platforms

This folder contains different platforms that can be built to test the compiled designs the mlir-aie generates. 
The current platform of choice for AI Engine (v1) is Versal devices on the vck190 board. The two supported platform versions are:

## vck190_bare_es
Bare platforms contain a minimal PL section (clock, reset, BRAM) and a configured PS+AIE+NoC to enable all NoC-to-AIE array connections through shim_dma (aka GMIO). 
This platform is targeted to the ES version of the vck190 board and is built with **Vitis/Vivado 2020.1**.

## vck190_bare_prod
See explanation of 'bare' platform above. 
This platform is targeted to the productio version of the vck190 board and is built with **Vitis/Vivado 2021.2**.
