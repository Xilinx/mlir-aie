There are two ways to use precompiled functions in your code.

## 1) Precompiled kernel (tests 02 and 04)

In this method, the *kernel.cc* manages the locks and is compiled to an elf file, which can be used to program the AIE.

You can modify and recompile the *kernel.cc* without recompiling the host.

## 2) Precompiled core function (tests 01, 03, 05, and 07)

In this method, locks are managed in aie.mlir and the *kernel.cc* is compiled to an objective file.

Both the host and the *kernel.cc* should be recompiled after modifying the *kernel.cc*. 
