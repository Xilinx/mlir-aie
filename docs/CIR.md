# aie++: a C++ DSL for spatial heterogeneous computing based on ClangIR CIR C++ front-end

Programming modern heterogeneous devices requires usually to write different
programs for different parts of the hardware, sometimes using different
programming language. This makes things very complicated with a lot of
boiler-plate details around the various programs.

This project proposes using a single-source C++-based DSL to express in a single
program all the details necessary for programming the heterogeneous device. This
simplify programming in a seamless way and the compiler can infer more details
as there is only main compiler with a holistic view of the program.

While the concept is very general, we start applying it as AIE++ to MLIR
AIE/IRON as a prototyping phase with the goal of generalizing it as a
configurable framework to handle for example MLIR AIR and spensors (to be
sensored on-line).

Since most of ACDC software stack is MLIR-based, we are using [ClangIR](https://github.com/llvm/clangir) to generate some MLIR from C/C++. ClangIR is a
fork of Clang generating MLIR CIR dialect during its code generation and sounds
like the most promising approach since it is in the process of being up-streamed
to Clang/LLVM/MLIR and it is backed by several companies.

The current AIE++ prototype has 2 components:

- [ClangIR for AIE++](https://github.com/llvm/clangir/pull/1334) which has some
  extensions on MLIR standard dialect lowering and specific features to support
  AIE tool-chain as a client. Some of the commits in this long-running branch
  have already been up-streamed to ClangIR but also to the LLVM project.

- AIE++ with ClangIR, which is an [ClangIR MLIR AIE
  version](https://github.com/Xilinx/mlir-aie/pull/1913) using ClangIR/LLVM/MLIR
  specific version as a client.


## Using the framework

This is a huge work-in-progress due to the fact that ClangIR for MLIR standard
dialects is also a huge work-in-progress.

There is no end-to-end working flow or integration with
[`aiecc.py`](../python/compiler/aiecc/main.py) yet.

There is a script running the compilation flow for the device part of the
program and leaving a local file for each intermediate phase for inspection.

A typical use case is:

```bash
PATH=$LLVM_DIR/build/bin:$MLIR_AIE_HOME/build/bin:$PATH $MLIR_AIE_HOME/utils/aie++-compile.sh example.cpp
```

which might generate locally the files

```
example.cir
example.prepare.aie.cir
example.aie.cir
example.aie.lambda.cir
example.aie.decapture.cir
example.aie.only.cir
example.aie.mlir
```

with the goal of the last `.aie.mlir` file being fed into `aiecc.py`.

To help navigating with the code, the project provides an [LSP
server](../tools/aie-lsp-server) to be used by an IDE (VS Code...) or editor
(Emacs...). This LSP server is knowledgeable of the CIR dialect too.

The C++ `aie++` user source code is just plain C++ code and can benefit from a
standard C++ LSP server like [`clangd`](https://clangd.llvm.org/) with
auto-completion with some AIE knowledge for free.


## Building the framework

The environment should be configured according to where the various tools are
installed according to the main project [README](../README.md) with something
like:

```bash
export LLVM_DIR=.../clangir
export MLIR_AIE_HOME=.../mlir-aie
export PEANO_DIR=.../peano
# To speed-up (re)compilation with ccache
export PATH="/usr/lib/ccache:$PATH"
export XILINX_VITIS=/opt/xilinx/Vitis/2023.2
export XILINX_XRT=/opt/xilinx/xrt
PATH=$PATH:$XILINX_XRT/bin:$XILINX_VITIS/aietools/bin:$XILINX_VITIS/bin
# Put the system libs before XILINX otherwise it breaks OpenSSL for git
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu:/usr/lib:$XILINX_XRT/lib:$XILINX_VITIS/lib/lnx64.o
cd $MLIR_AIE_HOME
# Convigure the venv Python environment
source utils/setup_python_packages.sh
rehash
```

Compiling the specific [ClangIR for MLIR AIE
version](https://github.com/keryell/clangir/tree/mlir-aie-version):

```bash
cmake -GNinja \
 -DCLANG_ENABLE_CIR=ON \
 -DCLANG_INCLUDE_DOCS=ON \
 -DCMAKE_BUILD_TYPE=Debug \
 -DCMAKE_C_COMPILER=clang \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DCMAKE_INSTALL_PREFIX=$LLVM_DIR/../install-llvm \
 -DLLVM_BUILD_DOCS=ON \
 -DLLVM_BUILD_EXAMPLES=ON \
 -DLLVM_BUILD_LLVM_DYLIB=ON \
 -DLLVM_BUILD_UTILS=ON \
 -DLLVM_ENABLE_ASSERTIONS=ON \
 -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
 -DLLVM_ENABLE_RTTI=ON \
 -DLLVM_INCLUDE_DOCS=ON \
 -DLLVM_INSTALL_UTILS=ON \
 -DLLVM_LINK_LLVM_DYLIB=ON \
 -DLLVM_PARALLEL_LINK_JOBS=8 \
 -DLLVM_TARGETS_TO_BUILD:STRING="X86;ARM;AArch64" \
 -DLLVM_USE_LINKER=lld \
 -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
 -DMLIR_INCLUDE_DOCS=ON \
 -DPython3_FIND_VIRTUALENV=ONLY \
 -B $LLVM_DIR/build -S $LLVM_DIR/llvm |& tee $LLVM_DIR/cmake.log
cmake --build $CLANGIR_DIR/build --target install --target check-clang-cir --verbose
```

Compiling the specific [MLIR AIE with ClangIR
version](https://github.com/keryell/mlir-aie/tree/clangir):
```bash
cmake \
  -GNinja \
  -DAIE_ENABLE_BINDINGS_PYTHON=ON \
  -DAIE_ENABLE_PYTHON_PASSES=OFF \
  -DAIE_ENABLE_XRT_PYTHON_BINDINGS=ON \
  -DAIE_INCLUDE_INTEGRATION_TESTS=ON \
  -DCLANG_ROOT=$LLVM_DIR/../install-llvm \
  -DCLANGIR_MLIR_FRONTEND=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX=$MLIR_AIE_HOME/../install-mlir-aie \
  -DCMAKE_MODULE_PATH=$MLIR_AIE_HOME/cmake/modulesXilinx \
  -DLLVM_ROOT=$LLVM_DIR/../install-llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_EXTERNAL_LIT=$LLVM_DIR/build/bin/llvm-lit \
  -DLLVM_USE_LINKER=lld \
  -DMLIR_ROOT=$LLVM_DIR/../install-llvm \
  -DPEANO_INSTALL_DIR=$PEANO_DIR/install \
  -DXRT_BIN_DIR=$XILINX_XRT/bin \
  -DXRT_INCLUDE_DIR=$XILINX_XRT/include \
  -DXRT_LIB_DIR=$XILINX_XRT/lib \
  -B $MLIR_AIE_HOME/build -S $MLIR_AIE_HOME |& tee $MLIR_AIE_HOME/cmake.log
LIT_OPTS="--workers=4" cmake --build $MLIR_AIE_HOME/build --target=install --target=check-aie --verbose
```

## Design & compilation flow

`aie++` got some inspiration from [SYCL](https://www.khronos.org/sycl) and some
extensions for [AIE](https://github.com/triSYCL/sycl) to represent everything in
pure modern C++ with classes and lambdas in a type-safe way, having together in
the same program the host code and the device code.


## TODO list

- Make the C++ code also compilable with normal C++ compiler for pure-host
  execution with AIE emulation for debugging and ease of development purpose.
