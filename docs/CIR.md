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

Since most of ACDC software stack is MLIR-based, we are using
[ClangIR](https://github.com/llvm/clangir) to generate some MLIR from
C/C++. ClangIR is a fork of Clang generating MLIR CIR dialect during its code
generation and sounds like the most promising approach since it is in the
process of being up-streamed to Clang/LLVM/MLIR and it is backed by several
companies.

This project requires the compiler flow in ClangIR going from MLIR CIR to the
MLIR standard dialects which is unfortunately quite less implemented than the
direct ClangIR CIR → LLVMIR lowering. This explains that most of the efforts in
the `aie++` project have been diverted into the implementation of the CIR→MLIR
flow itself. There are also some fundamental limitations in the MLIR standard
dialects themselves which have to be overcome.

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

A simple program like [`example.cpp`](../test/CIR/aie++/example.cpp):

```c++
#include "aie++.hpp"

int main() {
  aie::device<aie::npu1> d;
  auto t = d.tile<1, 4>();
  auto b = t.buffer<int, 8192>();
  t.program([&] { b[3] = 14; });
  d.tile<2, 3>().program([] {});
  d.run();
}
```
can be compiled with [`aie++-compile.sh`](../utils/aie++-compile.sh):

```bash
PATH=$LLVM_DIR/build/bin:$MLIR_AIE_HOME/build/bin:$PATH $MLIR_AIE_HOME/utils/aie++-compile.sh example.cpp
```
to produce among others an `example.aie.only.cir` file:

```MLIR
  aie.device(npu1) {
    // [...lot of skipped functions...]
    %tile_1_4 = aie.tile(1, 4) {cir.type = !cir.ptr<!ty_aie3A3Atile3C12C_42C_aie3A3Adevice3C3E3E>}
    %buffer_1_4 = aie.buffer(%tile_1_4) {cir.type = !cir.ptr<!ty_aie3A3Abuffer3Cint2C_8192UL3E>} : memref<8192xi32>
    %1 = builtin.unrealized_conversion_cast %buffer_1_4 : memref<8192xi32> to !cir.ptr<!ty_aie3A3Abuffer3Cint2C_8192UL3E>
    %core_1_4 = aie.core(%tile_1_4) {
      cir.scope {
        %3 = cir.const #cir.int<14> : !s32i
        %4 = cir.const #cir.int<3> : !s32i
        %5 = cir.cast(integral, %4 : !s32i), !u64i
        %6 = cir.call @_ZN3aie6bufferIiLm8192EEixEm(%1, %5) : (!cir.ptr<!ty_aie3A3Abuffer3Cint2C_8192UL3E>, !u64i) -> !cir.ptr<!s32i>
        cir.store %3, %6 : !s32i, !cir.ptr<!s32i>
      }
      aie.end
    }
    %tile_2_3 = aie.tile(2, 3) {cir.type = !cir.ptr<!ty_aie3A3Atile3C22C_32C_aie3A3Adevice3C3E3E>}
    %2 = builtin.unrealized_conversion_cast %tile_2_3 : index to !cir.ptr<!ty_aie3A3Atile3C22C_32C_aie3A3Adevice3C3E3E>
    %core_2_3 = aie.core(%tile_2_3) {
      cir.scope {
      }
      aie.end
    }
  } {cir.type = !cir.ptr<!ty_aie3A3Adevice3Caie3A3Anpu12C_aie3A3A28lambda_at_2E2Faie2B2B2Ehpp3A2183A63293E>}
```

and generate other local files like:

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

## C++ to MLIR strategy

Different solutions have been studied

###  Plan A: Polygeist + MLIR-AIR/MLIR-AIE

[Polygeist](https://github.com/llvm/Polygeist)

- C++ front-end from scratch started as PhD work @MIT

- Used by ACDC in the past

- Not being able to parse any real program (`std::`)

- Was lagging by being based on quite old LLVM version

- Not possible to use current MLIR-AIR/MLIR-AIE

- Difficult to modernize the code-base, even if other people are trying the same
  from time to time…


### Plan B: SYCL MLIR + MLIR-AIR/MLIR-AIE

[SYCL MLIR](https://github.com/intel/llvm/tree/sycl-mlir)

- Intel has developed SYCL MLIR branch with single-source C++ parser (host +
  device)

- Polygeist fork by Intel

- Extended and rebased on latest version of LLVM

- Idea for AIE: leverage Intel engineering

- MLIR-AIR/MLIR-AIE + C++ part of SYCL

Problems:

- Only device code goes through Polygeist because not robust enough to handle
  host code

- Host C++ code goes though usual Clang + LLVM → MLIR LLVM → raised to some MLIR
  SYCL & other dialects

- Not generic enough for plain C++ but could be adapted to AIE++

- Intel started deprecating this project and moving to ClangIR at the end of
  2023

- Branch is now stale


### Plan C:  VAST (Trail of bits)

[VAST](https://github.com/trailofbits/vast)

- Security company working on program analysis and instrumentation

- No MLIR std but 2 dialects, HL and LL 

  `-vast-emit-mlir=hl` to generate high-level dialect

  `-vast-emit-mlir=llvm` to generate LLVM MLIR dialect


### Plan D: Use ClangIR project

[ClangIR](https://llvm.github.io/clangir)

- Clang-based C/C++ parser generating MLIR CIR dialect pushed by Meta & Nvidia

- Pragmatic approach: `ASTConsumer` of Clang

  - Reuse Clang C++ semantics analysis

  - Duplicate proven skeleton CodeGen → LLVM IR with CodeGen → MLIR CIR dialect

    - Clever: keep most of the logic as is, because Clang is quite complicated

  - Can lower directly to LLVM IR or to standard MLIR dialects (affine, scf,
    cf…)

- Good traction in the industry: Meta (analyses), Nvidia (OpenACC, OpenMP,
  Flang), Intel (SYCL), Microsoft (HLSL), Google (Polygeist), Trail of Bits
  (VAST), NextSilicon, AMD (AIE++ 😏)…

- In the process of being up-streamed https://discourse.llvm.org/t/rfc-upstreaming-clangir/76587 😃

- Problems

  - Complex rebase-only development process on top of up-stream

    - Painful to stay close to up-stream → lagging behind

  - Not always up-to-date with upstream compared to MLIR-AIR/MLIR-AIE 

    - But able to merge MLIR-AIR/MLIR-AIE LLVM version into an AMD ClangIR
      branch relatively easily

  - WIP with Meta priority on LLVM direct compilation for Android: lowering to
    MLIR std quite in infancy 

- Solutions

  - AMD becomes a ClangIR contributor on CodeGen → MLIR CIR → MLIR std

  - Prioritize work on AIE++ to minimize my ClangIR contribution while giving a
    taste of AIE++


### Implementation of structs

#### The MLIR `tuple` tragedy

- Builtin MLIR type to represent product type

  - class, struct, C++`std::tuple`, Python `collections.namedtuple`…

- Orphan type in core MLIR! 😦

  - No operation

  - No attribute

  - Cannot be in a `memref`

→ No reuse design pattern

→ Pile of anti-patterns

  - Reimplement over again and again similar operations (insert element, extract
    element…)

  - Reimplement again and again similar type in front-end (`!cir.struct`) or
    back-end (`!llvm.struct`, `!spirv.struct`)

  - Replicate the datalayout anti-pattern

- Core feature for C++ and not implemented in ClangIR → MLIR std 😦


#### The MLIR `tuple` strategy

- Experimented various strategies to lower C/C++ struct

- Extension of `tuple` type itself

- Very intrusive on other uses

- Looked at Polygeist hack relying on `memref<!llvm.struct<…>>`

  - `polygeist.memref2pointer`

  - Compute the access with `llvm.getelementptr`

  - `polygeist.pointer2memref`

- Manual address computation + `named_tuple.cast`

- Create new minimal `named_tuple`

For example:

```c++
struct s {
  int a;
  double b;
  char c;
  float d[5];
};
int main() {
  s v;
  v.c = 'z’;
}
```

is lowered to:

```MLIR
 %c122_i8 = arith.constant 122 : i8
 %2 = named_tuple.cast %alloca_0 : memref<!named_tuple.named_tuple<"s", [i32, f64, i8, tensor<5xf32>]>> to memref<40xi8>
 %c16 = arith.constant 16 : index
 %view_3 = memref.view %2[%c16][] : memref<40xi8> to memref<i8>
 memref.store %c122_i8, %view_3[] : memref<i8>
```

## TODO list

- Documentation

- Minimal C++ → MLIR AIE e2e example

- Develop MLIR AIE C++ abstraction header & runtime

- Make the C++ code also compilable with normal C++ compiler for pure-host
  execution with AIE emulation for debugging and ease of development purpose.

- Minimal support of struct is required

- More tests & tutorial from AIR & AIE as examples

- Adapt examples or C++ applications to show how to use the framework incrementally

- Generalize C++ header for different DSL

- Encode in C++ header some transformation recipes to apply on each DSL class

- Create an MLIR library to handle generic lowering of CIR

- Merge PR upstream & integration to aiecc.py

- Development of ClangIR → MLIR standard dialect

- Develop CIR MLIR transformations to lower to target + standard dialect

- Help on ClangIR up-streaming

- Push for high-level-language-support in MLIR standard dialects
