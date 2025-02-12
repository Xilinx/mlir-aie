#! /bin/sh -vx

# Execute a simple flow lowering aie++ program to MLIR AIE kernels

# Execute this script with the right path according to your environment, for
# example with:
# PATH=$LLVM_DIR/build/bin:$MLIR_AIE_HOME/build/bin:$PATH $MLIR_AIE_HOME/utils/aie++-compile.sh example.cpp

source_file_name="$1"
# The local file part without the last extension
prefix="$(basename ${source_file_name%.*})"

# No debug
clang++ -c -std=c++26 -fno-exceptions -Xclang -emit-cir -c "$source_file_name" -o "$prefix".cir
# Same with debug mode
#clang++ -v -c -std=c++26 -fno-exceptions -Xclang -emit-cir -Xclang -clangir-verify-diagnostics -mmlir -debug -mmlir -mlir-print-op-generic -mmlir -mlir-print-stacktrace-on-diagnostic -mmlir -mlir-print-op-on-diagnostic -mmlir -mlir-print-ir-after-failure channel.cpp -o channel.cir
aie-opt --cir-to-aie-prepare "$prefix".cir -o "$prefix".prepare.aie.cir
aie-opt --cir-to-aie "$prefix".prepare.aie.cir -o "$prefix".aie.cir
aie-opt --cir-to-aie-inline-kernel-lambda --mem2reg "$prefix".aie.cir -o "$prefix".aie.lambda.cir
aie-opt --cir-to-aie-decapture-kernel "$prefix".aie.lambda.cir -o "$prefix".aie.decapture.cir
aie-opt --cir-keep-aie-device "$prefix".aie.decapture.cir -o "$prefix".aie.only.cir
aie-opt --cir-to-mlir "$prefix".aie.only.cir -o "$prefix".aie.mlir
