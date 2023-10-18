// REQUIRES: valid_xchess_license
// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg))" -o linalg.mlir
// RUN: mlir-opt linalg.mlir --linalg-fuse-elementwise-ops --eliminate-empty-tensors --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs-from-loops allow-unknown-ops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --drop-equivalent-buffer-results --buffer-results-to-out-params --buffer-deallocation --canonicalize --cse --convert-linalg-to-affine-loops --affine-super-vectorize="virtual-vector-size=16" -o affine.mlir 
// RUN: aie-opt affine.mlir --convert-vector-to-aievec="aie-target=aieml" -lower-affine -o aievec.mlir
// RUN: aie-translate aievec.mlir -aieml=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. %S/testbench.cc dut.cc
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out"
// RUN: clang++ %S/dut_ref.cc -o dut_ref
// RUN: ./dut_ref >& check.stdout
// RUN: FileCheck --input-file=./check.stdout %s
// CHECK: TEST PASSED
// Cycle count: 1134

func.func @dut(%arg0: tensor<1024xbf16>) -> (tensor<1024xbf16>) {
  %0 = tosa.erf %arg0 : (tensor<1024xbf16>) -> tensor<1024xbf16>
  return %0 : tensor<1024xbf16>
}
