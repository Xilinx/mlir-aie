// REQUIRES: valid_xchess_license
// RUN: aie-opt %s %tosa-to-linalg% | aie-opt %linalg-to-vector-v16% --convert-vector-to-aievec="aie-target=aieml" -lower-affine | aie-translate -aieml=true --aievec-to-cpp -o dut.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. -c %aie_runtime_lib%/AIE2/lut_based_ops.cpp -o lut_based_ops.cpp.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. -c %S/testbench.cc -o testbench.cc.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I%aie_runtime_lib%/AIE2 -I %aietools/include -D__AIEARCH__=20 -D__AIENGINE__ -I. ./work/lut_based_ops.cpp.o ./work/testbench.cc.o dut.cc
// RUN: mkdir -p data
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED
// Cycle count: 807

func.func @dut(%arg0: tensor<1024xbf16>) -> (tensor<1024xbf16>) {
  %0 = "tosa.tanh"(%arg0) : (tensor<1024xbf16>) -> tensor<1024xbf16>
  return %0 : tensor<1024xbf16>
}
