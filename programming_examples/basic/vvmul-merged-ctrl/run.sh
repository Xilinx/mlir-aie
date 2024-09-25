python /scratch/aba/micro/mlir-aie/install/bin/aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --no-compile-host --basic-alloc-scheme --generate-ctrl-pkt-overlay --xclbin-name=aie1.xclbin --npu-insts-name=insts1.txt ./aie1.mlir
python /scratch/aba/micro/mlir-aie/install/bin/aiecc.py --no-aiesim --aie-generate-ctrlpkt --aie-generate-npu --no-compile-host --basic-alloc-scheme --generate-ctrl-pkt-overlay --npu-insts-name=insts2.txt ./aie2.mlir
aie-translate -aie-ctrlpkt-to-bin aie2.mlir.prj/ctrlpkt.mlir -o ctrlpkt.txt
aie-opt -aie-ctrl-packet-infer-tiles -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" -aie-ctrl-packet-to-dma aie2.mlir.prj/ctrlpkt.mlir > aie3.mlir
python /scratch/aba/micro/mlir-aie/install/bin/aiecc.py --no-aiesim --aie-only-generate-npu --no-compile-host --generate-ctrl-pkt-overlay --xclbin-name=aie3.xclbin --npu-insts-name=insts3.txt aie3.mlir
clang ./test.cpp -o test.exe -std=c++11 -Wall -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil -lrt -lstdc++ -lboost_program_options -lboost_filesystem
./test.exe