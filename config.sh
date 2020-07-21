CLANG_VER=8
cmake -GNinja \
		-DLLVM_DIR=$1/../peano/lib/cmake/llvm \
		-DMLIR_DIR=$1/../peano/lib/cmake/mlir \
		-DCMAKE_C_COMPILER=clang-${CLANG_VER} \
		-DCMAKE_CXX_COMPILER=clang++-${CLANG_VER} \
		-DCMAKE_BUILD_TYPE=Debug \
		-B$1 -H$2

		# -DCMAKE_C_COMPILER=/wrk/hdstaff/stephenn/llvm-project/build_X86/bin/clang \
		# -DCMAKE_CXX_COMPILER=/wrk/hdstaff/stephenn/llvm-project/build_X86/bin/clang++ \
		# -DLLVM_USE_LINKER=lld \


#cmake -GNinja -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="AIE;RISCV;ARC" -DLLVM_TARGETS_TO_BUILD="" -DCMAKE_C_COMPILER=/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/bin/gcc  -DCMAKE_CXX_COMPILER=/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/bin/g++ -DLLVM_TOOL_MLIR_BUILD=OFF ../llvm

