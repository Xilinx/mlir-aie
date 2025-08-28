# Contains common definitions used across the Makefiles of all tutorials.

# MLIR-AIE install directory. If you have sourced utils/env_setup.sh before
# running make, the following should work to find the AIE install directory.
AIE_RUNTIME_LIB ?= $(shell realpath $(dir $(shell which aie-opt))/../runtime_lib)
AIE_INSTALL ?= $(AIE_RUNTIME_LIB)/aarch64

# VITIS related variables
VITIS_ROOT ?= $(shell realpath $(dir $(shell which vitis))/../)
VITIS_AIETOOLS_DIR ?= ${VITIS_ROOT}/aietools
VITIS_AIE_INCLUDE_DIR ?= ${VITIS_ROOT}/aietools/data/versal_prod/lib
VITIS_AIE2_INCLUDE_DIR ?= ${VITIS_ROOT}/aietools/data/aie_ml/lib

# An aarch64 sysroot is required for cross-compiling the host code. The default
# values of these variables assume that you have built a sysroot by running
# make in platforms/vck190_bare_prod, but you can use other aarch64 sysroots by

MLIR_AIE_SYSROOT ?= ${VITIS_ROOT}/gnu/aarch64/lin/aarch64-linux/aarch64-xilinx-linux/

# The libstdc++ version that is installed in the sysroot given above. This is
# used for include and library paths. If you built the sysroot with Vitis
# 2022.2 and PetaLinux 2022.2, libstdc++ 11.2.0 will be installed.
LIBCXX_VERSION ?= 11.2.0

# The following flags are passed to both AI core and host compilation for
# aiecc.py invocations.
AIECC_FLAGS += --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu

CHESSCC_FLAGS = -f -p me -P ${VITIS_AIE_INCLUDE_DIR} -I ${VITIS_AIETOOLS_DIR}/include
CHESS_FLAGS = -P ${VITIS_AIE_INCLUDE_DIR}

# The following additional flags are only applied for host code. We add the
# necessary search paths for the sysroot so clang++ can find the aarch64
# includes and libraries.
AIECC_HOST_FLAGS += -I$(AIE_INSTALL)/test_lib/include -I${AIE_INSTALL}/xaiengine/include -I${MLIR_AIE_SYSROOT}/usr/include -I${MLIR_AIE_SYSROOT}/usr/include/c++/${LIBCXX_VERSION} -I${MLIR_AIE_SYSROOT}/usr/include/c++/${LIBCXX_VERSION}/aarch64-xilinx-linux -L${MLIR_AIE_SYSROOT}/usr/lib/aarch64-xilinx-linux/${LIBCXX_VERSION} -L$(AIE_INSTALL)/test_lib/lib -ltest_lib -B${MLIR_AIE_SYSROOT}/usr/lib/aarch64-xilinx-linux/${LIBCXX_VERSION}

.PHONY: all clean

all: build/sim

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
build/test: test.cpp aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -j4 $(AIECC_FLAGS) ../$(word 2,$^) $(AIECC_HOST_FLAGS) ../$< -o ${@F}

build/sim : test.cpp aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -j4 --xchesscc --xbridge --aiesim ../$(word 2,$^) -ggdb -I$(AIE_RUNTIME_LIB)/x86_64/test_lib/include -L$(AIE_RUNTIME_LIB)/x86_64/test_lib/lib -ltest_lib ../$<

clean:
	rm -rf build
