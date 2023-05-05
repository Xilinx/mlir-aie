##===- utils/common.sh - Common functions for build scripts --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

# sanity_checks()
# Try to catch potential missing dependencies early to alert the user to it.
# Argument 1: N args on command line for calling function
# Argument 2: CMAKEMODULES_DIR
# Argument 3: LLVM_BUILD_DIR
sanity_checks() {
	if [ ! -f "${2}/modulesXilinx/FindVitis.cmake" ]; then
		echo "cmake/modulesXilinx not found. Make sure you clone all "\
		     "of the MLIR repository, including submodules, by "\
		     "running "\
		     "\`git submodule update --recursive\`."
		exit 1
	fi

	if [ ! -d "${3}" ]; then
		echo "llvm/build directory not found. Make sure you clone and "\
		     "build LLVM first by running \`./utils/clone-llvm.sh\`, "\
		     "and \`./utils/build-llvm-local.sh\`, respectively, first."
		if [ "${1}" -lt 1 ]; then
			echo "You may also specify a path to a LLVM build as "\
			     "the first argument to this script."
		fi
		exit 1
	fi

	if [ ! -f "$(which xchesscc)" ]; then
		echo "xchesscc not found. Make sure you run"\
		     "\`source <path to Vitis 2022.2>/settings64.sh\` first."
		exit 1
	fi
}
