# stack_sizes_makefile_kernel.mk -*- Makefile -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Support file for stack_sizes_makefile_kernel.test. Includes
# programming_examples/makefile-common itself, so this test compiles with the
# PEANOWRAP2P_FLAGS that every example Makefile uses, and fails when those
# flags lose -fstack-size-section.

include $(dir $(lastword $(MAKEFILE_LIST)))../../programming_examples/makefile-common

.PHONY: kernel
kernel:
	clang $(PEANOWRAP2P_FLAGS) -c $(SRC) -o $(OUT)
