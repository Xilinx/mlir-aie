# stack_sizes_makefile_kernel.mk -*- Makefile -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Support file for stack_sizes_makefile_kernel.test. Pulls in the real
# programming_examples/makefile-common (not a hand-copied mirror of its
# flags) so this test breaks if PEANOWRAP2P_FLAGS ever drops
# -mllvm -stack-size-section, the same way every example Makefile that
# includes it would.

include $(dir $(lastword $(MAKEFILE_LIST)))../../programming_examples/makefile-common

.PHONY: kernel
kernel:
	clang $(PEANOWRAP2P_FLAGS) -c $(SRC) -o $(OUT)
