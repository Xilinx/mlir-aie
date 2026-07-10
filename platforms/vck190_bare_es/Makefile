#
# Copyright (C) 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
PLATFORM ?= xilinx_vck190_bare
CWD=$(shell pwd)
XSA_DIR ?= $(CWD)/vivado

.phony: all

all: $(PLATFORM)

$(PLATFORM): xsa petalinux_build petalinux_sysroot
	@echo 'Platform build complete'

xsa:
	$(MAKE) -C vivado PLATFORM=$(PLATFORM)

xsa_clean:
	$(MAKE) -C vivado clean PLATFORM=$(PLATFORM)

petalinux_build:
	$(MAKE) -C petalinux all XSA_DIR=$(XSA_DIR) PLATFORM=$(PLATFORM)

petalinux_sysroot:
	$(MAKE) -C petalinux sysroot

petalinux_clean:
	$(MAKE) -C petalinux ultraclean PLATFORM=$(PLATFORM)

petalinux_bsp:
	$(MAKE) -C petalinux bsp

cleanall: petalinux_clean xsa_clean
	${RM} -r platform_repo
	
