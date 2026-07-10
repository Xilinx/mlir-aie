#
# Copyright (C) 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 

VIVADO=$(XILINX_VIVADO)/bin/vivado

all: hw

hw:
	$(VIVADO) -mode batch -notrace -source $(PLATFORM)_xsa.tcl

clean:
	$(RM) -r vivado.* $(PLATFORM) $(PLATFORM).xsa vck190_prod_bare_proj *.backup.* NOC_Power.xpe .Xil
