#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# Copyright (C) 2021, Xilinx, Inc. All rights reserved.
# 

VIVADO=$(XILINX_VIVADO)/bin/vivado

all: hw

hw:
		$(VIVADO) -mode batch -notrace -source $(PLATFORM)_xsa.tcl

clean:
		$(RM) -r vivado.* $(PLATFORM) $(PLATFORM).xsa vck190_bare_proj *.backup.* NOC_Power.xpe .Xil
