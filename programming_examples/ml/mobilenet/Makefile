#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}
# Copyright (C) 2024, Advanced Micro Devices, Inc.
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

VPATH := ${srcdir}/../../../aie_kernels/aie2

MLIR_FILE=aie_mobilenet
TEST_FILE=test_mobilenet


PYTHONPATH := $(PYTHONPATH):$(shell cd .. && pwd)
export PYTHONPATH

trace_size = 32768
# DUMMY = _dummy
DUMMY =

## ******** BELOW DESIGNS ARE FUNCTIONALLY CORRECT ********
## DESIGN_RUN can be "v1","v4","v5","v6", "v7" ,"v8","v9","v9.1","v10", or any other desired value to not fuse any bns
## v1: DESIGN_RUN ?= bn0+bn1 #pipeline with A+B+C. Extreme fusion of Bn0 and Bn1 on a single core.
## v4: DESIGN_RUN ?= bn8+bn9 #pipeline with A+B+C. Extreme fusion of Bn8 and Bn9 on a single core.
## v5: DESIGN_RUN ?= bn8+bn9+FC1_pad #pipeline with A+B+C+FC1. Extreme fusion of Bn8 and Bn9 on a single core.
## v6: DESIGN_RUN ?= bn8+bn9+FC1_pad+FC2_pad #end-to-end pipeline with A+B+C+FC1 and FC2 layers. Extreme fusion of Bn8 and Bn9 on a single core.
## v7: DESIGN_RUN ?= bn8+bn9+FC1_pad+FC2_pad_L1static #end-to-end pipeline with A+B+C+FC1 and FC2 layers with L1 weight initialization. Extreme fusion of Bn8 and Bn9 on a single core.
## v8: DESIGN_RUN ?= bn0+bn1_bn4+bn5_bn8+bn9+FC1_pad+FC2_pad_L1static #end-to-end pipeline with A+B+C+FC1 and FC2 layers with L1 weight initialization. Extreme fusion of BN0+BN1, BN4+BN5, and BN8+BN9 on a single core.
## v9: DESIGN_RUN ?= bn4+bn5_bn8+bn9_bn12_2+bn12_3_FC1_pad+FC2_pad_L1static #end-to-end pipeline with A+B+C+FC1 and FC2 layers with L1 weight initialization. Extreme fusion of BN4+BN5, BN8+BN9, and BN12_2+BN12_3 on a single core.
## v9.1: DESIGN_RUN ?= bn4+bn5_bn8+bn9_bn12_2+bn12_3_FC1_pad+FC2_pad_L1_L2static_FC #end-to-end pipeline with A+B+C+FC1 and FC2 layers with L1 weight initialization and L2 weights intialization for FC layers. Extreme fusion of BN4+BN5, BN8+BN9, and BN12_2+BN12_3 on a single core.
## v9.2: DESIGN_RUN ?= v9.1 + L2 weights intialization for L1+FC layers. 

## ******** BELOW DESIGNS ARE UNDER DEBUG ********
## v9.3: DESIGN_RUN ?= v9.1 + L2 weights intialization for bn13_1+FC layers. 
## v9.4: DESIGN_RUN ?= v9.1 + L2 weights intialization for bn13_3+FC layers. 
## v10: DESIGN_RUN ?= bn4+bn5_bn8+bn9_bn12_2+bn12_3_FC1_pad+FC2_pad_L1_L2static #end-to-end pipeline with A+B+C+FC1 and FC2 layers with L1 weight initialization. Extreme fusion of BN4+BN5, BN8+BN9, and BN12_2+BN12_3 on a single core.
## DESIGN_RUN ?= <anything else> #pipeline with A+B+C, without extreme fusion.

#DESIGN_RUN ?= v9.2
DESIGN_RUN ?= final


new= True
newFlag=_NEW

# Conditional logic based on the value of DESIGN_RUN
ifeq ($(DESIGN_RUN), v1)
	MLIR_FILE=aie_mobilenet_a+b_placement
	TEST_FILE=test_mobilenet_a+b
  OBJ = build/init_conv2dk3.o \
		build/fused_bn0_bn1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn8_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn9_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o 
else ifeq ($(DESIGN_RUN), v4)
  MLIR_FILE = aie2_mobilenet_init+a_bn8_bn9_fused+b_placement+c+postL2_split
  TEST_FILE = test_mobilenet_init+a_bn8_bn9_fused+b+c+postL2
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui8.o
else ifeq ($(DESIGN_RUN), v5)
  MLIR_FILE = aie2_mobilenet_init+a_bn8_bn9_fused+b_placement+c+post+FC1_pad
  TEST_FILE = test_mobilenet_init+a_bn8_bn9_fused+b+c+post+FC1_pad
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o

else ifeq ($(DESIGN_RUN), v6)
  MLIR_FILE = aie2_mobilenet_init+a_bn8_bn9_fused+b_placement+c+post+FC1_pad+FC2_pad
  TEST_FILE = test_mobilenet_init+a_bn8_bn9_fused+b+c+post+FC1_pad+FC2_pad
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o

# else ifeq ($(DESIGN_RUN), bn8+bn9+FC1_pad+FC2_pad_L1static)
else ifeq ($(DESIGN_RUN), v7)
  MLIR_FILE = aie2_mobilenet_init+a_bn8_bn9_fused+b_placement+c+post+FC1_pad+FC2_pad_L1static
  TEST_FILE = test_mobilenet_init+a_bn8_bn9_fused+b+c+post+FC1_pad+FC2_pad_L1static
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o

#else ifeq ($(DESIGN_RUN),bn0+bn1_bn8+bn9+FC1_pad+FC2_pad_L1static)
else ifeq ($(DESIGN_RUN),v7b)
  MLIR_FILE = aie2_mobilenet_init+a_bn0_bn1+bn8_bn9_fused+b_placement+c+post+FC1_pad+FC2_pad_L1static
  TEST_FILE = test_mobilenet_init+a_bn0+bn1_bn8+bn9_fused+b+c+post+FC1_pad+FC2_pad_L1static
  OBJ = build/init_conv2dk3.o \
		build/fused_bn0_bn1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o
#else ifeq ($(DESIGN_RUN), bn0+bn1_bn4+bn5_bn8+bn9+FC1_pad+FC2_pad_L1static)
else ifeq ($(DESIGN_RUN), v8)
  MLIR_FILE = aie2_mobilenet_init+a_bn0_bn1+bn4_bn5+bn8_bn9_fused+b_placement+c+post+FC1_pad+FC2_pad_L1static
  TEST_FILE = test_mobilenet_init+a_bn0_bn1+bn4_bn5+bn8_bn9_fused+b+c+post+FC1_pad+FC2_pad_L1static
  OBJ = build/init_conv2dk3.o \
		build/fused_bn0_bn1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/combined_bn_4_5.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o
# else ifeq ($(DESIGN_RUN), bn4+bn5_bn8+bn9_bn12_2+bn12_3_FC1_pad+FC2_pad_L1static)
else ifeq ($(DESIGN_RUN), v9)
  MLIR_FILE = aie2_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1static
  TEST_FILE = test_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1static
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/combined_bn_4_5.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/fused_bn12_layer2_3.a \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o

else ifeq ($(DESIGN_RUN), v9.1)
  MLIR_FILE = aie2_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1_L2static_FC
  TEST_FILE = test_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1static_FC
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/combined_bn_4_5.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/fused_bn12_layer2_3.a \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o


else ifeq ($(DESIGN_RUN), v9.2)
  MLIR_FILE = aie2_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1_L2static_L1+FC
  TEST_FILE = test_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1static_L1+FC
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/combined_bn_4_5.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/fused_bn12_layer2_3.a \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o

else ifeq ($(DESIGN_RUN), final)
  MLIR_FILE = aie2_mobilenet
  TEST_FILE = test_mobilenet
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/combined_bn_4_5.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/fused_bn12_layer2_3.a \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o


else ifeq ($(DESIGN_RUN), v9.3)
  MLIR_FILE = DEBUGGING_bn13_wts_layer1
  TEST_FILE = test_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1static_bn13_1+FC
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/combined_bn_4_5.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/fused_bn12_layer2_3.a \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o


else ifeq ($(DESIGN_RUN), v9.4)
  MLIR_FILE = DEBUGGING_bn13_wts_layer3
  TEST_FILE = test_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1static_bn13_1+FC
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/combined_bn_4_5.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/fused_bn12_layer2_3.a \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o
else ifeq ($(DESIGN_RUN), v10)
# else ifeq ($(DESIGN_RUN), bn4+bn5_bn8+bn9_bn12_2+bn12_3_FC1_pad+FC2_pad_L1_L2static)
  MLIR_FILE = aie2_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1_L2static
  TEST_FILE = test_mobilenet_init+a_bn4_bn5+bn8_bn9_fused+b_bn12_2+bn12_3_c+post+FC1_pad+FC2_pad_L1_L2static
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/combined_bn_4_5.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/fused_bn12_layer2_3.a \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui16_pad.o
else ifeq ($(DESIGN_RUN), bn8+bn9_tracing)
  MLIR_FILE = aie2_mobilenet_init+a_bn8_bn9_fused+b_placement+c+postL2_split_tracing
  TEST_FILE = test_mobilenet_init+a_bn8_bn9_fused+b+c+postL2_tracing
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/combined_bn_8_9.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui8.o

else
  # MLIR_FILE = aie2_mobilenet_init+a_unfused+b_placement
	# TEST_FILE=test_mobilenet_init+a+b
	# MLIR_FILE = aie2_mobilenet_init+a_unfused+b_placement+c
	# TEST_FILE =test_mobilenet_init+a+b+c
	# MLIR_FILE = aie2_mobilenet_init+a_unfused+b_placement+c+post_TEST
	# TEST_FILE =test_mobilenet_init+a+b+c+post_TEST
	
	# MLIR_FILE = aie2_mobilenet_init+a_unfused+b_placement+c+post
	# TEST_FILE =test_mobilenet_init+a+b+c+post
	
	MLIR_FILE = aie2_mobilenet_init+a_unfused+b_placement+c+postL2
	TEST_FILE =test_mobilenet_init+a+b+c+postL2
	
	
  OBJ = build/init_conv2dk3.o \
		build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a \
		build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a \
		build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn8_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a \
		build/bn9_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a \
		build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
		build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
		build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
		build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o\
		build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
		build/post_conv2dk1_relu_xy_pool_i8_ui8.o \
		build/post_L2_conv2dk1_relu_ui16_ui8.o
endif

all: build/final_mobilenet.xclbin run_py_mobilenet

trace: build/trace_mobilenet.xclbin trace_py_mobilenet

build/${MLIR_FILE}.mlir: ${srcdir}/${MLIR_FILE}.py 
	mkdir -p ${@D}
	python3 $< > $@

# build/${MLIR_FILE}_trace.mlir: ${srcdir}/${MLIR_FILE}_trace.py 
build/${MLIR_FILE}_trace.mlir: ${srcdir}/${MLIR_FILE}.py 
	mkdir -p ${@D}
	python3 $< -t ${trace_size} > $@

# init
build/init_conv2dk3.o: bottleneck/bn_conv2dk3${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -c $< -o ${@F}

# bn0+bn1
build/bn0_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN0 -DSCALAR -DSTRIDE1 -c $< -o ${@F}

build/bn0_conv2dk1_skipui8.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN0 -DSCALAR -DUNSIGNED_SKIP -c $< -o ${@F}

build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a: build/bn0_conv2dk3_dw_stride1.o build/bn0_conv2dk1_skipui8.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/bn0_combined_conv2dk3dwstride1_conv2dk1skipui8.a: build/bn0_conv2dk3_dw_stride1.o build/bn0_conv2dk1_skipui8.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/bn1_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN1 -DINT8_ACT -c $< -o ${@F}

build/bn1_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN1 -DSCALAR -DSTRIDE2 -c $< -o ${@F}

build/bn1_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN1 -c $< -o ${@F}

build/bn1_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a: build/bn1_conv2dk1_fused_relu.o build/bn1_conv2dk3_dw_stride2.o build/bn1_conv2dk1_i8.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/fused_bn0_bn1.a: build/bn0_conv2dk3_dw_stride1.o build/bn0_conv2dk1_skipui8.o build/bn1_conv2dk1_fused_relu.o build/bn1_conv2dk3_dw_stride2.o build/bn1_conv2dk1_i8.o 
	mkdir -p ${@D}
	ar rvs $@ $^ $(word 2,$^) $(word 3,$^) $(word 4,$^) $(word 5,$^)

# bn2
build/bn2_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN2 -DINT8_ACT -c $< -o ${@F}

build/bn2_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN2 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
build/bn2_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN2 -DSCALAR -c $< -o ${@F}

build/bn2_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a: build/bn2_conv2dk1_fused_relu.o build/bn2_conv2dk3_dw_stride1.o build/bn2_conv2dk1_skip.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

# bn3
build/bn3_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN3 -DINT8_ACT -c $< -o ${@F}
build/bn3_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN3 -DSCALAR -DSTRIDE2 -c $< -o ${@F}
build/bn3_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN3 -c $< -o ${@F}

build/bn3_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a: build/bn3_conv2dk1_fused_relu.o build/bn3_conv2dk3_dw_stride2.o build/bn3_conv2dk1_i8.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

# bn4
build/bn4_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN4 -DINT8_ACT -c $< -o ${@F}

build/bn4_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN4 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
build/bn4_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN4 -DSCALAR -c $< -o ${@F}

build/bn4_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a: build/bn4_conv2dk1_fused_relu.o build/bn4_conv2dk3_dw_stride1.o build/bn4_conv2dk1_skip.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

# bn5
build/bn5_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN5 -DINT8_ACT -c $< -o ${@F}

build/bn5_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN5 -DSCALAR -DSTRIDE1 -c $< -o ${@F}

build/bn5_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN5 -DSCALAR -c $< -o ${@F}

build/bn5_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN5 -c $< -o ${@F}

build/bn5_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a: build/bn5_conv2dk1_fused_relu.o build/bn5_conv2dk3_dw_stride1.o build/bn5_conv2dk1_i8.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/combined_bn_4_5.a: build/bn4_conv2dk1_fused_relu.o build/bn4_conv2dk3_dw_stride1.o build/bn4_conv2dk1_skip.o build/bn5_conv2dk1_fused_relu.o build/bn5_conv2dk3_dw_stride1.o build/bn5_conv2dk1_skip.o
	mkdir -p ${@D}
	ar rvs $@ $^

# bn6
build/bn6_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN6 -DINT8_ACT -c $< -o ${@F}
build/bn6_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN6 -DSCALAR -DSTRIDE2 -c $< -o ${@F}
build/bn6_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN6 -c $< -o ${@F}

build/bn6_combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a: build/bn6_conv2dk1_fused_relu.o build/bn6_conv2dk3_dw_stride2.o build/bn6_conv2dk1_i8.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)
# bn7
build/bn7_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN7 -DINT8_ACT -c $< -o ${@F}

build/bn7_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN7 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
build/bn7_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN7 -DSCALAR -c $< -o ${@F}

build/bn7_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a: build/bn7_conv2dk1_fused_relu.o build/bn7_conv2dk3_dw_stride1.o build/bn7_conv2dk1_skip.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

# bn8
build/bn8_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN8 -DINT8_ACT -c $< -o ${@F}

build/bn8_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN8 -DSCALAR -DSTRIDE1 -c $< -o ${@F}

build/bn8_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN8 -DSCALAR -c $< -o ${@F}

build/bn8_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN8 -c $< -o ${@F}

build/bn8_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a: build/bn8_conv2dk1_fused_relu.o build/bn8_conv2dk3_dw_stride1.o build/bn8_conv2dk1_i8.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

# bn9
build/bn9_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN9 -DINT8_ACT -c $< -o ${@F}

build/bn9_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN9 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
build/bn9_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN9 -DSCALAR -c $< -o ${@F}

build/bn9_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a: build/bn9_conv2dk1_fused_relu.o build/bn9_conv2dk3_dw_stride1.o build/bn9_conv2dk1_skip.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)
# ****************************************************************************************

build/combined_bn_8_9.a: build/bn8_conv2dk1_fused_relu.o build/bn8_conv2dk3_dw_stride1.o build/bn8_conv2dk1_skip.o build/bn9_conv2dk1_fused_relu.o build/bn9_conv2dk3_dw_stride1.o build/bn9_conv2dk1_skip.o
	mkdir -p ${@D}
	ar rvs $@ $^

# bn10
build/bn10_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN10 -DINT8_ACT -c $< -o ${@F}

build/bn10_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN10 -DSTRIDE1  -c $< -o ${@F}

build/bn10_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN10 -DSCALAR -c $< -o ${@F}
# bn11
build/bn11_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN11 -DINT8_ACT -c $< -o ${@F}

build/bn11_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN11 -DSTRIDE1 -c $< -o ${@F}

build/bn11_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN11 -DSCALAR -c $< -o ${@F}
# bn12
build/bn12_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN12 -DINT8_ACT -c $< -o ${@F}

build/bn12_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DSTRIDE2 -DBN12 -c $< -o ${@F}

build/bn12_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN12 -DSCALAR -c $< -o ${@F}

build/fused_bn12_layer2_3.a: build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o
	mkdir -p ${@D}
	ar rvs $@ $^ $(word 2,$^) $(word 3,$^) 
# ************************************** bottlneck 13 **************************************

build/bn13_1_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	echo $(new) World with ${newFlag}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN13_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn13_1_conv2dk1_get.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	echo $(new) World with ${newFlag}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN13_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn13_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	echo $(new) World with ${newFlag}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN13 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}

build/bn13_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag} -c $< -o ${@F}


build/bn13_conv2dk1_skip_get.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

# ************************************** bottlneck 14 **************************************

build/bn14_1_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN14_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn14_1_conv2dk1_get.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN14_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn14_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN14 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}


build/bn14_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn14_conv2dk1_skip_get.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

#build/${mlirFileName}.mlir: ${srcdir}/${pythonSource}.py
#	mkdir -p ${@D}
#	python3 $< > $@

# ***********************************************POST*****************************************
build/post_conv2dk1_relu_xy_pool_i8_ui8.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DCONV_XYPOOL_FUSED_LARGE -DINT8_ACT -c $< -o ${@F}

build/post_L2_conv2dk1_relu_ui16_ui8.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DPOSTL2 -DUINT16_ACT -c $< -o ${@F}

build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DCONV_XYPOOL_FUSED_LARGE_PADDED -DINT8_ACT -c $< -o ${@F}

build/post_L2_conv2dk1_relu_ui16_ui16_pad.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DPOSTL2_PAD -DUINT16_ACT -c $< -o ${@F}

insts.bin: build/${MLIR_FILE}.mlir
	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

build/final_mobilenet.xclbin: build/${MLIR_FILE}.mlir $(OBJ)
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin ${<F}

build/trace_mobilenet.xclbin: build/${MLIR_FILE}_trace.mlir $(OBJ)
	mkdir -p ${@D}
	cd ${@D} && && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin ${<F}

run_py_mobilenet: build/final_mobilenet.xclbin build/${MLIR_FILE}.mlir
	${powershell} python3 ${srcdir}/${TEST_FILE}.py -x build/final_mobilenet.xclbin -i build/insts.bin -k MLIR_AIE

trace_py_mobilenet: build/trace_mobilenet.xclbin build/${MLIR_FILE}_trace.mlir
	${powershell} python3 ${srcdir}/${TEST_FILE}.py -x build/trace_mobilenet.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --filename trace.txt --mlir build/${MLIR_FILE}_trace.mlir --colshift 1 > trace_mbv4_full.json

# clean:
# 	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
# 		chess* *.o insts.bin \
# 		*.log aie_partition.json *.bin BOOT.BIN _x test.exe
clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim \
		build/insts.bin \
		build/*.log aie_partition.json build/*.bin build/BOOT.BIN _x build/test.exe
