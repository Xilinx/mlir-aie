#
#Copyright 2020 Xilinx, Inc.
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#

SYSROOT ?= ../petalinux/build/petalinux/images/linux/sdk/sysroots/cortexa72-cortexa53-xilinx-linux
IMAGE    = ../petalinux/build/petalinux/images/linux/Image
ROOTFS   = ../petalinux/build/petalinux/images/linux/rootfs.ext4


#------------------------------------------------------------------------------


ROOT_DIR=$(realpath $(dir $(lastword $(MAKEFILE_LIST))))


.PHONY: all pfm prep_sd_dir prep_sw_comp prep_sysroot ultraclean clean

# Since making the platform doesn't require sysroot, sometimes we can skip running prep_sysroot by target main
#all: prep_sd_dir prep_sw_comp pfm prep_sysroot
#main: prep_sd_dir prep_sw_comp pfm 

# Target: sw_comp
# Copy PetaLinux genreated software components to platform preparation directory
prep_sd_dir:
	mkdir -p sd_dir && cp ../petalinux/build/petalinux/images/linux/boot.scr ./sd_dir/
#	mkdir -p sd_dir && cp ../step2_petalinux/build/petalinux/images/linux/boot.scr ./sd_dir/

# Target: prep_sw_comp
# Copy rootfs and kernel image to sw_comp directory
prep_sw_comp:
	mkdir -p sw_comp && \
	cp ../petalinux/build/petalinux/images/linux/rootfs.ext4 ./sw_comp && \
	cp ../petalinux/build/petalinux/images/linux/Image ./sw_comp
#	cp ../step2_petalinux/build/petalinux/images/linux/rootfs.ext4 ./sw_comp && \
#	cp ../step2_petalinux/build/petalinux/images/linux/Image ./sw_comp

# Target: prep_sysroot
# Install sysroot to sw_comp directory
prep_sysroot: ../petalinux/build/petalinux/images/linux/sdk.sh
	rm -rf ./sysroot && ../petalinux/build/petalinux/images/linux/sdk.sh -d ./sw_comp -y
#prep_sysroot: ../step2_petalinux/build/petalinux/images/linux/sdk.sh
#	rm -rf ./sysroot && ../step2_petalinux/build/petalinux/images/linux/sdk.sh -d ./sw_comp -y

# Target: pfm
# Generate the Vitis platform with XSCT
# Input: vck190_custom.xsa in ../step1_vivado/build/vivado directory
pfm:
	xsct xsct_create_pfm.tcl xilinx_vck190_prod_bare ../vivado && \
#	xsct xsct_create_pfm.tcl vck190_prod_bare_mb ../vivado/vck190_prod_bare_mb_proj && \
	echo "Platform created:" && \
	find . -name "*.xpfm" && \
	cp ../petalinux/build/petalinux/images/linux/system.dtb ./platform_repo/xilinx_vck190_prod_bare/export/xilinx_vck190_prod_bare/sw/. && \
	cp ../petalinux/build/petalinux/images/linux/bl31.elf ./platform_repo/xilinx_vck190_prod_bare/export/xilinx_vck190_prod_bare/sw/. && \
	cp ../petalinux/build/petalinux/images/linux/u-boot.elf ./platform_repo/xilinx_vck190_prod_bare/export/xilinx_vck190_prod_bare/sw/. 
#	xsct xsct_create_pfm.tcl vck190_custom ../step0_bootable_design/build/vivado && \
#	echo "Platform created:" && \
#	find . -name "*.xpfm"

#clean:
#	$(RM) -r vitis_* xrc.log build platform_repo ./sd_dir/boot.scr rootfs

#ultraclean:
#	$(RM) -r vitis_* xrc.log build platform_repo ./sd_dir/boot.scr rootfs sw_comp



#------------------------------------------------------------------------------


#TARGET   = hw_emu
TARGET   = hw
MODE	 = linux
PLATFORM = ./platform_repo/xilinx_vck190_prod_bare/export/xilinx_vck190_prod_bare/xilinx_vck190_prod_bare.xpfm

XCLBIN   = vck190_aie_base_graph_${TARGET}.xclbin
XSA	 = vck190_aie_base_graph_${TARGET}.xsa
XPFM	 = vck190_xpfm
HOST_EXE = host.exe
SDKTARGETSYSROOT ?= ${SYSROOT}

GRAPH    = aie/graph.cpp
LIBADF  = libadf.a
AIE_CMPL_CMD = aiecompiler -platform=${PLATFORM} -include="./aie" -workdir=./Work ${GRAPH} 
AIE_SIM_CMD = aiesimulator --pkg-dir=./Work --dump-vcd foo
EMU_CMD = ./launch_hw_emu.sh

##########################################################################################################################################################
### DO NOT MODIFY BELOW THIS LINE UNLESS NECESSARY
################################################################################################################################################


VCC      = v++
VPP_SPEC =system.cfg
VPP_FLAGS=--save-temps --verbose --config ${VPP_SPEC}  
LDCLFLAGS=

.PHONY: clean

###
# Guarding Checks. Do not modify.
###
check_defined = \
	$(strip $(foreach 1,$1, \
		$(call __check_defined,$1,$(strip $(value 2)))))

__check_defined = \
	$(if $(value $1),, \
		$(error Undefined $1$(if $2, ($2))))

guard-ROOTFS:
	$(call check_defined, ROOTFS, Set to: xilinx-versal-common-v2021.1/rootfs.ext4)

guard-IMAGE:
	$(call check_defined, IMAGE, Set to: xilinx-versal-common-v2021.1/Image)

guard-CXX:
	$(call check_defined, CXX, Run: xilinx-versal-common-v2021.1/environment-setup-cortexa72-cortexa53-xilinx-linux)

guard-SDKTARGETSYSROOT:
	$(call check_defined, SDKTARGETSYSROOT, Run: xilinx-versal-common-v2021.1/environment-setup-cortexa72-cortexa53-xilinx-linux)

###

all: prep_sd_dir prep_sw_comp pfm prep_sysroot ${XCLBIN} ${HOST_EXE} package
#all: ${XCLBIN} ${HOST_EXE} package
#run: all run_hw_emu
sd_card: all

aie: ${LIBADF}
${LIBADF}: aie/*
	${AIE_CMPL_CMD}

aiesim: ${LIBADF}
	${AIE_SIM_CMD}

xclbin: ${XCLBIN}


${XCLBIN}: ${LIBADF} ${VPP_SPEC} 
	${VCC} -g -l --platform ${PLATFORM} ${LIBADF} -t ${TARGET} ${VPP_FLAGS} -o $@

host: guard-CXX guard-SDKTARGETSYSROOT ${HOST_EXE}
${HOST_EXE}: ${GRAPH} ./Work/ps/c_rts/aie_control_xrt.cpp
	$(MAKE) -C sw/

package: guard-ROOTFS guard-IMAGE package_${TARGET}
package_${TARGET}: ${LIBADF} ${XCLBIN} ${HOST_EXE} 
	${VCC} -p -t ${TARGET} -f ${PLATFORM} \
		--package.rootfs ${ROOTFS} \
		--package.kernel_image ${IMAGE} \
		--package.boot_mode=sd \
		--package.image_format=ext4 \
		--package.defer_aie_run \
		--package.sd_file ${HOST_EXE} \
		${XCLBIN} ${LIBADF}

#		--package.sd_file embedded_exec.sh \

run_hw_emu: launch_hw_emu.sh
launch_hw_emu.sh: package_hw_emu
	$(EMU_CMD)

#clean:
#	$(RM) -r vitis_* xrc.log build platform_repo ./sd_dir/boot.scr rootfs

clean:
	$(RM) -rf vitis_* xrc.log ./sd_dir/boot.scr rootfs
	$(RM) -rf _x v++_* ${XOS} ${OS} ${LIBADF} *.o.* *.o *.xpe *.xo.* \
	       vck190_aie_base*.xclbin* *.xsa *.log *.jou xnwOut Work Map_Report.csv \
	       ilpProblem* sol.db drivers .Xil
	$(RM) -rf AIECompiler.log xcd.log BOOT.BIN a.xclbin *.bin sd_card.img *summary *bif
	$(RM) -rf platform_repo build sd_card sd_dir sw_comp
	$(MAKE) -C sw clean


ultraclean: clean
	$(RM) -rf vitis_* xrc.log build platform_repo ./sd_dir/boot.scr rootfs sw_comp
	$(RM) -rf sd_dir Work

