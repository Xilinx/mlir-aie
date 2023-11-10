include ../../../test/ipu-xrt/makefile-common

VPATH := ../vision_kernels

all: build/final_tiny.xclbin

all_720: build/final_720.xclbin

all_1080: build/final_1080.xclbin

build/insts%.txt: aie2_lineBased_8b%.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=${@F} $(<:%=../%)

build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc ${CHESSCC2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

build/combined_gray2rgba_addWeighted.a: build/gray2rgba.cc.o build/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build/final_%.xclbin: aie2_lineBased_8b_%.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host --xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

# build/test.exe: test.cpp
# 	mkdir -p ${@D}
# 	$(CXX) $(LD_FLAGS) $(OpenCV_LIBS) ../utils/OpenCVUtils.cpp ../utils/xrtUtils.cpp $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

clean:
	rm -rf build
