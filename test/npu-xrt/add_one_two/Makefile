include ../makefile-common

targetname = addOneObjfifo

all: build2/final.xclbin addOneObjfifo.exe

build1/first.xclbin: aie1.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py \
				--xclbin-kernel-name=ADDONE --xclbin-kernel-id=0x901 --xclbin-instance-name=ADDONEINST\
				--aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt ../aie1.mlir

build2/final.xclbin: build1/first.xclbin aie2.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py \
				--xclbin-kernel-name=ADDTWO --xclbin-kernel-id=0x902 --xclbin-instance-name=ADDTWOINST \
				--aie-generate-cdo --no-compile-host \
				--xclbin-input-name=../build1/first.xclbin --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt ../aie2.mlir

${targetname}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build2/final.xclbin
	${powershell} ./$< -v 1 -x build2/final.xclbin -i build2/insts.txt -k ADDTWO

run1: ${targetname}.exe build1/first.xclbin
	${powershell} ./$< -v 1 -x build1/first.xclbin -i build1/insts.txt -k ADDONE

clean:
	rm -rf build1 build2 _build ${targetname}.exe
