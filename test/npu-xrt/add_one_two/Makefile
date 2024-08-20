include ../makefile-common

targetname = addOneObjfifo

all: build/final.xclbin build/insts.txt

build/one.xclbin: aie1.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --xclbin-kernel-name=ADDONE --xclbin-kernel-id=0x901 --xclbin-instance-name=ADDONEINST --no-aiesim  --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)

build/two.xclbin: aie2.mlir build/one.xclbin
	cd ${@D} && aiecc.py --xclbin-kernel-name=ADDTWO --xclbin-kernel-id=0x902 --xclbin-instance-name=ADDTWOINST --no-aiesim  --aie-generate-cdo --no-compile-host --xclbin-input=one.xclbin  --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)

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

run: ${targetname}.exe build/two.xclbin build/insts.txt 
	${powershell} ./$< -x build/two.xclbin -i build/insts.txt

clean:
	rm -rf build _build ${targetname}.exe