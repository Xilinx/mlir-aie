import os;

for action in ["rm -f","touch"]:
    cmd = f"{action} results.csv"
    os.system(cmd)


for s in [16384,32768,65536,131072,262144]:
    for i in [64,128,256,512,1024]:
        for f in ["bf16_softmax.mlir", "test.cpp", "aie2.py"]:
            sed = f"sed 's\\1024\\{i}\g' {f}.orig > {f}.first"
            os.system(sed)
            sed = f"sed 's\\65536\\{s}\g' {f}.first > {f}"
            os.system(sed)
        make_clean = f"make clean > /dev/null"
        os.system(make_clean)
        make_all = f"make all"
        os.system(make_all)
        make_profile = f"make profile"
        os.system(make_profile)    