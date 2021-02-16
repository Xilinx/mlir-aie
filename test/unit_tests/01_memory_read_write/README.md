To build design, simply type:
```
make 
```
Then you can copy the entire 01_memory_read_write diretory to the board and then compile the arm code on the board and run the test.
```
scp -r 01_memory_read_write xilinx@<board ip>:/home/xilinx/.
```

On the board, execute:
```
cd 01_memory_read_write
cp acdc_project/aie_inc.cpp .
cp acdc_project/core*elf .
make test.exe
sudo ./test.exe
```

To use the synopsys built design instead, you can do the following during the build step:
```
make aie.elf
make aie_inc.cpp
cd chess_example
./build_run.sh
<ctrl-c to exit after simulation is run>
cp work/Release_LLVM/kernel.prx/kernel ../aie.elf
cd ..
```

