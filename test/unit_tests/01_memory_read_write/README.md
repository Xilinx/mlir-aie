Current test doesn't create a working elf from peana so we need to build the design directly via xchessmk.

```
cd chess_example
./build_run.sh
<ctrl-c to exit after simulation is run>
cp kernel ..
cd ..
make aie.elf
make aie_inc.cpp
```

Then you can copy the entire 01_memory_read_write diretory to the board and then compile the arm code on the board and run the test.
```
scp -r 01_memory_read_write xilinx@<board ip>:/home/xilinx/.
```

On the board, execute:
```
cd 01_memory_read_write
make test.exe
sudo ./test.exe
```
