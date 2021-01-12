To build design, simply type:
```
make aie13.elf
make aie23.elf
make aie_inc.cpp
```
Then you can copy the entire 04_shared_memory diretory to the board and then compile the arm code on the board and run the test.
```
scp -r 04_shared_memory xilinx@<board ip>:/home/xilinx/.
```

On the board, execute:
```
cd 04_shared_memory
make test.exe
sudo ./test.exe
```

