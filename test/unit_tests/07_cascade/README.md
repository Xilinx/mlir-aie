To build design, simply type:
```
make aie13.elf
make aie23.elf
make aie_inc.cpp
```
Then you can copy the entire 07_cascade diretory to the board and then compile the arm code on the board and run the test.
```
scp -r 07_cascade xilinx@<board ip>:/home/xilinx/.
```

On the board, execute:
```
cd 07_cascade
make test.exe
sudo ./test.exe
```

