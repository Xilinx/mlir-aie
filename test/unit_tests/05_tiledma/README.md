To build design, simply type:
```
make aie13.elf
make aie33.elf
make aie_inc.cpp
```
Then you can copy the entire 05_tiledma diretory to the board and then compile the arm code on the board and run the test.
```
scp -r 05_tiledma xilinx@<board ip>:/home/xilinx/.
```

On the board, execute:
```
cd 05_tiledma
make test.exe
sudo ./test.exe
```

