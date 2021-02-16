To build design, simply type:
```
make 
```
Then you can copy the entire 05_tiledma diretory to the board and then compile the arm code on the board and run the test.
```
scp -r 05_tiledma xilinx@<board ip>:/home/xilinx/.
```

On the board, execute:
```
cd 05_tiledma
cp acdc_project/aie_inc.cpp .
cp acdc_project/core*elf .
make test.exe
sudo ./test.exe
```

