To build design, simply type:
```
make aie.elf
make aie_inc.cpp
```
Then you can copy the entire 02_lock_acquire_release diretory to the board and then compile the arm code on the board and run the test.
```
scp -r 02_lock_acquire_release xilinx@<board ip>:/home/xilinx/.
```

On the board, execute:
```
cd 02_lock_acquire_release
make test.exe
sudo ./test.exe
```

