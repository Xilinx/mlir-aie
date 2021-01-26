To build design, simply type:
```
make aie13.elf
make aie33.elf
make aie_inc.cpp
```
Then you can copy the entire 08_stream_broadcast diretory to the board and then compile the arm code on the board and run the test.
```
scp -r 08_stream_broadcast xilinx@<board ip>:/home/xilinx/.
```

On the board, execute:
```
cd 08_stream_broadcast
make test.exe
sudo ./test.exe
```
