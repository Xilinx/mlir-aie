To build design, you need to have xchessde tools in your path. Then simply type:
```
cd cascade_kernels
./build_run.sh
cd ..
make 
```
This will generate the elfs and copy them into the 07_cascade directory. Then you can copy the entire 07_cascade diretory to the board and then compile the arm code on the board and run the test.
```
scp -r 07_cascade xilinx@<board ip>:/home/xilinx/.
```

On the board, execute:
```
cd 07_cascade
cp acdc_project/aie_inc.cpp .
make test.exe
sudo ./test.exe
```
