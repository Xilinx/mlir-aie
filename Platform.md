# Building a platform for further exploration in hardware

## Prerequisites

```
Xilinx Vitis 2020.1
Petalinux 2020.1
SD card (16 GB+)
vck190 board + usb-c to usb cable + ethernet cable
```

In order to facilitate additional exploration of the MLIR AIE tools in hardware, a basic platform can be built to run the generated programs on an AI Engine-enabled device + board (vck190). The 'platform' sub-directory contains the necessary Makefile and build scripts to simplify the build process. The same system requirements to run Vitis and Petalinux will be required for the platform build flow.

## vck190 'bare' platform

The vck190 'bare' platform is a nearly empty design that enables the necessary NoC connections for any AIE-compiled design. In addition to configuring the CIPS, AI Engines and Noc, it contains minimal PL components sush as clock, reset and a BRAM (64kB) scratchpad attached to the NoC. 


## Build steps

To start the build, do the following:
```sh
cd platforms/vck190_bare
make all
```

The Makefile will first call a tcl script to build the Vivado design and generate an .xsa file. Then, a Makefile in the petalinux subdirectory generates the petalinux linux kernel, a ramdisk rootfs, and boot files. Lastly, petalinux is called to generate a sysroot which can be used during MLIR AIE compilation. 

After the build is complete, format an SD card (16 GB+) in fat32 format and copy the files from the linux subdirectory onto the SD card.
```sh
cp -r platforms/vck190_bare/petalinux/images/linux/* <SD card directory>
```

## Board setup steps
Put the sd card into the micro sd slot to boot up the Versal device (top of the board), connect the board usb-c connector to your host machine and turn on the board. You should run a program like TeraTerm and configure it as a serial port (115200 baud, 8b data, no parity, 1b stop, no flow control). The serial port for Versal is usually the first port but may vary in your setup.

You should then be able to copy binaries compiled by the MLIR AIE tools to test on the board. This is done by first copying the files onto the sd card directly while the card is inserted/ mounted in your host machine. Then reinsert the sd card back into the board. The board boots up as root and the rootfs is running in a ramdisk so no files are saved except those in the primary sd partition. 


## Run binaries steps

To run binaries on the board, do the following after bootup:
```sh
cd /mnt/sd-mmcblk0p1
export XILINX_XRT=/usr
<executable>
```

Another option for copying files onto the card is via an ethernet connection. If the board is connected to your host machine through an ethernet switch, you should be able to scp files to the board. In addition to root, there is a username/password account for petalinux/petalinux. From the host, you can copy files as follows:
```sh
scp <your files> petalinux@192.168.0.101:/home/petalinux/.
```
Then on the board, run:
```sh
cp /home/petalinux/<your files> /mnt/sd-mmcblk0p1/.
```
and run on the board as usual.

-----

<p align="center">Copyright&copy; 2019-2021 Xilinx</p>
