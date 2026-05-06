# Bottleneck A Implementation on a Single AI Engine Core

## Overview

This project implements a bottleneck block for a neural network on a single AI Engine core. The bottleneck block is a crucial component in many deep learning architectures, such as in MobileNet and ResNet models. It helps reduce computational complexity and the number of parameters while maintaining performance. The block consists of three layers: a 1x1 convolution, a depthwise 3x3 convolution, and another 1x1 convolution. A skip connection may also be included.

## Contents

- `README.md`: This file, providing an overview and setup instructions.
- `Makefile`: Makefile for building the project.
- `aie2_bottleneckA_subblock.py`: Main bottleneck A design block, repeated for most of the 9 stages in A processing section (with exception to stage 0)
- `aie2_bottleneckA_subblockStatic.py`: static weights version of the bottleneck A subblock
- `aie2_bottleneckA_subblock0.py`: slight variation of the A subblock for stage 0
- `aie2_bottleneckA_subblock0Static.py`: static weights version of the A subblock for stage 0
- `aie2_bottleneckA_subblock_fused2.py` : fused block of two A subblocks (used in such combinations as stage8+stage9 or stage4+stage5)
- `aie2_bottleneckA_subblock_fused2Static.py`: static weights version of fused block
- `aie2_bottleneckAStatic.py`: Full chain of all bottleneck A subblocks for stages 0-9 (uses static weights version of subblocks). In this design, we fused stages 4+5 and stages 8+9.
- `aie2_bn_0_1_2_3_4_5_6_7_8_9.py`: Top level design instantiating `aie2_bottleneckAStatic` chain
- `test_bottleneckA.py`: Testbench host code for the complete chain for stages 0-9

- `aie2_bn_*.py`: Top level design file for single bottleneck stage used for single stage testing
- `test_bn_single.py`: Testbench host code for single bottleneck stage tests (0, 1, 2, 3, 6, 7 available)
- `test_bn_fused2.py`: Testbench host code for 2 fused bottleneck stage tests (8 & 9, 4 & 5)
- `gen_golden_*.py`: Scripts to generate input stimulus, golden reference, weights and scale factors for the given layer.

The below figures shows our implementation of the bottleneck A mapping using one AIE core.
<p align="center">
 <picture>
 <source media="(prefers-color-scheme: light)" srcset="./bottleneck_a.png">
 <img alt="block" src="./bottleneck_a.png">
</picture>
 <h3 align="center">Bottleneck A depth-first mapping on a single AIE core to avoid unnecessary off-chip data movement.
 </h3>
</p>

## Setup

### Building the Project

To compile and run the full chain design:
```
cd bottleneck_A
make run_py
```

To run some of the single kernel design tests where N is the stage number to build and run:
```
make run_py_N
```

To generate input stimulus, golden reference, weights, and scale factors for the full A block:
```
python3 ./gen_golden.py
```
The single test stimulus, golden reference, weights, and scale factors can be likewise generated via:
```
python3 ./gen_golden_bnN.py
```
where N is the layer that you want to generate for.
