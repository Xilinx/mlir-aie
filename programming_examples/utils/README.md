
<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Programming Exmaples Utilities</ins>

These utilities are helpful in building and measuring performance of the programming examples designs. They incluse helpful C/C++ libraries, python and shell scripts.

- [Open CV Utilities (OpenCVUtils.h)](./OpenCVUtils.h)
- [Trace parser - eventIR based (parse_eventIR.py)](./parse_eventIR.py)
- [Trace parser, custom - in progress (parse_trace.py)](./parse_trace.py)
- [Clean microcode shell script (clean_microcode.sh)](./clean_microcode.sh)

## <u>Trace parser - eventIR based (parse_eventIR.py)</u>

Once the trace data is stored in a text file, we want to parse it to generate waveform json file. There are 2 flows to do this at the moment, one is a custom parser `parse_trace.py` that will generate a .json file which we can open in Perfetto to view the waveforms. The other is to use the eventIR parser `parse_eventiR.py` which will also generate a .json file. In order to use this second parser, we must first convert our trace data into eventIR format using the Vitis hwfrontend parser which is used by aiesimulator. The goal of this second flow is to leverage the existing trace packet parsing from aiesimulator. Both flows are described below:

### a) Custom trace data parser --> .json
To call our custom parser, we need the following files:
* `trace data text file` - This is generated during the running of our python host code/ jupyter notebook
* `source mlir` - This is needed to parse what events and tiles we are monitoring to generate labels for our waveform visualizer
* `column shift` - This specifies how much the actual design was shifted from the default position when it was scheduled and called. The reason we need this is becuase even if our design is configured for column 0, the actual loading and execution of the design may place it in column 1, 2, 3 etc. We account for this shift since the parser needs to match the actual column location of the generated trace data. Usually 2 is the right value. NOTE - the underlying tools currently default to column 1 to avoid using column 0 on Ryzen AI since that column does not have a shimDMA and is therefore avoided at the moment.

From the notebook folder, where the resnet designs are run, we should make sure we have a trace output folder created (by default, we use `traces`). Then we can run the following command.
```
../../../utils/parse_trace.py --filename traces/bottleneck_cifar_split_vector.txt --mlir ../bottleneck_block/bottleneck_cifar_split_vector/aieWithTrace.mlir --colshift 2 > trace.json
```

### b) Vitis hwfrontend + parser --> .json

NOTE: This flow is still being developed and many of the required steps at this moment should likely be rolled into the script directly. For now, it's probably better to just use the custom parser flow and only use this flow to compare results.

1. Create a dummy file (`.target`) in the current directory with the file content 'hw'
2. Create a template json with the matching tile position and events - <custom config>.json
3. Prepend 0x in front of all event trace packet in trace text file - <0x trace text file>
4. Modify trace text file to workaround possible bugs (see below)
5. Run Vitis frontend parser to generate an event IR text file.
6. Run custom eventIR-to-json parser script (`parse_eventIR.py`)

Step #1 is needed by the `hwfrontend` tool and is just a hidden file that has `hw` in the first line. 

For step #2, we need a template json file that matches the position and event list of our trace. This file should ideally be auto-generated but an example version of this file can be found in `reference_designs/ipu-xrt/resnet/bottleneck_block/bottleneck_cifar_split_vector/traces/bottleneck_cifar_split_vector.json`

In step #3, we need to prepend the trace text file data with `0x` because that's what `hwfrontend` expects. Then in step #4, there are currently a few bugs that we may need to work around by editing the trace text file further as described below.

<u>Workaround 1</u>: For case where the start event is 1 or maybe in general, the trace output might have a few packets with just `0xdbffdbff` data. These seem to give the following error and needs to have those packets deleted up to an actual valid event packet.
```
CRITICAL WARNING: [hwanalyze 77-5570] trace_out_hex3:1 Start Frame for Tile(2, 4) Module: cm looks to be missing as trace configuration is not available.
```
<u>Workaround 2</u>: If the start timer value is too large, it reports an error:
```
WARNING: [hwanalyze 77-5569] trace_out_hex2:1 Overrun Tile(2, 4) Module: cm. Hence further decoding of this particular module will be skipped.
```
So reducing the start frame from something like:
```
0xf4000000
0x00a93c62
```
to
```
0xf0000000
0x0005d0f7
```
which reduces the timer from 11,091,042 cycles to 381,175 seems to fix it.
Step #5 is running the aiesimulator frontend parser which generates the eventIR text file.
```
hwfrontend --trace <0x trace text file> --trace_config <custom config>.json --pkg-dir . --outfile <output text file>
```

Step #6 is running the custom eventIR-to-json parser script (`parse_eventIR.py`) to generate the json file.
```
../../../utils/parse_eventIR.py --filename <output text file> --mlir ../bottleneck_block/bottleneck_cifar_split_vector/aieWithTrace.mlir --colshift 2 > trace_eventIR.json
```
Note that there is a sample of these steps for both the custom parser and the eventIR parser in the Makefile for bottleneck_cifar_split_vector located at `reference_designs/ipu-xrt/resnet/bottleneck_block/bottleneck_cifar_split_vector/traces/Makefile`. It doesn't have commands for step 3 and 4 which would need to be done by hand.

