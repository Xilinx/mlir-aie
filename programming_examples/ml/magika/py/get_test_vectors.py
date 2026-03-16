#  (c) Copyright 2019-2022 Xilinx, Inc. All rights reserved.
#  (c) Copyright 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
#  This file contains confidential and proprietary information
#  of Xilinx, Inc. and is protected under U.S. and
#  international copyright and other intellectual property
#  laws.
#
#  DISCLAIMER
#  This disclaimer is not a license and does not grant any
#  rights to the materials distributed herewith. Except as
#  otherwise provided in a valid license issued to you by
#  Xilinx, and to the maximum extent permitted by applicable
#  law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
#  WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
#  AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
#  BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
#  INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
#  (2) Xilinx shall not be liable (whether in contract or tort,
#  including negligence, or under any other theory of
#  liability) for any loss or damage of any kind or nature
#  related to, arising under or in connection with these
#  materials, including for any direct, or any indirect,
#  special, incidental, or consequential loss or damage
#  (including loss of data, profits, goodwill, or any type of
#  loss or damage suffered as a result of any action brought
#  by a third party) even if such damage or loss was
#  reasonably foreseeable or Xilinx had been advised of the
#  possibility of the same.
#
#  CRITICAL APPLICATIONS
#  Xilinx products are not designed or intended to be fail-
#  safe, or for use in any application requiring fail-safe
#  performance, such as life-support or safety devices or
#  systems, Class III medical devices, nuclear facilities,
#  applications related to the deployment of airbags, or any
#  other applications that could lead to death, personal
#  injury, or severe property or environmental damage
#  (individually and collectively, "Critical
#  Applications"). Customer assumes the sole risk and
#  liability of any use of Xilinx products in Critical
#  Applications, subject only to applicable laws and
#  regulations governing limitations on product liability.
#
#  THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
#  PART OF THIS FILE AT ALL TIMES.
import onnx
import onnxruntime as ort
import numpy as np
from collections import OrderedDict
import os
import sys

if len(sys.argv) < 2:
    model_name = "model.onnx"
else:
    model_name = sys.argv[1]

if len(sys.argv) < 3:
    ntest = 2
else:
    ntest = int(sys.argv[2])

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

# check if the data folder exists or not
output_path = "../data/"

if not os.path.exists(output_path):
    os.mkdir(output_path)

# model_orig = onnx.load(model_name)

ort_session = ort.InferenceSession(
    model_name, providers=["CPUExecutionProvider"], sess_options=so
)

for testid in range(ntest):

    ort_input0 = np.random.randint(0, 257, size=(1, 2048), dtype=np.int32)

    # ort_session_orig = ort.InferenceSession(model_orig.SerializeToString(), providers=["CPUExecutionProvider"], sess_options=so)
    names_outs = [x.name for x in ort_session.get_outputs()]

    ort_outs = ort_session.run(names_outs, {"bytes": ort_input0})

    # Dictionary which contains all intermediate outputs,
    # intermediate outputs can be extracted by calling ort_outs_orig_dict['<intermedite_name>']
    ort_outs_dict = OrderedDict(zip(names_outs, ort_outs))

    # save input with 2 integers per line
    a = np.reshape(ort_input0, [1024, 2])

    if testid == 0:
        fmode = "w"
    else:
        fmode = "a"

    with open(output_path + "/din.txt", fmode) as f:
        np.savetxt(f, a, fmt="%d")

    # save output
    a = ort_outs_dict["target_label"]
    a = np.reshape(a, [-1])
    with open(output_path + "/ref.txt", fmode) as f:
        np.savetxt(f, a, fmt="%e")

del ort_session

print(
    "\nTest vectors are generated for "
    + str(ntest)
    + " inferences of "
    + model_name
    + ".\n\n"
)
