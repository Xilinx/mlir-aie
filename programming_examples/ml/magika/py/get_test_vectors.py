# Copyright (C) 2019-2022 Xilinx, Inc.
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: LicenseRef-AMD-Proprietary
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
