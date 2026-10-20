#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Check the numpy oracle against ONNX Runtime on the XINT8 model, tensor by tensor.

    python verify_oracle.py --n 8

Runs both on letterboxed coco128 images and reports, for every int8 tensor
the spec names, how many elements differ, then whether the final (300, 6)
detections are identical. Exits nonzero on any difference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx  # pyright: ignore[reportMissingImports]
import onnxruntime as ort  # pyright: ignore[reportMissingImports]
from letterbox import letterbox
from yolo26n_numpy import Oracle
from yolo_params import DATA_DIR, WEIGHTS_DIR, Params


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--onnx", type=Path, default=WEIGHTS_DIR / "yolo26n_xint8.onnx")
    parser.add_argument("--params", type=Path, default=DATA_DIR / "yolo26n_params.npz")
    parser.add_argument(
        "--images", type=Path, default=WEIGHTS_DIR / "coco128" / "images" / "train2017"
    )
    parser.add_argument("--n", type=int, default=4, help="images to compare")
    args = parser.parse_args()

    params = Params.load(args.params)
    oracle = Oracle(params)
    names = {
        key.removesuffix("|onnx"): str(value)
        for key, value in params.items()
        if key.endswith("|onnx")
    }
    model = onnx.load(args.onnx)
    for onnx_name in names.values():
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(onnx_name))
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    outputs = [o.name for o in session.get_outputs()]

    failed = False
    for path in sorted(args.images.glob("*.jpg"))[: args.n]:
        image = letterbox(path)
        ours = oracle.run(image)
        x = (image.transpose(2, 0, 1)[None] / np.float32(255)).astype(np.float32)
        theirs = {
            name: np.asarray(value)
            for name, value in zip(outputs, session.run(outputs, {"images": x}))
        }
        diffs = []
        for name, onnx_name in names.items():
            a, b = ours[name], theirs[onnx_name].reshape(ours[name].shape).astype(
                np.int64
            )
            if (a != b).any():
                diffs.append(
                    f"{name}: {(a != b).sum()}/{a.size} differ, max |d| {np.abs(a - b).max()}"
                )
        dets = ours[oracle.network.ops[-1].dst.name]
        same = np.array_equal(dets, theirs[outputs[0]][0])
        print(
            f"{path.name}: {len(names) - len(diffs)}/{len(names)} tensors exact, detections {'exact' if same else 'DIFFER'}"
        )
        for line in diffs[:10]:
            print("   ", line)
        failed |= bool(diffs) or not same
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
