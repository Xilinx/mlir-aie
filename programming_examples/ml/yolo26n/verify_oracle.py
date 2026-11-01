#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Check the numpy oracle against ONNX Runtime on the XINT8 model, tensor by tensor.

    python verify_oracle.py --n 8

Runs both on letterboxed coco128 images and reports, for every int8 tensor
the spec names, how many elements differ, then whether the final (300, 6)
detections are identical.

The two softmaxes round differently only near a rounding boundary. Every
attention probability on which they differ goes to `SoftmaxArbiter`, which
computes it exactly. When ONNX Runtime's is the one off at each, the oracle
is run again with ONNX Runtime's probabilities, and must then match it in
every tensor: the image counts as "ORT off-boundary". Exits 0 only when
every image is exact or ORT off-boundary.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx  # pyright: ignore[reportMissingImports]
from gen_yolo_data import QdqGraph, ort_session
from letterbox import letterbox
from yolo26n_numpy import Oracle, SoftmaxArbiter
from yolo_params import DATA_DIR, WEIGHTS_DIR, Params
from yolo_spec import Attention


def differences(
    names: dict[str, str], ours: dict[str, np.ndarray], theirs: dict[str, np.ndarray]
) -> list[str]:
    """One line per spec tensor on which the oracle and ONNX Runtime differ."""
    lines = []
    for name, onnx_name in names.items():
        a = ours[name]
        b = theirs[onnx_name].reshape(a.shape).astype(np.int64)
        if (a != b).any():
            lines.append(
                f"{name}: {(a != b).sum()}/{a.size} differ, max |d| {np.abs(a - b).max()}"
            )
    return lines


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
    paths = sorted(args.images.glob("*.jpg"))[: args.n]
    if len(paths) < args.n:
        parser.error(f"{args.images} holds {len(paths)} .jpg images, not --n {args.n}")

    params = Params.load(args.params)
    oracle = Oracle(params)
    names = {
        key.removesuffix("|onnx"): str(value)
        for key, value in params.items()
        if key.endswith("|onnx")
    }
    graph = QdqGraph(args.onnx)
    attentions = [op for op in oracle.network.ops if isinstance(op, Attention)]
    softmax = {}
    for op in attentions:
        scores = graph.node(f"{op.name}/MatMul", "MatMul").output[0]
        probs = graph.node(f"{op.name}/Softmax", "Softmax").output[0]
        softmax[op.name] = (graph.quantized(scores)[0], graph.quantized(probs)[0])
    model = graph.model
    for onnx_name in [*names.values(), *(n for pair in softmax.values() for n in pair)]:
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(onnx_name))
    session = ort_session(model.SerializeToString())
    outputs = [o.name for o in session.get_outputs()]

    outcomes: Counter[str] = Counter()
    for path in paths:
        image, _ = letterbox(path)
        ours = oracle.run(image)
        x = (image.transpose(2, 0, 1)[None] / np.float32(255)).astype(np.float32)
        theirs = {
            name: np.asarray(value)
            for name, value in zip(outputs, session.run(outputs, {"images": x}))
        }
        diffs = differences(names, ours, theirs)
        detections = oracle.network.ops[-1].dst.name
        same = np.array_equal(ours[detections], theirs[outputs[0]][0])
        verdicts: Counter[str] = Counter()
        followed = {}
        for op in attentions:
            score_name, prob_name = softmax[op.name]
            scores = ours[f"{op.name}/scores"]
            probs = ours[f"{op.name}/probs"]
            ort_probs = theirs[prob_name].reshape(probs.shape).astype(np.int64)
            if not np.array_equal(
                theirs[score_name].reshape(scores.shape).astype(np.int64), scores
            ):
                verdicts["scores differ"] += 1
                continue
            arbiter = SoftmaxArbiter(
                int(params.of(op.name, "score_scale")),
                int(params.of(op.name, "prob_scale")),
            )
            for h, i, j in zip(*np.nonzero(probs != ort_probs)):
                verdicts[
                    arbiter.verdict(
                        scores[h, i],
                        int(j),
                        int(probs[h, i, j]),
                        int(ort_probs[h, i, j]),
                    )
                ] += 1
            followed[op.name] = ort_probs
        if not diffs and same:
            outcome = "exact"
        elif set(verdicts) == {"ORT off-boundary"}:
            rerun = oracle.run(image, probabilities=followed)
            exact_after = not differences(names, rerun, theirs) and np.array_equal(
                rerun[detections], theirs[outputs[0]][0]
            )
            outcome = "ORT off-boundary" if exact_after else "FAIL"
        else:
            outcome = "FAIL"
        outcomes[outcome] += 1
        print(
            f"{path.name}: {len(names) - len(diffs)}/{len(names)} tensors exact, "
            f"detections {'exact' if same else 'DIFFER'}; {outcome}"
            + "".join(f"; {n} probabilities {v}" for v, n in sorted(verdicts.items()))
        )
        for line in diffs[:10]:
            print("   ", line)
    print(
        f"verify_oracle: {outcomes['exact']} exact, {outcomes['ORT off-boundary']} "
        f"ORT off-boundary, {outcomes['FAIL']} FAIL of {len(paths)} images"
    )
    sys.exit(1 if outcomes["FAIL"] else 0)


if __name__ == "__main__":
    main()
