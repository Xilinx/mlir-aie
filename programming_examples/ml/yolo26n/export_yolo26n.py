#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Fetch Ultralytics yolo26n, export it to ONNX at 640 and quantize it to XINT8.

Writes into `WEIGHTS_DIR` (`$IRON_EXAMPLE_WEIGHTS_DIR/yolo26n`, default
`/srv/yolo26n`), which is kept out of the repository: the checkpoint is
AGPL-3.0. Produces:

    yolo26n.pt          the stock checkpoint (Ultralytics assets v8.4.0)
    yolo26n.onnx        FP32, end-to-end (one2one head, top-k inside, (1, 300, 6))
    yolo26n_xint8.onnx  AMD Quark XINT8: int8, power-of-two scales, zero points 0
    coco128/            the calibration images

The checkpoint and coco128 are checked against pinned sha256 digests before
use; until a digest is pinned, the script stops and prints the file's.

Quantization choices, each measured on coco128 (mAP50-95, FP32 0.474):
- The end-to-end postprocess after the Detect convs stays float. Quantizing
  its final concat puts boxes, scores and class ids on one int8 scale.
- `SimulateDPU` is off, so SiLU keeps its real sigmoid rather than Quark's
  HardSigmoid. Every SiLU is an int8 -> int8 table on the NPU either way.
"""

import argparse
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import onnx  # pyright: ignore[reportMissingImports]
from letterbox import IMAGE_SIZE, letterbox
from onnxruntime.quantization import (  # pyright: ignore[reportMissingImports]
    CalibrationDataReader,
)
from quark.onnx import (  # pyright: ignore[reportMissingImports]
    AdaRoundConfig,
    ModelQuantizer,
    QConfig,
    QLayerConfig,
    XInt8Spec,
)
from ultralytics import YOLO  # pyright: ignore[reportMissingImports]
from ultralytics.utils.downloads import (  # pyright: ignore[reportMissingImports]
    safe_download,
)
from yolo_params import WEIGHTS_DIR

CHECKPOINT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
)
COCO128_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
)
# Pin each from a download you have checked; None stops the export.
CHECKPOINT_SHA256: str | None = (
    "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
)
COCO128_SHA256: str | None = (
    "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"
)


def check_sha256(path: Path, pinned: str | None) -> None:
    """Refuse a downloaded file whose sha256 is not the pinned one.

    Args:
        path: The file.
        pinned: Its expected sha256, or None while none is pinned.

    Raises:
        SystemExit: If no digest is pinned yet, or the file's differs.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if pinned is None:
        raise SystemExit(
            f"{path.name}'s sha256 is not pinned yet; it is {digest}. Check the "
            "file, pin that digest in export_yolo26n.py and run again."
        )
    if digest != pinned:
        raise SystemExit(f"{path} has sha256 {digest}, not the pinned {pinned}")


class CalibrationImages(CalibrationDataReader):
    def __init__(self, paths: list[Path]) -> None:
        self.batches = [
            {
                "images": (
                    letterbox(p)[0].transpose(2, 0, 1)[None] / np.float32(255)
                ).astype(np.float32)
            }
            for p in paths
        ]
        self.index = 0

    # ORT's reader protocol ends with None, which the base annotation omits.
    def get_next(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.index == len(self.batches):
            return None
        self.index += 1
        return self.batches[self.index - 1]

    def rewind(self) -> None:
        self.index = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    parser.add_argument(
        "--n-calib", type=int, default=128, help="coco128 images to calibrate on"
    )
    parser.add_argument(
        "--no-adaround", action="store_true", help="skip AdaRound weight rounding"
    )
    args = parser.parse_args()

    out = args.weights_dir
    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / "yolo26n.onnx"
    if not fp32.exists():
        checkpoint = out / "yolo26n.pt"
        if not checkpoint.exists():
            safe_download(CHECKPOINT_URL, file=checkpoint)
        check_sha256(checkpoint, CHECKPOINT_SHA256)
        exported = YOLO(str(checkpoint)).export(
            format="onnx",
            imgsz=IMAGE_SIZE,
            opset=17,
            simplify=True,
            dynamic=False,
            nms=False,
        )
        Path(exported).replace(fp32)

    images = out / "coco128" / "images" / "train2017"
    if not images.is_dir():
        archive = out / "coco128.zip"
        safe_download(COCO128_URL, file=archive, unzip=False)
        check_sha256(archive, COCO128_SHA256)
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
        archive.unlink()
    paths = sorted(images.glob("*.jpg"))[: args.n_calib]

    config = QConfig(
        global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
        algo_config=[] if args.no_adaround else [AdaRoundConfig()],
        exclude=[
            node.name
            for node in onnx.load(fp32).graph.node
            if node.name.startswith("/model.23/")
            and node.op_type != "Conv"
            and "/act/" not in node.name
        ],
        extra_options={"SimulateDPU": False},
        use_external_data_format=False,
    )
    ModelQuantizer(config).quantize_model(
        str(fp32), str(out / "yolo26n_xint8.onnx"), CalibrationImages(paths)
    )
    print(f"wrote {out / 'yolo26n_xint8.onnx'} (calibrated on {len(paths)} images)")


if __name__ == "__main__":
    main()
