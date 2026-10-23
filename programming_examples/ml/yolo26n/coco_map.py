#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""mAP@0.5 of the detector's outputs against YOLO-format labels, as Ultralytics 8.4's validator computes it.

Labels go into the letterboxed frame the detections are in; per image, each
detection above the confidence floor, in the model's (descending
confidence) order, claims the unclaimed label of its class it overlaps most,
a true positive at IoU 0.5 or more; per labelled class, the AP is the
101-point interpolated area under its precision envelope; the mAP is their
mean over the classes the labels hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ultralytics' validation defaults: the confidence floor, IoU threshold and
# the epsilons of its box_iou and ap_per_class.
CONF = 0.001
IOU = 0.5
_IOU_EPS = np.float32(1e-7)
_RECALL_EPS = 1e-16


@dataclass(frozen=True)
class Labels:
    """One image's labelled objects in the letterboxed frame.

    Attributes:
        classes: `(n,)` class ids.
        boxes: `(n, 4)` float32 x1, y1, x2, y2, in letterboxed pixels.
    """

    classes: np.ndarray
    boxes: np.ndarray

    @classmethod
    def read(
        cls, path: Path, height: int, width: int, ratio: float, left: int, top: int
    ) -> Labels:
        """Read a YOLO label file (class, then centre x, y, width, height over the image's size).

        Args:
            path: The label file; an image with no objects has none.
            height: The original image's height.
            width: Its width.
            ratio: The letterbox's scale.
            left: Its left padding.
            top: Its top padding.

        Returns:
            The labels, placed as the letterbox placed the image.
        """
        rows = (
            np.loadtxt(path, dtype=np.float32, ndmin=2)
            if path.exists()
            else np.zeros((0, 5), np.float32)
        )
        cx, cy, w, h = rows[:, 1:5].T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        scale = np.array([width, height, width, height], np.float32) * np.float32(ratio)
        offset = np.array([left, top, left, top], np.float32)
        return cls(rows[:, 0].astype(np.int64), xyxy * scale + offset)


class MeanAP50:
    """mAP@0.5 over the images it is given, one at a time."""

    def __init__(self) -> None:
        self.images = 0
        self._tp: list[np.ndarray] = []
        self._conf: list[np.ndarray] = []
        self._pred_classes: list[np.ndarray] = []
        self._label_classes: list[np.ndarray] = []

    @property
    def objects(self) -> int:
        """Labelled objects seen so far."""
        return sum(len(c) for c in self._label_classes)

    def add(self, detections: np.ndarray, labels: Labels) -> None:
        """Score one image's detections.

        Args:
            detections: `(max_det, 6)` float32 x1, y1, x2, y2, score, class,
                in descending confidence (the model's own order, since its
                top-k already sorts them): two detections of one label's
                class that both reach it are matched in this order, the
                first claiming it and the second scored a false positive.
            labels: The image's labels.
        """
        kept = detections[detections[:, 4] > CONF]
        boxes, classes = kept[:, :4], kept[:, 5].astype(np.int64)
        tp = np.zeros(len(kept), bool)
        if len(kept) and len(labels.classes):
            lo = np.maximum(labels.boxes[:, None, :2], boxes[None, :, :2])
            hi = np.minimum(labels.boxes[:, None, 2:], boxes[None, :, 2:])
            wh = np.clip(hi - lo, 0, None)
            inter = wh[..., 0] * wh[..., 1]
            area_l = (labels.boxes[:, 2] - labels.boxes[:, 0]) * (
                labels.boxes[:, 3] - labels.boxes[:, 1]
            )
            area_d = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iou = inter / (area_l[:, None] + area_d[None, :] - inter + _IOU_EPS)
            iou = iou * (labels.classes[:, None] == classes[None, :])
            claimed = np.zeros(len(labels.classes), bool)
            for j in np.flatnonzero((iou >= IOU).any(0)):
                available = np.where(claimed, 0, iou[:, j])
                k = available.argmax()
                tp[j] = available[k] >= IOU
                claimed[k] |= tp[j]
        self.images += 1
        self._tp.append(tp)
        self._conf.append(kept[:, 4])
        self._pred_classes.append(classes)
        self._label_classes.append(labels.classes)

    def value(self) -> float:
        """The mean over labelled classes of each one's AP at IoU 0.5; 0 with no labels."""
        tp, conf = np.concatenate(self._tp), np.concatenate(self._conf)
        pred_classes = np.concatenate(self._pred_classes)
        order = np.argsort(-conf)
        tp, pred_classes = tp[order], pred_classes[order]
        classes, counts = np.unique(
            np.concatenate(self._label_classes), return_counts=True
        )
        if not len(classes):
            return 0.0
        ap = np.zeros(len(classes))
        for ci, (c, n_labels) in enumerate(zip(classes, counts)):
            mine = pred_classes == c
            if not mine.any():
                continue
            tpc = tp[mine].cumsum()
            fpc = (1 - tp[mine]).cumsum()
            recall = tpc / (n_labels + _RECALL_EPS)
            precision = tpc / (tpc + fpc)
            mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
            mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            x = np.linspace(0, 1, 101)
            y = np.interp(x, mrec, mpre)
            ap[ci] = (np.diff(x) * (y[1:] + y[:-1]) / 2.0).sum()
        return float(ap.mean())
