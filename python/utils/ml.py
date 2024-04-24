# ml.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import os
from torch.utils.data import Dataset

# from PIL import Image
import json
import sys
import csv
import json
import argparse
import numpy as np

# import cv2
import numpy as np

# from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn

# from prettytable import PrettyTable
import math


# class ImageNetKaggle(Dataset):
#     def __init__(self, root, split, transform=None):
#         self.samples = []
#         self.targets = []
#         self.transform = transform
#         self.syn_to_class = {}
#         with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
#             json_file = json.load(f)
#             for class_id, v in json_file.items():
#                 self.syn_to_class[v[0]] = int(class_id)
#         with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
#             self.val_to_syn = json.load(f)
#         samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
#         for entry in os.listdir(samples_dir):
#             if split == "train":
#                 syn_id = entry
#                 target = self.syn_to_class[syn_id]
#                 syn_folder = os.path.join(samples_dir, syn_id)
#                 for sample in os.listdir(syn_folder):
#                     sample_path = os.path.join(syn_folder, sample)
#                     self.samples.append(sample_path)
#                     self.targets.append(target)
#             elif split == "val":
#                 syn_id = self.val_to_syn[entry]
#                 target = self.syn_to_class[syn_id]
#                 sample_path = os.path.join(samples_dir, entry)
#                 self.samples.append(sample_path)
#                 self.targets.append(target)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         x = Image.open(self.samples[idx]).convert("RGB")
#         if self.transform:
#             x = self.transform(x)
#         return x, self.targets[idx]


class CSVLogger:
    def __init__(self, filename, sep=","):
        self.filename = str(filename)
        if os.path.exists(self.filename):
            with open(self.filename) as f:
                self.columns = csv.DictReader(f).fieldnames
        else:
            self.columns = None
        self.fh = open(self.filename, "a", newline="")
        self.csvwriter = csv.writer(self.fh, delimiter=sep)
        self.count = 0

    def set_columns(self, columns):
        if self.columns:
            raise Exception("Columns already set")
        self.columns = list(columns)
        self.csvwriter.writerow(self.columns)

    def append(self, row):
        if self.columns is None:
            self.set_columns(row.keys())
        self.csvwriter.writerow([row.get(k, "-") for k in self.columns])
        self.count += 1
        if self.count > 100:
            self.count = 0
            self.fh.flush()

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_class_label(class_label_file: str, num_classes: int) -> list:
    class_label = json.load(open(class_label_file))
    class_label_list = [class_label[str(i)] for i in range(num_classes)]

    return class_label_list


# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         param = parameter.numel()
#         table.add_row([name, param])
#         total_params += param
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict


# def extract_cifar():
#     datafile = r"./data_torchvision/cifar-10-batches-py/test_batch"
#     metafile = r"./data_torchvision/cifar-10-batches-py/batches.meta"

#     data_batch_1 = unpickle(datafile)
#     metadata = unpickle(metafile)

#     images = data_batch_1["data"]
#     labels = data_batch_1["labels"]
#     images = np.reshape(images, (10000, 3, 32, 32))

#     import os

#     dirname = "cifar_images"
#     if not os.path.exists(dirname):
#         os.mkdir(dirname)

#     # Extract and dump first 10 images
#     for i in range(0, 100):
#         im = images[i]
#         im = im.transpose(1, 2, 0)
#         im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#         im_name = f"./cifar_images/image_{i}.png"
#         cv2.imwrite(im_name, im)


def fuse_single_conv_bn_pair(bn_mean, bn_var, bn_wts, bn_bias, conv_wts):
    # https://github.com/ChoiDM/Pytorch_BN_Fold/blob/master/bn_fold.py
    eps = 1e-05
    mu = bn_mean
    var = bn_var
    gamma = bn_wts
    # if 'bias' in bn_bias:
    #     beta = bn_bias
    # else:
    #     beta = torch.zeros(gamma.size(0)).float()
    beta = bn_bias
    # Conv params
    W = conv_wts

    denom = torch.sqrt(var + eps)

    A = gamma.div(denom)
    # bias = torch.zeros(W.size(0)).float()
    # b = beta - gamma.mul(mu).div(denom)
    # bias *= A
    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)
    A = A.to(torch.int8)
    W.mul_(A)
    # bias.add_(b)

    return W


class DataShaper:
    def __init__(self, defOrder="RC", print_info=False):
        self.defOrder = defOrder
        self.print_info = print_info
        self.log_msg = []

    def _reorder_granularity_range(
        self, order, z, start_data_dim=-1, stop_str_dim=None
    ):
        if stop_str_dim is None:
            stop_str_dim = len(order)
        gran = 1
        pre_group = {}
        for idx, s in enumerate(order.split(z)[1:]):
            step = ""
            pg = ""
            off = 0
            if s.find(")") >= 0 and (s.find("(") > s.find(")") or s.find("(") < 0):
                pg, s = s.split(")", 1)
            for i, c in enumerate(s):
                if c.isdigit():
                    step += c
                else:
                    if c == ">":
                        off = int(s[i + 1])
                    if c == "<":
                        off = -int(s[i + 1])
                    break
            if len(pg) > 0 and pg[0] in "<>":
                if pg[0] == ">":
                    off = int(pg[1])
                if pg[0] == "<":
                    off = -int(pg[1])
                pg = pg[2:]
            if step and idx + off > start_data_dim and idx <= stop_str_dim:
                gran *= int(step)
                for p in pg:
                    if p.isdigit() or p in "<>":
                        continue
                    elif p in pre_group:
                        pre_group[p] *= int(step)
                    else:
                        pre_group[p] = int(step)
        return gran, pre_group

    def _reorder_decode(self, shape, order, defOrder=None):
        if not defOrder:
            defOrder = self.defOrder
        Ds = [order.count(c) for c in defOrder]
        D = list(shape)
        size = [0] * sum(Ds)
        perm = [0] * sum(Ds)
        pad_im = [0] * len(shape)
        pad_ex = [0] * sum(Ds)
        brdcst = [1] * sum(Ds)
        align = [1] * sum(Ds)
        val = ""
        val_gi = ""
        off = 0
        group = False
        d = [sum(Ds[0 : i + 1]) - 1 for i in range(len(Ds))]
        p = sum(Ds) - 1
        for z in reversed(order):
            if z.isdigit():
                if group:
                    val_gi = z + val_gi
                else:
                    val = z + val
            elif z == ">":
                if group:
                    off = int(val_gi)
                    val_gi = ""
                else:
                    off = int(val)
                    val = ""
            elif z == "<":
                if group:
                    off = -int(val_gi)
                    val_gi = ""
                else:
                    off = -int(val)
                    val = ""
            elif z == ")":
                group = True
            elif z == "(":
                group = False
                off = 0
                val = ""
                val_gi = ""
            elif z == "%":  # Pad dimension by N
                pad_ex[p + 1] += max(0, int(val) - 1) * (
                    size[perm[p + 1]] + pad_ex[p + 1]
                )
                val = ""
            elif z == "*":  # Broadcast dimension by N
                brdcst[p] *= int(val)
                val = ""
            elif z == "|":  # Align data after a dimension to N
                align[p] *= int(val)
                val = ""
            elif z in defOrder:
                idx = defOrder.find(z)
                perm[p] = d[idx] + off
                if off < 0:
                    start_dim = d[idx] + off - sum(Ds[0:idx]) if val else -1
                    stop_dim = d[idx] - sum(Ds[0:idx])
                    gran, pre_group = self._reorder_granularity_range(
                        order, z, start_dim, stop_dim
                    )
                    for i, c in enumerate(defOrder):
                        if c in pre_group:
                            if D[i] >= pre_group[c]:
                                gran //= pre_group[c]
                            elif D[i] > 1:
                                gran = int(math.ceil(1.0 * gran / D[i]))
                    D_rem = max(1, D[idx] // gran)
                else:
                    D_rem = D[idx]
                if val:
                    vi = int(val)
                    if group:
                        if vi > D_rem:
                            vi_rem = vi // D_rem
                            vi //= vi_rem
                            val = str(vi_rem)
                        else:
                            val = "1"
                    else:
                        val = ""
                else:
                    vi = D_rem
                if vi > 0:
                    if D[idx] % vi != 0:
                        dim_sub = np.prod(
                            np.maximum(1, size[sum(Ds[0:idx]) : sum(Ds[0 : idx + 1])])
                        )
                        pad_im[idx] += (vi - D[idx] % vi) * dim_sub
                    size[d[idx] + off] = vi
                    D[idx] = int(math.ceil(1.0 * D[idx] / vi))
                if not group:
                    off = 0
                d[idx] -= 1
                p -= 1
        if self.print_info:
            self.log_msg.append(
                "[INFO]: reorder s={:<15} o={:<15} -> pi={:<15} s={:<30} p={:<30} pe={:<30}, b={:<30}, a={:<30}".format(
                    *map(str, (shape, order, pad_im, size, perm, pad_ex, brdcst, align))
                )
            )
        return pad_im, size, perm, pad_ex, brdcst, align

    def reorder_mat(self, mat, order, defOrder=None, inverse=False):
        pad_im, size, perm, pad_ex, brdcst, align = self._reorder_decode(
            mat.shape, order, defOrder
        )
        if not inverse:
            if sum(pad_im) > 0:
                mat = np.pad(mat, tuple(zip([0] * len(pad_im), pad_im)), "constant")
            mat = mat.reshape(*size).transpose(perm)
            if sum(pad_ex) > 0:
                mat = np.pad(mat, tuple(zip([0] * len(pad_ex), pad_ex)), "constant")
            if np.prod(brdcst) > 1:
                for idx, b in enumerate(brdcst):
                    if b > 1:
                        mat = np.repeat(mat, b, axis=idx)
            if np.prod(align) > 1:
                for idx, a in reversed(tuple(enumerate(align))):
                    if a > 1:
                        mat = mat.reshape(mat.shape[: idx + 1] + (-1,))
                        pad = a - (mat.shape[-1] % a)
                        if pad < a:
                            mp = np.zeros((len(mat.shape), 2), dtype=np.int)
                            mp[-1, -1] = pad
                            mat = np.pad(mat, mp, "constant")
        else:
            assert sum(pad_im) == 0, "Reverse of implicit padding not supported"
            assert sum(pad_ex) == 0, "Reverse of explicit padding not supported"
            assert np.prod(brdcst) == 1, "Reverse of broadcasting not supported"
            assert np.prod(align) == 1, "Reverse of alignment not supported"
            perm_inv = [perm.index(p) for p in range(len(perm))]
            size_inv = [size[p] for p in perm]
            mat = mat.reshape(*size_inv)
            mat = mat.transpose(perm_inv)

        return mat.reshape(-1)

    def get_dim_steps(
        self, shape, order, defOrder=None, bits=8, ebs=None, sparse_ratio=1
    ):
        pad_im, size, perm, pad_ex, brdcst, align = self._reorder_decode(
            shape, order, defOrder
        )
        sz = 1
        d = len(shape) - 1
        sp = len(perm) - 1
        dim = [0] * len(shape)
        for i, s in enumerate(reversed(size)):
            sz *= s
            if sz >= shape[d] + pad_im[d]:
                # current dimension contains all elements
                p = len(perm) - 1 - i
                dim[d] = pi0 = perm.index(p)
                if p + 1 < sp and p + 1 in perm:
                    pi1 = perm.index(p + 1)
                    if pi0 + 1 == pi1:
                        # Found XX coupling
                        dim[d] = pi1
                        self.log_msg.append(
                            "INFO: Found XX coupling (order={}, size={}, perm={}, p={})".format(
                                order, size, perm, p
                            )
                        )
                    elif len(perm) > pi0 + 1:
                        pb = perm[pi0 + 1]
                        if pi0 + 2 == pi1 and size[pb] == 1:
                            # Found XNX sequence with N=1, simplify
                            dim[d] = pi1
                            self.log_msg.append(
                                "INFO: Found XNX sequency with N=1, simplify (order={}, size={}, perm={}, p={})".format(
                                    order, size, perm, p
                                )
                            )
                sz = 1
                d -= 1
                sp = p
        # dim = [perm.index(p) for p in dim]
        size_inv = (np.array(size)[perm] + pad_ex) * brdcst
        idx = -2 if bits == 4 and size_inv[-1] == 2 else -1
        if ebs or sparse_ratio:
            assert (
                size_inv[idx] >= 8
            ), "Sparse/exponent block is too small. Data (order) unexpected or update to script is required"
        size_inv[idx] = int(
            size_inv[idx] * sparse_ratio * (bits - (8 if ebs else 0)) / 8
        ) + (size_inv[idx] // ebs if ebs else 0)
        step = [0] * (len(shape) + 1)
        cur = 1
        for i_rev, (s, al) in enumerate(reversed(tuple(zip(size_inv, align)))):
            i = len(perm) - 1 - i_rev
            if al > 1:
                cur = ((cur + al - 1) // al) * al
            if i in dim:
                step[dim.index(i)] = cur
            cur *= s
        step[-1] = cur
        return step


if __name__ == "__main__":
    extract_cifar()
