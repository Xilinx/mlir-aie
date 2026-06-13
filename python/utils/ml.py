# ml.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
"""
ML related utilties

* class `CSVLogger`
* `load_class_label`
* `unpickle`
* `fuse_single_conv_bn_pair`
* class `DataShaper`
* `run_conv_torch_test`
"""

import csv
import json
import math
import os
import sys

import numpy as np
import torch  # pyright: ignore[reportMissingImports]


class CSVLoggerError(Exception):
    """Raised by CSVLogger for invalid logger state."""


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
            raise CSVLoggerError("Columns already set")
        self.columns = list(columns)
        self.csvwriter.writerow(self.columns)

    def append(self, row):
        if self.columns is None:
            self.set_columns(row.keys())
        assert self.columns is not None
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


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict


def fuse_single_conv_bn_pair(bn_mean, bn_var, bn_wts, bn_bias, conv_wts):
    # https://github.com/ChoiDM/Pytorch_BN_Fold/blob/master/bn_fold.py
    eps = 1e-05
    mu = bn_mean
    var = bn_var
    gamma = bn_wts
    beta = bn_bias
    W = conv_wts

    denom = torch.sqrt(var + eps)

    A = gamma.div(denom)
    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)
    A = A.to(torch.int8)
    W.mul_(A)

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
                            mp = np.zeros((len(mat.shape), 2), dtype=np.int_)
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


def run_conv_torch_test(
    *,
    xclbin_path,
    insts_path,
    golden_model,
    int_inp,
    int_weights,
    out_shape_in_layout,
    out_shape_final,
    out_scale,
    atol,
    kernel_name=None,
    in_layout=("YCXC8", "CYX"),
    wts_layout=("OIYXI8O8", "OIYX"),
    out_reorder=("CDYX", "YCXD"),
    dtype_in=np.int8,
    dtype_wts=np.int8,
    dtype_out=np.int8,
    trace_size=0,
    trace_file=None,
    log_dir=None,
) -> bool:
    """Run a torch conv-style design on the NPU and compare against a golden model.

    Loads the JIT-built ``xclbin``/``insts``, feeds reshaped input + concatenated
    weights, executes via ``DefaultNPURuntime``, reshapes the output back to a
    torch-compatible layout, prints ``"PASS!"`` / ``"Failed."``, and returns
    ``True`` on match / ``False`` on mismatch.  The caller is responsible for
    translating the bool into a process exit code if needed.

    Intended for the conv-style ``test.py`` harnesses under ``programming_examples/ml/``;
    those files collapse to building ``golden_model`` + computing ``int_inp`` and
    ``int_weights``, then delegating here.

    Args:
        xclbin_path, insts_path: paths from the Makefile (``--xclbin`` / ``--instr``).
        golden_model: ``torch.nn.Module`` instance with weights already loaded;
            called as ``golden_model(int_inp)`` to produce the reference output.
        int_inp: ``torch.Tensor`` with shape ``(1, ci, h, w)`` of integer-valued
            ``FloatTensor`` values (the conv harnesses store ints in float
            tensors for torch compatibility).
        int_weights: list of ``torch.Tensor`` weights (one per conv) — each
            reshaped via ``DataShaper(wts_layout)`` and concatenated into a
            single flat ``int_wts`` buffer for the NPU.
        out_shape_in_layout: shape used to reshape the raw NPU output before
            the AIE→torch reorder (e.g. ``(h, co8, w, 8)``).
        out_shape_final: shape of the torch-ready output after the reorder
            (e.g. ``(co, h, w)``).
        out_scale: float multiplier applied to the AIE int output before comparison.
        atol: ``np.allclose`` absolute tolerance.
        kernel_name: name passed to :class:`NPUKernel`; default ``None`` lets
            the runtime pick the first kernel in the xclbin.
        in_layout, wts_layout, out_reorder: ``(order, defOrder)`` token pairs
            passed positionally to :meth:`DataShaper.reorder_mat`.  Defaults
            match the standard conv2d harness shapes (``YCXC8 / CYX``,
            ``OIYXI8O8 / OIYX``, ``CDYX / YCXD``).
        dtype_in, dtype_wts, dtype_out: numpy dtypes for the NPU buffers.
            Default int8 / int8 / int8.
        trace_size: when >0, enables the trace shim and writes a trace file.
            Default 0 (off).
        trace_file: path to write the trace text; required when ``trace_size > 0``.
        log_dir: when set, writes ``before_ifm.txt`` / ``after_ifm.txt`` /
            ``weights.txt`` / ``after_ofm.txt`` debug dumps to this directory.
            Default ``None`` (no dumps).
    """
    import aie.iron as iron
    from aie.utils import HostRuntime, NPUKernel, DefaultNPURuntime, TraceConfig

    dtype_in = np.dtype(dtype_in)
    dtype_wts = np.dtype(dtype_wts)
    dtype_out = np.dtype(dtype_out)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    golden_model.eval()
    golden_output = golden_model(int_inp)

    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    ifm_mem_fmt = ds.reorder_mat(before_input, *in_layout)

    reordered = [
        ds.reorder_mat(w.data.numpy().astype(dtype_wts), *wts_layout)
        for w in int_weights
    ]
    total_wts = np.concatenate(reordered, axis=None)

    if log_dir is not None:
        before_input.tofile(
            os.path.join(log_dir, "before_ifm.txt"), sep=",", format="%d"
        )
        ifm_mem_fmt.tofile(os.path.join(log_dir, "after_ifm.txt"), sep=",", format="%d")
        total_wts.tofile(os.path.join(log_dir, "weights.txt"), sep=",", format="%d")

    in1 = iron.tensor(ifm_mem_fmt, dtype=dtype_in)
    in2 = iron.tensor(total_wts, dtype=dtype_wts)
    out_size = int(np.prod(out_shape_in_layout) * dtype_out.itemsize)
    out = iron.zeros(out_size, dtype=dtype_out)
    buffers = [in1, in2, out]

    trace_config = None
    if trace_size > 0:
        if trace_file is None:
            raise ValueError("trace_file is required when trace_size > 0")
        trace_config = TraceConfig(
            trace_size=trace_size,
            trace_file=trace_file,
            ddr_id=-1,
            enable_ctrl_pkts=False,
            last_tensor_shape=out.shape,
            last_tensor_dtype=out.dtype,
        )
        HostRuntime.prepare_args_for_trace(buffers, trace_config)

    npu_kernel_kwargs = {} if kernel_name is None else {"kernel_name": kernel_name}
    npu_kernel = NPUKernel(xclbin_path, insts_path, **npu_kernel_kwargs)
    if DefaultNPURuntime is None:
        raise RuntimeError("No default NPU runtime available (is XRT installed?)")
    kernel_handle = DefaultNPURuntime.load(npu_kernel)
    ret = DefaultNPURuntime.run(kernel_handle, buffers)

    if trace_config is not None:
        trace_buffer, _ = HostRuntime.extract_trace_from_args(buffers, trace_config)
        trace_config.write_trace(trace_buffer.view(np.uint32))

    out_tensor = buffers[-1]
    if not isinstance(out_tensor, np.ndarray):
        out_tensor = out_tensor.numpy()
    data_buffer = out_tensor * out_scale

    temp_out = data_buffer.reshape(out_shape_in_layout)
    temp_out = ds.reorder_mat(temp_out, *out_reorder)
    ofm_mem_fmt = temp_out.reshape(out_shape_final)
    if log_dir is not None:
        ofm_mem_fmt.tofile(os.path.join(log_dir, "after_ofm.txt"), sep=",", format="%d")
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)

    # npu_time is in nanoseconds in the runtime; harnesses divide by 1000
    # to print microseconds.
    print(f"\nAvg NPU time: {int(ret.npu_time / 1000)}us.")

    if np.allclose(
        ofm_mem_fmt_out.detach().numpy(),
        golden_output.detach().numpy(),
        rtol=0,
        atol=atol,
    ):
        print("\nPASS!\n")
        return True
    print("\nFailed.\n")
    return False
