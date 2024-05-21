#!/usr/bin/env python3
# matrix_multiplication/plot_sweep.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import argparse
import csv
import os.path
import re
import numpy as np
import matplotlib.pyplot as plt
import sys


def n_bytes(M, K, N):
    dtype_sz = 2
    return (M * K + K * N + M * N) * dtype_sz


def macs(M, K, N):
    return M * K * N


def flops(M, K, N):
    # Each MAC consists of a multiply and an add, i.e. 2 flops
    return 2 * macs(M, K, N)


def arithmetic_intensity(xs):
    M, K, N = xs
    return macs(M, K, N) / n_bytes(M, K, N)


def arithmetic_intensity_flops(xs):
    M, K, N = xs
    return flops(M, K, N) / n_bytes(M, K, N)


def macs_per_s(ys):
    M, K, N, *ts = ys
    return macs(M, K, N) / (np.mean(ts) / 1e6)


def gflops_per_s(ys):
    M, K, N, *ts = ys
    return flops(M, K, N) / 1e9 / (np.mean(ts) / 1e6)


def tflops_per_s(ys):
    M, K, N, *ts = ys
    return flops(M, K, N) / 1e12 / (np.mean(ts) / 1e6)


def throughput(ys):
    M, K, N, *ts = ys
    dtype_size = 2
    n_bytes = (M * K + K * N + M * N) * dtype_size
    return float(n_bytes) / (np.mean(ts) / 1e6)


def efficiency(ys):
    return tflops_per_s(ys) / 4.096 * 100


transforms = {
    "prod": np.prod,
    "sum": sum,
    "mean": np.mean,
    "intens": arithmetic_intensity,
    "intens_f": arithmetic_intensity_flops,
    "macs": macs_per_s,
    "gflops": gflops_per_s,
    "tflops": tflops_per_s,
    "thru": throughput,
    "eff": efficiency,
}


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", "-i", type=argparse.FileType("r"), required=True)
    argparser.add_argument("--output", "-o", type=argparse.FileType("wb"), default=None)
    argparser.add_argument("--outputfmt", choices=["png", "pdf"], default="pdf")
    argparser.add_argument(
        "--title", "-t", type=str, default="Matrix Multiplication Sweep"
    )
    argparser.add_argument("--xnames", "-x", type=str, action="append", default=[])
    argparser.add_argument("--xtrans", choices=transforms, default="prod")
    argparser.add_argument("--xlabel", type=str)
    argparser.add_argument("--ynames", "-y", type=str, action="append", default=[])
    argparser.add_argument("--ytrans", choices=transforms, default="mean")
    argparser.add_argument("--ylabel", type=str)
    argparser.add_argument("--filter", type=str, action="append", default=[])
    argparser.add_argument("--xlog", action="store_true", default=False)
    argparser.add_argument("--ylog", action="store_true", default=False)
    args = argparser.parse_args()
    if not args.xnames:
        args.xnames = ["M", "K", "N"]
    if args.xlabel is None:
        if args.xtrans == "intens":
            args.xlabel = "Arithmetic Intensity [MAC/byte]"
        elif args.xtrans == "intens_f":
            args.xlabel = "Arithmetic Intensity [FLOP/byte]"
        elif args.xtrans == "prod":
            args.xlabel = "*".join(args.xnames)
    if args.ylabel is None:
        if args.ytrans == "mean":
            args.ylabel = "Mean Runtime [us]"
        elif args.ytrans == "macs":
            args.ylabel = "MACs/s"
        elif args.ytrans == "gflops":
            args.ylabel = "GFLOP/s"
        elif args.ytrans == "tflops":
            args.ylabel = "TFLOP/s"
        elif args.ytrans == "thru":
            args.ylabel = "Throughput [bytes/s]"
        elif args.ytrans == "eff":
            args.ylabel = "Percent Throughput Efficiency [achieved/peak]"
    if args.output is None:
        args.output = "{0}.{1}".format(
            os.path.basename(args.input.name), args.outputfmt
        )
    if not args.ynames:
        header = args.input.readline()
        args.input.seek(0)
        iteration_ys = [
            "It{}".format(m.group(1)) for m in re.finditer(r"It(\d+)", header)
        ]
        if args.ytrans == "mean":
            args.ynames = iteration_ys
        elif args.ytrans in {"macs", "gflops", "tflops", "thru", "eff"}:
            args.ynames = ["M", "K", "N"] + iteration_ys
    args.xtrans = transforms[args.xtrans]
    args.ytrans = transforms[args.ytrans]
    return args


def get_plot_values(csv_file, x_names, y_names, x_trans, y_trans, filters=[]):
    xs = []
    ys = []
    csvreader = csv.DictReader(csv_file, skipinitialspace=True)
    errors = []
    for i, row in enumerate(csvreader):
        # Strip whitespace
        row = {
            k.strip(): v.strip() if v is not None else None
            for k, v in row.items()
            if k is not None
        }
        if not all(eval(f, row) for f in filters):
            continue
        try:
            x_pieces = [float(row[x_name]) for x_name in x_names]
            y_pieces = [float(row[y_name]) for y_name in y_names]
        except (TypeError, ValueError):
            errors.append(i)
            continue
        x = x_trans(x_pieces)
        y = y_trans(y_pieces)
        xs.append(x)
        ys.append(y)
    if errors:
        sys.stderr.write(
            "Warning: Invalid data in row(s) {}.\n".format(", ".join(map(str, errors)))
        )
    print("{}: {} rows".format(csv_file.name, len(ys)))
    return xs, ys


def plot(ax, xs, ys, title, xlabel, ylabel, xlog=False, ylog=False):
    ax.scatter(xs, ys, marker=".")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")


def plot_max(ax, xs, ys, xtrans, ytrans):
    """Draw the roofline"""

    bandwidth = 2e9  # 2 GB/s peak memory bandwidth per channel
    max_flops = 1e12  # 1 TFLOP/s / core for bf16
    n_channels = 8
    n_cores = 4

    max_flops *= n_cores
    bandwidth *= n_channels

    slope = bandwidth  # [byte/s]
    max_y = max_flops  # [FLOP/s]

    # In bandwidth-bound area, # ops per second is equal to # bytes per second movable
    # times arithmetic intensity (ops per each received byte), because processing is
    # faster than data movement.

    if xtrans == transforms["intens"]:
        # 1 [FLOP/byte] = 2 [MAC/byte]
        slope *= 2
    elif xtrans == transforms["intens_f"]:
        pass
    else:
        # Not a roofline plot
        return

    if ytrans == transforms["macs"]:
        # 1 [FLOP] = 2 [MACs]
        max_y /= 2
        slope /= 2
    elif ytrans == transforms["gflops"]:
        max_y /= 1e9
        slope /= 1e9
    elif ytrans == transforms["tflops"]:
        max_y /= 1e12
        slope /= 1e12
    else:
        # Not a roofline plot
        return

    max_x = max(xs)
    # ridge point x is where max_y == slope*x
    ridge_point_x = min(max_y / slope, max_x)
    ridge_point_y = min(max_y, ridge_point_x * slope)
    max_y = min(max_y, max_x * slope)

    ax.plot([0, ridge_point_x, max_x], [0, ridge_point_y, max_y])


def main():
    args = get_args()
    xs, ys = get_plot_values(
        args.input, args.xnames, args.ynames, args.xtrans, args.ytrans, args.filter
    )
    fig, ax = plt.subplots()
    plot(ax, xs, ys, args.title, args.xlabel, args.ylabel, args.xlog, args.ylog)
    plot_max(ax, xs, ys, args.xtrans, args.ytrans)
    plt.savefig(args.output, format=args.outputfmt)


if __name__ == "__main__":
    main()
