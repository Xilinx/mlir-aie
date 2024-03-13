#!/usr/bin/env python3

import argparse
import csv
import os.path
import re
import numpy as np
import matplotlib.pyplot as plt
import sys


def arithmetic_intensity(xs):
    M, K, N = xs
    return M * N * K / (M * K + N * K + M * N)


def macs_per_s(ys):
    M, K, N, *ts = ys
    n_macs = M * K * N
    return n_macs / (np.mean(ts) / 1e6)


def gflops(ys):
    macs = macs_per_s(ys)
    return 2 * macs / 1e9


def tflops(ys):
    macs = macs_per_s(ys)
    return 2 * macs / 1e12


def throughput(ys):
    M, K, N, *ts = ys
    dtype_size = 2
    n_bytes = (M * K + K * N + M * N) * dtype_size
    return float(n_bytes) / (np.mean(ts) / 1e6)


transforms = {
    "prod": np.prod,
    "sum": sum,
    "mean": np.mean,
    "intens": arithmetic_intensity,
    "macs": macs_per_s,
    "gflops": gflops,
    "tflops": tflops,
    "thru": throughput,
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
    args = argparser.parse_args()
    if not args.xnames:
        args.xnames = ["M", "K", "N"]
    if args.xlabel is None:
        if args.xtrans == "intens":
            args.xlabel = "Arithmetic Intensity M*K*N/(M*K+N*K+M*N)"
        elif args.xtrans == "prod":
            args.xlabel = "*".join(args.xnames)
    if args.ylabel is None:
        if args.ytrans == "mean":
            args.ylabel = "Mean Runtime [us]"
        elif args.ytrans == "macs":
            args.ylabel = "MACs/s"
        elif args.ytrans == "gflops":
            args.ylabel = "GFLOPS"
        elif args.ytrans == "tflops":
            args.ylabel = "TFLOPS"
        elif args.ytrans == "thru":
            args.ylabel = "Throughput [bytes/s]"
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
        elif args.ytrans in {"macs", "gflops", "tflops", "thru"}:
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
    print(
        "{}: {} rows, {} data points".format(
            csv_file.name, len(ys), len(ys) * len(y_names)
        )
    )
    return xs, ys


def plot(ax, xs, ys, title, xlabel, ylabel):
    ax.scatter(xs, ys, marker=".")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    args = get_args()
    xs, ys = get_plot_values(
        args.input, args.xnames, args.ynames, args.xtrans, args.ytrans, args.filter
    )
    fig, ax = plt.subplots()
    plot(ax, xs, ys, args.title, args.xlabel, args.ylabel)
    plt.savefig(args.output, format=args.outputfmt)


if __name__ == "__main__":
    main()
