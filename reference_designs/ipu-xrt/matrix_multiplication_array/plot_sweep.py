#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

def arithmetic_intensity(xs):
    M, K, N = xs
    return M*N*K / (M*K + N*K + M*N)

transforms = {
    'prod': np.prod,
    'sum': sum,
    'mean': np.mean,
    'intens': arithmetic_intensity
}

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', '-i', type=argparse.FileType('r'), required=True)
    argparser.add_argument('--output', '-o', type=argparse.FileType('wb'), default="sweep.pdf")
    argparser.add_argument('--outputfmt', choices=['png', 'pdf'], default="pdf")
    argparser.add_argument('--title', '-t', type=str, default="Matrix Multiplication Sweep")
    argparser.add_argument('--xnames', '-x', type=str, action='append', default=['M', 'K', 'N'])
    argparser.add_argument('--xtrans', choices=transforms, default='prod')
    argparser.add_argument('--xlabel', type=str)
    argparser.add_argument('--ynames', '-y', type=str, action='append', default=['It{}'.format(i) for i in range(1, 11)])
    argparser.add_argument('--ytrans', choices=transforms, default='mean')
    argparser.add_argument('--ylabel', type=str)
    args = argparser.parse_args()
    if args.xlabel is None:
        if args.xtrans == 'intens':
            args.xlabel = "Arithmetic Intensity M*K*N/(M*K+N*K+M*N)"
        elif args.xtrans == 'prod':
            args.xlabel = "M*K*N"
    if args.ylabel is None:
        if args.ytrans == 'mean':
            args.ylabel = "Mean Runtime"
    args.xtrans = transforms[args.xtrans]
    args.ytrans = transforms[args.ytrans]
    return args

def get_plot_values(csv_file, x_names, y_names, x_trans, y_trans):
    xs = []
    ys = []
    csvreader = csv.DictReader(csv_file, skipinitialspace=True)
    errors = []
    for i, row in enumerate(csvreader):
        # Strip whitespace
        row = {k.strip() : v.strip() if v is not None else None for k, v in row.items() if k is not None}
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
        sys.stderr.write("Warning: Invalid data in row(s) {}.\n".format(", ".join(map(str, errors))))
    print("{}: {} rows, {} data points".format(csv_file.name, len(ys), len(ys)*len(y_names)))
    return xs, ys

def plot(ax, xs, ys, title, xlabel, ylabel):
    ax.scatter(xs, ys, marker='.')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def main():
    args = get_args()
    xs, ys = get_plot_values(args.input, args.xnames, args.ynames, args.xtrans, args.ytrans)
    fig, ax = plt.subplots()
    plot(ax, xs, ys, args.title, args.xlabel, args.ylabel)
    plt.savefig(args.output, format=args.outputfmt)

if __name__ == "__main__":
    main()