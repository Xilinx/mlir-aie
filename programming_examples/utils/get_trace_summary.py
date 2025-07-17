#!/usr/bin/env python3
import json
import argparse
import sys
import re
import trace_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Trace file", required=True)
    # parser.add_argument("--mlir", help="mlir source file", required=True)
    # parser.add_argument(
    #    "--colshift", help="column shift adjustment to source mlir", required=False
    # )
    parser.add_argument("--debug", help="debug mode", required=False)
    # TODO tracelabels removed since we can have multiple sets of labels for each pkt_type & loc combination
    # parser.add_argument('--tracelabels',
    #         nargs='+',
    #         help='Labels for traces', required=False)
    return parser.parse_args(sys.argv[1:])


opts = parse_args()
cycles = trace_utils.get_cycles_summary(opts.input)

# print(cycles)
for i in range(len(cycles)):
    print(cycles[i][0])
    runs = len(cycles[i]) - 1
    print("Total number of full kernel invocations is " + str(runs))
    if runs > 0:
        print(
            "First/Min/Avg/Max cycles is "
            + str(cycles[i][1])
            + "/ "
            + str(min(cycles[i][1:]))
            + "/ "
            + str(sum(cycles[i][1:]) / (len(cycles[i]) - 1))
            + "/ "
            + str(max(cycles[i][1:]))
        )
