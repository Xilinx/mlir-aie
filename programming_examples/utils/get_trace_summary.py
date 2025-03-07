#!/usr/bin/env python3
import json
import argparse
import sys
import re
import trace_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Trace file", required=True)
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
cycles = trace_utils.get_cycles_summary(opts.filename)

print("Total number of full kernel invocations is " + str(len(cycles)))
print(
    "First/Min/Avg/Max is "
    + str(cycles[0])
    + "/ "
    + str(min(cycles))
    + "/ "
    + str(sum(cycles) / len(cycles))
    + "/ "
    + str(max(cycles))
)
