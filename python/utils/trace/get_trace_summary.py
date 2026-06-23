#!/usr/bin/env python3
# (c) Copyright 2026 Advanced Micro Devices, Inc.
import argparse
import logging
import sys
from aie.utils.trace.utils import print_cycles_summary

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Trace file", required=True)
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    opts = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
    print_cycles_summary(opts.input)
