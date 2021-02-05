import argparse
import os
import shlex
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog='aiecc')
    parser.add_argument('filename',
            metavar="file",
            help='File to compile')
    parser.add_argument('--xbridge',
            dest="xbridge",
            default=True,
            action='store_true',
            help='Link using xbridge')
    parser.add_argument('--no-xbridge',
            dest="xbridge",
            action='store_false',
            help='Link using xbridge')

    opts = parser.parse_args(sys.argv[1:])

    return opts


def _positive_int(arg):
    return _int(arg, 'positive', lambda i: i > 0)


def _non_negative_int(arg):
    return _int(arg, 'non-negative', lambda i: i >= 0)


def _int(arg, kind, pred):
    desc = "requires {} integer, but found '{}'"
    try:
        i = int(arg)
    except ValueError:
        raise _error(desc, kind, arg)
    if not pred(i):
        raise _error(desc, kind, arg)
    return i


def _case_insensitive_regex(arg):
    import re
    try:
        return re.compile(arg, re.IGNORECASE)
    except re.error as reason:
        raise _error("invalid regular expression: '{}', {}", arg, reason)


def _error(desc, *args):
    msg = desc.format(*args)
    return argparse.ArgumentTypeError(msg)
