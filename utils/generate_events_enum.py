#!/usr/bin/env python3

"""
Takes the xaie_events_aie.h header file from aie-rt and generates an
importable Python file containing an enum of all events.

The generated enum is included in python/utils/trace_events_enum.py and
used by the trace utilities in python/utils/trace.py and in
programming_examples/utils/parse_trace.py
"""

import sys, re, argparse, collections

template = """# Enumeration of AIE2 trace events
# Automatically generated from utils/generate_events_enum.py

from enum import Enum


class CoreEvent(Enum):
{core_items}


class MemEvent(Enum):
{mem_items}


class PLEvent(Enum):
{pl_items}


class MemTileEvent(Enum):
{mem_tile_items}
"""

core_regex = r"^\s*#define\s+XAIEML_EVENTS_CORE_([a-zA-Z0-9_]+)\s+(\d+)U\s*$"
mem_regex = r"^\s*#define\s+XAIEML_EVENTS_MEM_(?!TILE)([a-zA-Z0-9_]+)\s+(\d+)U\s*$"
pl_regex = r"^\s*#define\s+XAIEML_EVENTS_PL_([a-zA-Z0-9_]+)\s+(\d+)U\s*$"
mem_tile_regex = r"^\s*#define\s+XAIEML_EVENTS_MEM_TILE_([a-zA-Z0-9_]+)\s+(\d+)U\s*$"


def parse_event_declaration(regex, dict, line):
    match = re.match(regex, line)
    if not match:
        return
    name, num = match.group(1), int(match.group(2))
    if match.group(2) in dict:
        sys.stderr.write("Duplicate event number {}.".format(num))
        return
    dict[num] = name


def write_enum_items(dict):
    return "\n".join("    {} = {}".format(name, num) for num, name in dict.items())


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", type=argparse.FileType("r"), default=sys.stdin)
    argparser.add_argument("-o", type=argparse.FileType("w"), default=sys.stdout)
    args = argparser.parse_args()

    lines = args.i.readlines()
    core_events = collections.OrderedDict()
    mem_events = collections.OrderedDict()
    mem_tile_events = collections.OrderedDict()
    pl_events = collections.OrderedDict()
    for line in lines:
        parse_event_declaration(core_regex, core_events, line)
        parse_event_declaration(mem_regex, mem_events, line)
        parse_event_declaration(pl_regex, pl_events, line)
        parse_event_declaration(mem_tile_regex, mem_tile_events, line)

    core_str = write_enum_items(core_events)
    mem_str = write_enum_items(mem_events)
    pl_str = write_enum_items(pl_events)
    mem_tile_str = write_enum_items(mem_tile_events)

    args.o.write(
        template.format(
            core_items=core_str,
            mem_items=mem_str,
            pl_items=pl_str,
            mem_tile_items=mem_tile_str,
        )
    )


if __name__ == "__main__":
    main()
