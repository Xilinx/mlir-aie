#!/usr/bin/env python3
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
"""
Takes the xaie_events_*.h header files from aie-rt and generates a JSON
database containing all events for all architectures.

The generated JSON file is used by the C++ register database
(AIERegisterDatabase.cpp) for event name-to-number lookups.

Usage (offline, run manually when aie-rt headers change):
    python utils/generate_events_json.py -o lib/Dialect/AIE/Util/
"""

import sys, re, argparse, collections, json, os
from pathlib import Path

# Architecture configurations
ARCH_CONFIGS = {
    "xaie_events_aie.h": {
        "name": "aie",
        "display_name": "AIE",
        "prefix": "XAIE_EVENTS_",
    },
    "xaie_events_aieml.h": {
        "name": "aie2",
        "display_name": "AIE2",
        "prefix": "XAIEML_EVENTS_",
    },
    "xaie_events_aie2p.h": {
        "name": "aie2p",
        "display_name": "AIE2P",
        "prefix": "XAIE2P_EVENTS_",
    },
}


def get_regex_patterns(prefix):
    """Generate regex patterns for a given architecture prefix."""
    return {
        "core": re.compile(rf"^\s*#define\s+{prefix}CORE_([a-zA-Z0-9_]+)\s+(\d+)U\s*$"),
        "mem": re.compile(
            rf"^\s*#define\s+{prefix}MEM_(?!TILE)([a-zA-Z0-9_]+)\s+(\d+)U\s*$"
        ),
        "pl": re.compile(rf"^\s*#define\s+{prefix}PL_([a-zA-Z0-9_]+)\s+(\d+)U\s*$"),
        "mem_tile": re.compile(
            rf"^\s*#define\s+{prefix}MEM_TILE_([a-zA-Z0-9_]+)\s+(\d+)U\s*$"
        ),
        "uc": re.compile(rf"^\s*#define\s+{prefix}UC_([a-zA-Z0-9_]+)\s+(\d+)U\s*$"),
    }


def parse_event_declaration(regex, dict, line):
    """Parse a single event declaration line."""
    match = regex.match(line)
    if not match:
        return
    name, num = match.group(1), int(match.group(2))
    if num in dict:
        sys.stderr.write(
            f"Error: Duplicate event number {num} for {name} (already used by {dict[num]})\n"
        )
        sys.exit(1)
    dict[num] = name


def parse_events_file(filepath, arch_config):
    """Parse an events header file and return events by module."""
    patterns = get_regex_patterns(arch_config["prefix"])
    core_events = collections.OrderedDict()
    mem_events = collections.OrderedDict()
    mem_tile_events = collections.OrderedDict()
    pl_events = collections.OrderedDict()
    uc_events = collections.OrderedDict()

    with open(filepath, "r") as f:
        for line in f:
            parse_event_declaration(patterns["core"], core_events, line)
            parse_event_declaration(patterns["mem"], mem_events, line)
            parse_event_declaration(patterns["pl"], pl_events, line)
            parse_event_declaration(patterns["mem_tile"], mem_tile_events, line)
            parse_event_declaration(patterns["uc"], uc_events, line)

    return {
        "core": core_events,
        "mem": mem_events,
        "pl": pl_events,
        "mem_tile": mem_tile_events,
        "uc": uc_events,
    }


def events_dict_to_list(events_dict):
    """Convert OrderedDict to list of {number, name} objects for JSON."""
    return [{"number": num, "name": name} for num, name in events_dict.items()]


def write_json_database(output_path, all_events):
    """Write unified JSON database with all architectures."""
    json_data = {}

    for filename, data in all_events.items():
        arch_config = ARCH_CONFIGS[filename]
        arch_name = arch_config["name"]

        json_data[arch_name] = {
            "display_name": arch_config["display_name"],
            "prefix": arch_config["prefix"],
            "modules": {
                "core": events_dict_to_list(data["core"]),
                "memory": events_dict_to_list(data["mem"]),
                "pl": events_dict_to_list(data["pl"]),
                "mem_tile": events_dict_to_list(data["mem_tile"]),
                "uc": events_dict_to_list(data["uc"]),
            },
        }

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)


def find_default_input_files():
    """Find all event header files in the default location."""
    default_path = (
        Path(__file__).parent.parent
        / "third_party"
        / "aie-rt"
        / "driver"
        / "src"
        / "events"
    )
    if not default_path.exists():
        return []

    files = []
    for filename in ARCH_CONFIGS.keys():
        filepath = default_path / filename
        if filepath.exists():
            files.append(str(filepath))

    return files


def main():
    argparser = argparse.ArgumentParser(
        description="Generate JSON event database from AIE event headers"
    )
    argparser.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Input header files (default: all files in third_party/aie-rt/driver/src/events)",
    )
    argparser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Output directory for generated file (default: current directory)",
    )
    argparser.add_argument(
        "--json",
        default="events_database.json",
        help="JSON database filename (default: events_database.json)",
    )
    args = argparser.parse_args()

    # Determine input files
    input_files = args.input if args.input else find_default_input_files()

    if not input_files:
        sys.stderr.write(
            "Error: No input files specified and default location not found.\n"
            "Please specify input files with -i or ensure third_party/aie-rt exists.\n"
        )
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all input files
    all_events = {}
    for filepath in input_files:
        filename = os.path.basename(filepath)
        if filename not in ARCH_CONFIGS:
            sys.stderr.write(
                f"Warning: Unknown architecture file {filename}, skipping\n"
            )
            continue

        arch_config = ARCH_CONFIGS[filename]
        sys.stderr.write(f"Processing {arch_config['display_name']} from {filepath}\n")

        events = parse_events_file(filepath, arch_config)
        all_events[filename] = events

    # Write unified JSON database
    json_output_path = output_dir / args.json
    write_json_database(json_output_path, all_events)
    sys.stderr.write(f"Generated JSON database: {json_output_path}\n")

    sys.stderr.write(f"\nSuccessfully processed {len(all_events)} architecture(s)\n")


if __name__ == "__main__":
    main()
