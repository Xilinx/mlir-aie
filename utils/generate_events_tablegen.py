#!/usr/bin/env python3

"""
Takes the xaie_events_*.h header files from aie-rt and generates:
1. TableGen (.td) files containing enum definitions for MLIR AIE dialect

The generated files are used by the AIE dialect for event enums.
"""

import sys, re, argparse, collections, os
from pathlib import Path

td_template = """//===- AIEEvents{suffix}.td.inc - AIE Event Enums -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Automatically generated from utils/generate_events_tablegen.py
// DO NOT EDIT
//
//===----------------------------------------------------------------------===//

// {arch_display_name} Core Module Events
def CoreEvent{suffix}: I32EnumAttr<"CoreEvent{suffix}", "Core module event enumeration for {arch_display_name}",
  [
{core_items}
  ]> {{
  let cppNamespace = "xilinx::AIE";
}}

// {arch_display_name} Memory Module Events
def MemEvent{suffix}: I32EnumAttr<"MemEvent{suffix}", "Memory module event enumeration for {arch_display_name}",
  [
{mem_items}
  ]> {{
  let cppNamespace = "xilinx::AIE";
}}

// {arch_display_name} Shim Tile Events
def ShimTileEvent{suffix}: I32EnumAttr<"ShimTileEvent{suffix}", "Shim tile event enumeration for {arch_display_name}",
  [
{pl_items}
  ]> {{
  let cppNamespace = "xilinx::AIE";
}}

// {arch_display_name} Memory Tile Events
def MemTileEvent{suffix}: I32EnumAttr<"MemTileEvent{suffix}", "Memory tile event enumeration for {arch_display_name}",
  [
{mem_tile_items}
  ]> {{
  let cppNamespace = "xilinx::AIE";
}}
{uc_section}"""

# Architecture configurations
ARCH_CONFIGS = {
    "xaie_events_aie.h": {
        "name": "aie",
        "display_name": "AIE",
        "prefix": "XAIE_EVENTS_",
        "suffix": "",  # No suffix for AIE1
    },
    "xaie_events_aieml.h": {
        "name": "aieml",
        "display_name": "AIE2",
        "prefix": "XAIEML_EVENTS_",
        "suffix": "AIE2",
    },
    "xaie_events_aie2p.h": {
        "name": "aie2p",
        "display_name": "AIE2P",
        "prefix": "XAIE2P_EVENTS_",
        "suffix": "AIE2P",
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


def write_enum_items_tablegen(dict):
    """Format dictionary as TableGen enum cases."""
    if not dict:
        return "    // No events defined"

    items = []
    for num, name in dict.items():
        items.append(f'    I32EnumAttrCase<"{name}", {num}>')
    
    return ",\n".join(items)


def write_tablegen_file(output_path, arch_config, events):
    """Write a TableGen file for one architecture."""
    suffix = arch_config["suffix"]
    arch_display_name = arch_config["display_name"]
    
    core_str = write_enum_items_tablegen(events["core"])
    mem_str = write_enum_items_tablegen(events["mem"])
    pl_str = write_enum_items_tablegen(events["pl"])
    mem_tile_str = write_enum_items_tablegen(events["mem_tile"])
    uc_str = write_enum_items_tablegen(events["uc"])

    # Only include UCEvent if there are events defined
    if events["uc"]:
        uc_section = f"""
// {arch_display_name} UC Events
def UCEvent{suffix}: I32EnumAttr<"UCEvent{suffix}", "UC event enumeration for {arch_display_name}",
  [
{uc_str}
  ]> {{
  let cppNamespace = "xilinx::AIE";
}}
"""
    else:
        uc_section = ""

    content = td_template.format(
        suffix=suffix,
        arch_display_name=arch_display_name,
        core_items=core_str,
        mem_items=mem_str,
        pl_items=pl_str,
        mem_tile_items=mem_tile_str,
        uc_section=uc_section,
    )

    with open(output_path, "w") as f:
        f.write(content)


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
        description="Generate TableGen enums from AIE event headers"
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
        help="Output directory for generated files (default: current directory)",
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

        # Write TableGen file
        suffix = arch_config["suffix"]
        td_filename = f"AIEEvents{suffix}.td.inc"
        td_output_path = output_dir / td_filename
        write_tablegen_file(td_output_path, arch_config, events)
        sys.stderr.write(f"  Generated {td_output_path}\n")

    sys.stderr.write(f"\nSuccessfully processed {len(input_files)} architecture(s)\n")


if __name__ == "__main__":
    main()
