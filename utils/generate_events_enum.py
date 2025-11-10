#!/usr/bin/env python3

"""
Takes the xaie_events_*.h header files from aie-rt and generates:
1. A JSON database containing all events for all architectures
2. Per-architecture Python files with enums

The generated files are used by trace utilities for event decoding and analysis.
"""

import sys, re, argparse, collections, json, glob, os
from pathlib import Path

py_template = """# Enumeration of {arch_name} trace events
# Automatically generated from utils/generate_events_enum.py

from enum import Enum


class CoreEvent(Enum):
{core_items}


class MemEvent(Enum):
{mem_items}


class ShimTileEvent(Enum):
{pl_items}


class MemTileEvent(Enum):
{mem_tile_items}
{uc_section}"""

# Architecture configurations
ARCH_CONFIGS = {
    "xaie_events_aie.h": {
        "name": "aie",
        "display_name": "AIE",
        "prefix": "XAIE_EVENTS_",
        "python_name": "aie",  # Python module name
    },
    "xaie_events_aieml.h": {
        "name": "aieml",
        "display_name": "AIEML (AIE2)",
        "prefix": "XAIEML_EVENTS_",
        "python_name": "aie2",  # Python module name
    },
    "xaie_events_aie2p.h": {
        "name": "aie2p",
        "display_name": "AIE2P",
        "prefix": "XAIE2P_EVENTS_",
        "python_name": "aie2p",  # Python module name
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
        sys.stderr.write(f"Warning: Duplicate event number {num} for {name}\n")
        return
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


def write_enum_items(dict):
    """Format dictionary as Python enum items, filling gaps with reserved placeholders."""
    if not dict:
        return "    pass  # No events defined"

    # Fill gaps with rsvd_XX placeholders
    if dict:
        min_val = min(dict.keys())
        max_val = max(dict.keys())
        filled_dict = {}
        for val in range(min_val, max_val + 1):
            if val in dict:
                filled_dict[val] = dict[val]
            else:
                filled_dict[val] = f"rsvd_{val}"
        dict = filled_dict

    return "\n".join("    {} = {}".format(name, num) for num, name in dict.items())


def write_python_file(output_path, arch_name, events):
    """Write a Python enum file for one architecture."""
    core_str = write_enum_items(events["core"])
    mem_str = write_enum_items(events["mem"])
    pl_str = write_enum_items(events["pl"])
    mem_tile_str = write_enum_items(events["mem_tile"])
    uc_str = write_enum_items(events["uc"])

    # Only include UCEvent if there are events defined
    if events["uc"]:
        uc_section = f"\n\nclass UCEvent(Enum):\n{uc_str}\n"
    else:
        uc_section = ""

    content = py_template.format(
        arch_name=arch_name,
        core_items=core_str,
        mem_items=mem_str,
        pl_items=pl_str,
        mem_tile_items=mem_tile_str,
        uc_section=uc_section,
    )

    with open(output_path, "w") as f:
        f.write(content)


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
        description="Generate Python enums and JSON database from AIE event headers"
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
    argparser.add_argument(
        "--json",
        default="events_database.json",
        help="JSON database filename (default: events_database.json)",
    )
    argparser.add_argument(
        "--python-prefix",
        default="trace_events_enum_",
        help="Prefix for Python enum files (default: trace_events_enum_)",
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

        # Write per-architecture Python file using python_name
        python_name = arch_config.get("python_name", arch_config["name"])
        py_filename = f"{args.python_prefix}{python_name}.py"
        py_output_path = output_dir / py_filename
        write_python_file(py_output_path, arch_config["display_name"], events)
        sys.stderr.write(f"  Generated {py_output_path}\n")

    # Write unified JSON database
    json_output_path = output_dir / args.json
    write_json_database(json_output_path, all_events)
    sys.stderr.write(f"Generated JSON database: {json_output_path}\n")

    sys.stderr.write(f"\nSuccessfully processed {len(all_events)} architecture(s)\n")


if __name__ == "__main__":
    main()
