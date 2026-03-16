#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
#
"""
Parse Local AMD AIE-ML Register Documentation with Bit-Fields

Usage:
    python parse_register_docs.py [--docs-dir /workspace/am025/html] [--output aie_registers.json]
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: BeautifulSoup is required.")
    print("Install with: pip install beautifulsoup4")
    exit(1)

try:
    import lxml
except ImportError:
    print("Error: lxml is required.")
    print("Install with: pip install lxml")
    exit(1)

# Module file mappings
MODULE_FILES = {
    "CORE_MODULE": "mod___core_module.html",
    "MEMORY_MODULE": "mod___memory_module.html",
    "MEMORY_TILE_MODULE": "mod___memory_tile_module.html",
    "SHIM_MODULE": [
        ("NOC_MODULE", "mod___noc_module.html"),
        ("PL_MODULE", "mod___pl_module.html"),
    ],
}


class EnhancedAIEDocsParser:
    """Enhanced parser for AMD AIE-ML HTML documentation with bit-field extraction"""

    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir

    def parse_bit_range(self, bits_str: str) -> List[int]:
        """Parse bit range string into [start, end] list

        Examples:
            "0" -> [0, 0]
            "1" -> [1, 1]
            "14:8" -> [8, 14]
            "31:0" -> [0, 31]
        """
        bits_str = bits_str.strip()

        if ":" in bits_str:
            # Range like "14:8"
            parts = bits_str.split(":")
            high = int(parts[0].strip())
            low = int(parts[1].strip())
            return [low, high]
        else:
            # Single bit like "1"
            bit = int(bits_str)
            return [bit, bit]

    def parse_register_bitfields(
        self, register_name: str, module_name: str
    ) -> List[Dict]:
        """Parse bit-field information from individual register HTML file"""

        # Convert register name to filename format
        # e.g., "Core_Control" -> "core_module___core_control.html"
        module_prefix = module_name.lower()
        reg_filename = f"{module_prefix}___{register_name.lower()}.html"
        filepath = os.path.join(self.docs_dir, reg_filename)

        if not os.path.exists(filepath):
            # File doesn't exist, return empty list
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "lxml")

            # Find the bit-field summary table
            # Look for table after "Bit-Field Summary" heading
            bit_fields = []

            tables = soup.find_all("table")
            for table in tables:
                # Skip tables with class 'noborder'
                if table.get("class") and "noborder" in table.get("class"):
                    continue

                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue

                # Check if this is a bit-field table
                header_row = rows[0]
                headers = [th.get_text().strip() for th in header_row.find_all("th")]

                if not any("Field Name" in h or "Bits" in h for h in headers):
                    continue

                # This is the bit-field table
                for row in rows[1:]:
                    cells = row.find_all("td")

                    if len(cells) >= 5:  # Field Name, Bits, Type, Reset, Description
                        try:
                            field_name = cells[0].get_text().strip()
                            bits_str = cells[1].get_text().strip()
                            field_type = cells[2].get_text().strip()
                            field_reset = cells[3].get_text().strip()
                            field_desc = cells[4].get_text().strip()

                            # Parse bit range
                            bit_range = self.parse_bit_range(bits_str)

                            bit_field = {
                                "name": field_name,
                                "bits": bits_str,
                                "bit_range": bit_range,
                                "type": field_type,
                                "reset": field_reset,
                                "description": field_desc,
                            }

                            bit_fields.append(bit_field)

                        except Exception as e:
                            print(
                                f"Warning: Could not parse bit-field in {reg_filename}: {e}"
                            )
                            continue

            return bit_fields

        except Exception as e:
            print(f"Warning: Could not parse bit-fields for {register_name}: {e}")
            return []

    def parse_module_file(self, module_name: str, filename: str) -> Dict:
        """Parse a module HTML file and extract register information"""
        filepath = os.path.join(self.docs_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            return {
                "error": f"File not found: {filepath}",
                "base_address_info": {},
                "registers": [],
            }

        print(f"Parsing {module_name} from {filename}...")

        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "lxml")

        # Extract base address information
        base_info = self.extract_base_address_info(soup)

        # Extract register table
        registers = self.extract_register_table(soup, module_name)

        print(f"  Found {len(registers)} registers in {module_name}")

        # Now parse bit-fields for each register
        print(f"  Extracting bit-fields for {module_name} registers...")
        registers_with_bitfields = 0

        for register in registers:
            bit_fields = self.parse_register_bitfields(register["name"], module_name)
            if bit_fields:
                register["bit_fields"] = bit_fields
                registers_with_bitfields += 1

        print(
            f"  Added bit-fields for {registers_with_bitfields}/{len(registers)} registers"
        )

        return {"base_address_info": base_info, "registers": registers}

    def extract_base_address_info(self, soup: BeautifulSoup) -> Dict:
        """Extract base address formula and description from the page"""
        info = {"base_address_formula": "", "description": "", "notes": []}

        # Find the description table
        tables = soup.find_all("table", class_="noborder")

        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    header = cells[0].get_text().strip()
                    value = cells[1].get_text().strip()

                    if "Base Address" in header:
                        # Extract formula from the text
                        formula_match = re.search(
                            r"Register Base Address\s*=\s*([^\n]+)", value
                        )
                        if formula_match:
                            info["base_address_formula"] = formula_match.group(
                                1
                            ).strip()

                        # Extract notes
                        note_matches = re.findall(r"Note[^:]*:\s*([^\n]+)", value)
                        info["notes"] = [note.strip() for note in note_matches]

                    elif "Description" in header:
                        info["description"] = value

        return info

    def extract_register_table(
        self, soup: BeautifulSoup, module_name: str
    ) -> List[Dict]:
        """Extract register information from the register summary table"""
        registers = []

        # Find all tables (the register summary table doesn't have a specific class)
        tables = soup.find_all("table")

        for table in tables:
            # Skip tables with class 'noborder' (those are description tables)
            if table.get("class") and "noborder" in table.get("class"):
                continue

            rows = table.find_all("tr")

            if len(rows) < 2:
                continue

            # Check if this is a register table by examining the header
            header_row = rows[0]
            headers = [th.get_text().strip() for th in header_row.find_all("th")]

            # Look for register table indicators
            if not any("Register Name" in h or "Offset Address" in h for h in headers):
                continue

            # This is the register table! Process all data rows
            for row in rows[1:]:
                cells = row.find_all("td")

                if len(cells) >= 6:  # Name, Offset, Width, Type, Reset, Description
                    try:
                        # Extract text from cells
                        name = cells[0].get_text().strip()
                        offset = cells[1].get_text().strip()
                        width = cells[2].get_text().strip()
                        reg_type = cells[3].get_text().strip()
                        reset = cells[4].get_text().strip()
                        description = cells[5].get_text().strip()

                        # Try to convert width to int
                        try:
                            width = int(width)
                        except (ValueError, TypeError):
                            pass  # Keep as string if not a number

                        register = {
                            "name": name,
                            "offset": offset,
                            "width": width,
                            "type": reg_type,
                            "reset": reset,
                            "description": description,
                        }

                        registers.append(register)

                    except Exception as e:
                        print(f"Warning: Could not parse row: {e}")
                        continue

        return registers

    def parse_all_modules(self) -> Dict:
        """Parse all module documentation files with bit-fields"""
        database = {
            "version": "AM025-2024-11-13-1.1",
            "source": "Local HTML documentation with bit-fields",
            "source_path": self.docs_dir,
            "parsed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "modules": {},
        }

        for module_name, filename_or_list in MODULE_FILES.items():
            try:
                # Handle SHIM_MODULE which contains multiple files
                if module_name == "SHIM_MODULE":
                    shim_registers = []
                    shim_base_info = {
                        "base_address_formula": "",
                        "description": "Shim Tile Control and Status Registers (NOC and PL Interface)",
                        "notes": [],
                    }

                    for sub_module_name, filename in filename_or_list:
                        sub_module_data = self.parse_module_file(
                            sub_module_name, filename
                        )
                        shim_registers.extend(sub_module_data.get("registers", []))

                        # Merge base address info if available
                        sub_base_info = sub_module_data.get("base_address_info", {})
                        if sub_base_info.get("base_address_formula"):
                            if not shim_base_info["base_address_formula"]:
                                shim_base_info["base_address_formula"] = sub_base_info[
                                    "base_address_formula"
                                ]
                            else:
                                shim_base_info["base_address_formula"] += (
                                    " / " + sub_base_info["base_address_formula"]
                                )

                        # Merge notes
                        if sub_base_info.get("notes"):
                            shim_base_info["notes"].extend(sub_base_info["notes"])

                    module_data = {
                        "base_address_info": shim_base_info,
                        "registers": shim_registers,
                    }
                    database["modules"]["shim"] = module_data
                    print(
                        f"  Combined shim module: {len(shim_registers)} total registers"
                    )
                else:
                    # Single module files
                    module_data = self.parse_module_file(module_name, filename_or_list)
                    # Convert module name: "CORE_MODULE" -> "core"
                    clean_module_name = module_name.replace("_MODULE", "").lower()
                    database["modules"][clean_module_name] = module_data
            except Exception as e:
                print(f"Error parsing {module_name}: {e}")
                clean_name = module_name.replace("_MODULE", "").lower()
                database["modules"][clean_name] = {
                    "error": str(e),
                    "base_address_info": {},
                    "registers": [],
                }

        return database


def main():
    parser = argparse.ArgumentParser(
        description="Parse local AMD AIE-ML register documentation"
    )
    parser.add_argument(
        "--docs-dir",
        "-d",
        default="/workspace/am025/html",
        help="Path to HTML documentation directory (default: /workspace/am025/html)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="aie_registers_full.json",
        help="Output JSON file path (default: aie_registers_full.json)",
    )

    args = parser.parse_args()

    print("AMD AIE-ML Local Documentation Parser")
    print("=" * 60)
    print(f"Documentation directory: {args.docs_dir}")
    print(f"Output file: {args.output}")
    print()

    if not os.path.exists(args.docs_dir):
        print(f"Error: Documentation directory not found: {args.docs_dir}")
        return 1

    parser_obj = EnhancedAIEDocsParser(args.docs_dir)

    try:
        print("Parsing module summaries and extracting bit-fields...")
        print("This may take a minute as we parse individual register files...")
        print()

        database = parser_obj.parse_all_modules()

        # Save to JSON file
        with open(args.output, "w") as f:
            json.dump(database, f, indent=2)

        print()
        print("=" * 60)
        print(f"Successfully parsed register database to {args.output}")

        # Print summary
        total_registers = 0
        total_with_bitfields = 0

        for module_name, module_data in database["modules"].items():
            registers = module_data.get("registers", [])
            total_registers += len(registers)

            with_bitfields = sum(
                1 for r in registers if "bit_fields" in r and r["bit_fields"]
            )
            total_with_bitfields += with_bitfields

        print(f"Total modules: {len(database['modules'])}")
        print(f"Total registers: {total_registers}")
        print(f"Registers with bit-fields: {total_with_bitfields}")

        # Print per-module breakdown
        print()
        print("Per-module breakdown:")
        for module_name, module_data in database["modules"].items():
            registers = module_data.get("registers", [])
            reg_count = len(registers)
            with_bitfields = sum(
                1 for r in registers if "bit_fields" in r and r["bit_fields"]
            )
            print(
                f"  {module_name}: {reg_count} registers ({with_bitfields} with bit-fields)"
            )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
