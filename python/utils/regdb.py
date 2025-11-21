#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
#
"""
AIE Register Database Utilities

Provides the AIEAddressDecoder class for decoding AIE array addresses,
and MLIRModuleAnnotator for annotating MLIR files with register information.

Usage:
    from aie.utils.regdb import AIEAddressDecoder

    decoder = AIEAddressDecoder()
    info = decoder.parse_address(0x1D200)

CLI Usage:
    # Decode an address
    python -m aie.utils.regdb 0x100010

    # Annotate an MLIR file
    python -m aie.utils.regdb -a input.mlir -o output.mlir
    python -m aie.utils.regdb -a input.mlir --in-place
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from aie.utils.config import root_path


class AIEAddressDecoder:
    """Decoder for AIE array addresses"""

    # Memory region definitions: (name, base_offset, size_bytes)
    # These are regions where addresses within the range should show offset from base
    MEMORY_REGIONS = {
        "core": [
            ("Program_Memory", 0x20000, 0x4000),  # 16KB program memory
            ("DataMemory", 0x0, 0x10000),  # 64KB data memory
        ],
        "memory": [
            ("DataMemory", 0x0, 0x10000),  # 64KB data memory for memory module
        ],
        "memory_tile": [
            ("DataMemory", 0x0, 0x80000),  # 512KB data memory for memtile
        ],
    }

    def __init__(self):
        """Initialize decoder with register database"""
        self.database = None
        self.load_database()

    def load_database(self):
        """Load the register database from JSON file"""

        db_path = Path(root_path()) / "lib" / "regdb" / "aie_registers_aie2.json"

        try:
            with open(db_path, "r") as f:
                self.database = json.load(f)
        except FileNotFoundError:
            print(f"Error: Database file '{db_path}' not found.")
            print("Please run the appropriate parser script to generate the database.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in database file: {e}")
            raise

    def parse_address(self, address: int) -> Optional[Dict]:
        """
        Parse an AIE address and return decoded information

        Args:
            address: Address to decode (hex or decimal)

        Address format: ((col << column_shift) | (row << row_shift)) + offset

        Where:
        - col: column number
        - row: row number
          - row 0: shim tile (shim module)
          - row 1: memtile (memory_tile module)
          - row 2+: core tile (core, memory modules)
        """
        import aie.dialects.aie as aiedialect

        # Get target model from npu1 device
        device = aiedialect.AIEDevice.npu1
        target_model = aiedialect.get_target_model(device)

        # Extract col, row, and offset using target model
        col = (address >> target_model.get_column_shift()) & 0x7F
        row = (address >> target_model.get_row_shift()) & 0x1F
        register_offset = address & 0xFFFFF  # bits 0-19 (offset within tile)

        # Determine tile type and candidate modules
        if target_model.is_mem_tile(col, row):
            tile_type = "memory_tile"
            module_candidates = ["memory_tile"]
        elif target_model.is_core_tile(col, row):
            tile_type = "core_tile"
            module_candidates = ["core", "memory"]
        else:
            tile_type = "shim_tile"
            module_candidates = ["shim"]

        # Look for a match in the candidate modules
        for module_name in module_candidates:
            module_data = self.database.get("modules", {}).get(module_name, {})
            register_info = self.find_register_by_offset(module_data, register_offset)

            if register_info:
                result = {
                    "address": hex(address),
                    "col": col,
                    "row": row,
                    "tile_type": tile_type,
                    "module": module_name,
                    "register": register_info["name"],
                    "offset": register_info["offset"],
                    "width": register_info.get("width", "unknown"),
                    "type": register_info.get("type", "unknown"),
                    "description": register_info.get("description", ""),
                    "reset_value": register_info.get("reset", "unknown"),
                }

                # Include bit-fields if available
                if "bit_fields" in register_info:
                    result["bit_fields"] = register_info["bit_fields"]

                return result

        # If no exact match found, check for memory regions
        for module_name in module_candidates:
            memory_region = self.find_memory_region(module_name, register_offset)
            if memory_region:
                return {
                    "address": hex(address),
                    "col": col,
                    "row": row,
                    "tile_type": tile_type,
                    "module": module_name,
                    "register": f"{memory_region['name']}+0x{memory_region['region_offset']:X}",
                    "offset": hex(register_offset),
                    "description": memory_region["description"],
                    "is_memory_region": True,
                    "memory_region_name": memory_region["name"],
                    "memory_region_offset": memory_region["region_offset"],
                }

        # If no match found, return basic info
        return {
            "address": hex(address),
            "col": col,
            "row": row,
            "tile_type": tile_type,
            "module": "unknown",
            "register": "unknown",
            "offset": hex(register_offset),
            "description": "Address not found in database",
        }

    def find_register_by_offset(self, module_data: Dict, offset: int) -> Optional[Dict]:
        """Find a register in a module by its offset"""
        registers = module_data.get("registers", [])

        for register in registers:
            reg_offset_str = register.get("offset", "")

            # Parse offset (handle formats like "0x0000032000")
            try:
                if reg_offset_str.startswith("0x"):
                    reg_offset = int(reg_offset_str, 16)
                else:
                    reg_offset = int(reg_offset_str)

                if reg_offset == offset:
                    return register
            except (ValueError, AttributeError):
                continue

        return None

    def find_memory_region(self, module_name: str, offset: int) -> Optional[Dict]:
        """
        Check if an offset falls within a known memory region.

        Args:
            module_name: The module name (core, memory, memory_tile, etc.)
            offset: The register offset within the tile

        Returns:
            Dict with memory region info if found, None otherwise
        """
        # Check for memory regions in this module
        regions = self.MEMORY_REGIONS.get(module_name, [])

        for region_name, base_offset, size_bytes in regions:
            if base_offset <= offset < base_offset + size_bytes:
                # Calculate offset within the memory region
                region_offset = offset - base_offset
                return {
                    "name": region_name,
                    "base_offset": base_offset,
                    "region_offset": region_offset,
                    "size_bytes": size_bytes,
                    "description": f"{region_name} region ({size_bytes // 1024}KB)",
                }

        return None

    def encode_address(
        self,
        col: int,
        row: int,
        register_name: str,
        module_name: Optional[str] = None,
    ) -> Optional[int]:
        """
        Encode col, row, and register name into an address (reverse lookup)

        Address format: ((col << column_shift) | (row << row_shift)) + offset

        Args:
            col: Column number
            row: Row number
            register_name: Name of the register
            module_name: Optional module name to search in
        """
        import aie.dialects.aie as aiedialect

        # Get target model from npu1 device
        device = aiedialect.AIEDevice.npu1
        target_model = aiedialect.get_target_model(device)

        # Find the register
        register_info = None
        register_offset = None

        # AIE-ML format: nested modules
        modules_to_search = (
            [module_name] if module_name else self.database.get("modules", {}).keys()
        )

        for mod_name in modules_to_search:
            module_data = self.database["modules"].get(mod_name, {})
            for register in module_data.get("registers", []):
                if register.get("name") == register_name:
                    register_info = register
                    break
            if register_info:
                break

        if not register_info:
            return None

        # Parse register offset
        offset_str = register_info.get("offset", "")
        try:
            if offset_str.startswith("0x"):
                register_offset = int(offset_str, 16)
            else:
                register_offset = int(offset_str)
        except (ValueError, AttributeError):
            return None

        # Calculate address using target model
        address = (
            (col << target_model.get_column_shift())
            | (row << target_model.get_row_shift())
        ) + register_offset

        return address

    def format_result(self, result: Dict, include_delimiters: bool = False) -> str:
        """Format decoded result for display
        
        Args:
            result: Decoded address information
            include_delimiters: If True, include '=' and '-' separator lines
        """
        if not result:
            return "Unable to decode address"

        lines = []
        
        if include_delimiters:
            lines.append("=" * 70)
        
        lines.extend([
            f"Address: {result['address']}",
        ])
        
        if include_delimiters:
            lines.append("-" * 70)
        
        lines.extend([
            f"Column:      {result['col']}",
            f"Row:         {result['row']}",
            f"Tile Type:   {result['tile_type']}",
            f"Module:      {result['module']}",
            f"Register:    {result['register']}",
            f"Offset:      {result['offset']}",
        ])

        if "width" in result:
            lines.append(f"Width:       {result['width']} bits")
        if "type" in result:
            lines.append(f"Access Type: {result['type']}")
        if "reset_value" in result:
            lines.append(f"Reset Value: {result['reset_value']}")

        if include_delimiters:
            lines.append("-" * 70)

        if "description" in result and result["description"]:
            lines.append(f"Description: {result['description']}")

        if include_delimiters:
            lines.append("=" * 70)

        return "\n".join(lines)


class MLIRModuleAnnotator:
    """Annotates MLIR modules with register information using Python bindings"""

    def __init__(self):
        """Initialize with AIE decoder"""
        self.decoder = AIEAddressDecoder()

    def decode_bit_field_value(self, value: int, bit_range: str) -> int:
        """
        Extract bits from value based on bit range

        Args:
            value: The value to extract from
            bit_range: Bit range like "31:24" or "0"

        Returns:
            Extracted field value
        """
        if ":" in bit_range:
            # Range like "31:24"
            parts = bit_range.split(":")
            high = int(parts[0])
            low = int(parts[1])
            mask = (1 << (high - low + 1)) - 1
            field_value = (value >> low) & mask
        else:
            # Single bit like "0"
            bit = int(bit_range)
            field_value = (value >> bit) & 1

        return field_value

    def format_bit_fields(
        self, value: int, bit_fields: List[Dict], mask: Optional[int] = None
    ) -> List[str]:
        """
        Format bit-field interpretations for a value

        Args:
            value: The register value
            bit_fields: List of bit-field definitions
            mask: Optional mask to filter fields (only include fields with mask bits set)

        Returns:
            List of formatted bit-field strings
        """
        interpretations = []

        for field in bit_fields:
            field_name = field.get("name", "")
            bit_range = field.get("bits", "")

            if not bit_range:
                continue

            # If mask is provided, check if any bits of this field are in the mask
            if mask is not None:
                # Extract the field bits to see if they overlap with mask
                if ":" in bit_range:
                    parts = bit_range.split(":")
                    high = int(parts[0])
                    low = int(parts[1])
                    field_mask = ((1 << (high - low + 1)) - 1) << low
                else:
                    bit = int(bit_range)
                    field_mask = 1 << bit

                # Skip this field if it doesn't overlap with the mask
                if (field_mask & mask) == 0:
                    continue

            try:
                field_value = self.decode_bit_field_value(value, bit_range)

                # Format: "field_name=value"
                if field_name:
                    interp = f"{field_name}={field_value}"
                    interpretations.append(interp)
                else:
                    # No field name, just show value
                    interpretations.append(f"bits[{bit_range}]={field_value}")

            except (ValueError, IndexError):
                continue

        return interpretations

    def generate_comment(
        self,
        decoded_info: Dict,
        value: Optional[int] = None,
        mask: Optional[int] = None,
    ) -> str:
        """
        Generate comment string for register

        Format:
        Tile(col, row) Offset 0xOFFSET Name: Register | Value 0xX: field1=val1, field2=val2

        Args:
            decoded_info: Decoded address information
            value: Optional register value for bit-field decoding
            mask: Optional mask value (for maskwrite32 operations)

        Returns:
            Comment string
        """
        col = decoded_info.get("col", "?")
        row = decoded_info.get("row", "?")
        offset = decoded_info.get("offset", "0x0")
        register = decoded_info.get("register", "unknown")

        # Ensure offset is in hex format with 5 hex digits (20 bits)
        if isinstance(offset, int):
            offset = f"0x{offset:05X}"
        elif not offset.startswith("0x"):
            try:
                offset_int = int(offset, 16)
                offset = f"0x{offset_int:05X}"
            except (ValueError, TypeError):
                pass
        else:
            # Already has 0x prefix, reformat to 5 digits
            try:
                offset_int = int(offset, 16)
                offset = f"0x{offset_int:05X}"
            except (ValueError, TypeError):
                pass

        # Build comment string
        comment_parts = [f"Tile({col}, {row}), Offset {offset}, Name: {register}"]

        # Add bit-field interpretations (if value provided and bit-fields exist)
        if value is not None and "bit_fields" in decoded_info:
            bit_fields = decoded_info["bit_fields"]
            interpretations = self.format_bit_fields(value, bit_fields, mask)

            if interpretations:
                interp_str = ", ".join(interpretations)
                if mask is not None:
                    comment_parts.append(
                        f"Mask 0x{mask:X}, Value 0x{value:X}: {interp_str}"
                    )
                else:
                    comment_parts.append(f"Value 0x{value:X}: {interp_str}")

        return " | ".join(comment_parts)

    def get_address_from_op(
        self, op, target_model
    ) -> tuple[int, Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        Extract address, row, col, value, and mask from an operation.

        For operations with column/row attributes, computes the full address.
        For operations without column/row, extracts them from the address.

        Args:
            op: The MLIR operation (opview)
            target_model: The AIE target model for address calculations

        Returns:
            Tuple of (address, row, col, value, mask)
            - address: The register offset (20 bits)
            - row: The tile row
            - col: The tile column
            - value: The value being written (if applicable)
            - mask: The mask value (if applicable, for maskwrite32)
        """
        address = None
        row = None
        col = None
        value = None
        mask = None

        # Get address - could be 'address' or 'addr' attribute
        if hasattr(op, "address") and op.address is not None:
            address = int(op.address.value)
        elif hasattr(op, "addr") and op.addr is not None:
            address = int(op.addr.value)

        # Get row and column if present
        if hasattr(op, "row") and op.row is not None:
            row = int(op.row.value)
        if hasattr(op, "column") and op.column is not None:
            col = int(op.column.value)

        # Get value if present
        if hasattr(op, "value") and op.value is not None:
            value = int(op.value.value)

        # Get mask if present (for maskwrite32)
        if hasattr(op, "mask") and op.mask is not None:
            mask = int(op.mask.value)

        # If row/col not present, extract from address
        if address is not None and row is None and col is None:
            row = (address >> target_model.get_row_shift()) & 0x1F
            col = (address >> target_model.get_column_shift()) & 0x1F
            address = address & 0xFFFFF  # 20 bits register offset

        return address, row, col, value, mask

    def annotate_operation(self, op, target_model) -> bool:
        """
        Add a 'comment' attribute to an operation with register info.

        Args:
            op: The MLIR operation to annotate
            target_model: The AIE target model

        Returns:
            True if annotation was added, False otherwise
        """
        # Import here to avoid circular imports and allow module to load without MLIR
        from aie.ir import StringAttr

        address, row, col, value, mask = self.get_address_from_op(op, target_model)

        if address is None:
            return False

        # If we have row/col from operation attributes, we need to construct
        # a full address for the decoder (which expects col/row encoded in address)
        if row is not None and col is not None:
            # Construct full address using target model
            full_address = (
                (col << target_model.get_column_shift())
                | (row << target_model.get_row_shift())
            ) + address
        else:
            # Address already contains col/row encoded
            full_address = address

        # Decode the address
        decoded = self.decoder.parse_address(full_address)

        # Generate comment
        comment = self.generate_comment(decoded, value, mask)

        # Add comment attribute to the operation
        op.operation.attributes["comment"] = StringAttr.get(comment)

        return True

    def annotate_module(self, module) -> int:
        """
        Walk through module and annotate all relevant aiex.npu operations.

        Args:
            module: The MLIR module to annotate

        Returns:
            Number of operations annotated
        """
        # Import here to avoid circular imports and allow module to load without MLIR
        from aie.extras.util import find_ops
        import aie.dialects.aie as aiedialect
        import aie.dialects.aiex as aiexdialect

        annotated_count = 0

        # Get device and target model
        devices = find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiedialect.DeviceOp),
        )

        if not devices:
            print("Warning: No aie.device found, using default npu1 target model")
            device = aiedialect.AIEDevice.npu1
        else:
            device = aiedialect.AIEDevice(int(devices[0].device))

        target_model = aiedialect.get_target_model(device)

        # Find and annotate write32 operations
        write32_ops = find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiexdialect.NpuWrite32Op),
        )
        for op in write32_ops:
            if self.annotate_operation(op.opview, target_model):
                annotated_count += 1

        # Find and annotate blockwrite operations
        blockwrite_ops = find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiexdialect.NpuBlockWriteOp),
        )
        for op in blockwrite_ops:
            if self.annotate_operation(op.opview, target_model):
                annotated_count += 1

        # Find and annotate maskwrite32 operations
        maskwrite_ops = find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiexdialect.NpuMaskWrite32Op),
        )
        for op in maskwrite_ops:
            if self.annotate_operation(op.opview, target_model):
                annotated_count += 1

        # Find and annotate address_patch operations
        address_patch_ops = find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiexdialect.NpuAddressPatchOp),
        )
        for op in address_patch_ops:
            if self.annotate_operation(op.opview, target_model):
                annotated_count += 1

        # Find and annotate control_packet operations
        control_packet_ops = find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiexdialect.NpuControlPacketOp),
        )
        for op in control_packet_ops:
            if self.annotate_operation(op.opview, target_model):
                annotated_count += 1

        return annotated_count

    def annotate_file(self, input_path: str, output_path: Optional[str] = None) -> int:
        """
        Load MLIR file, annotate it, and write output.

        Args:
            input_path: Path to input MLIR file
            output_path: Path to output file (None = stdout)

        Returns:
            Number of operations annotated
        """
        # Import here to avoid circular imports and allow module to load without MLIR
        from aie.ir import Context, Module, Location
        from aie._mlir_libs import get_dialect_registry

        # Read input file
        with open(input_path, "r") as f:
            mlir_str = f.read()

        # Parse and annotate
        with Context() as ctx, Location.unknown():
            # Register the AIE dialects with the context
            ctx.append_dialect_registry(get_dialect_registry())
            ctx.load_all_available_dialects()

            module = Module.parse(mlir_str)
            count = self.annotate_module(module)

            # Get the annotated MLIR string
            output_str = str(module)

        # Write output
        if output_path:
            with open(output_path, "w") as f:
                f.write(output_str)
            print(f"Annotated {count} operations. Output written to: {output_path}")
        else:
            # Write to stdout
            sys.stdout.write(output_str)
            print(f"\n// Annotated {count} operations", file=sys.stderr)

        return count


def parse_address_arg(addr_str: str) -> int:
    """Parse address from string (supports hex and decimal)"""
    addr_str = addr_str.strip()

    if addr_str.startswith("0x") or addr_str.startswith("0X"):
        return int(addr_str, 16)
    else:
        return int(addr_str)


# A simple CLI for testing
def main():
    parser = argparse.ArgumentParser(
        description="AIE Register Database - Decode addresses or annotate MLIR files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decode an address
  %(prog)s 0x32000
  
  # Reverse lookup: find address for a register
  %(prog)s --col 0 --row 2 --register Core_Control

  # Annotate MLIR file and write to output
  %(prog)s -a input.mlir -o output.mlir
  
  # Annotate MLIR file in place
  %(prog)s -a input.mlir --in-place
  
  # Annotate MLIR file and write to stdout
  %(prog)s -a input.mlir
        """,
    )

    parser.add_argument("address", nargs="?", help="Address to decode (hex or decimal)")
    parser.add_argument("--col", type=int, help="Column number (for reverse lookup)")
    parser.add_argument("--row", type=int, help="Row number (for reverse lookup)")
    parser.add_argument("--register", "-r", help="Register name (for reverse lookup)")
    parser.add_argument("--module", "-m", help="Module name (for reverse lookup)")
    parser.add_argument(
        "--memory-tile-rows",
        type=int,
        default=1,
        help="Number of memory tile rows (default: 1)",
    )

    # MLIR annotation options
    parser.add_argument(
        "-a",
        "--annotate",
        metavar="MLIR_FILE",
        help="Annotate MLIR file with register information",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file for annotated MLIR (default: stdout)",
    )
    parser.add_argument(
        "--in-place",
        "-i",
        action="store_true",
        help="Modify MLIR file in place (use with -a)",
    )

    args = parser.parse_args()

    # MLIR annotation mode
    if args.annotate:
        # Determine output path
        if args.in_place:
            output_path = args.annotate
        else:
            output_path = args.output

        try:
            annotator = MLIRModuleAnnotator()
            annotator.annotate_file(args.annotate, output_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1
        return 0

    # Initialize decoder for address operations
    decoder = AIEAddressDecoder()

    # Reverse lookup mode
    if args.col is not None and args.row is not None and args.register:
        address = decoder.encode_address(args.col, args.row, args.register, args.module)

        if address:
            # Decode the address to get full information
            result = decoder.parse_address(address)
            # Format without delimiters
            print(decoder.format_result(result, include_delimiters=False))
        else:
            print(f"Error: Could not find register '{args.register}'")
            if args.module:
                print(f"       in module '{args.module}'")
            return 1

    # Forward lookup mode
    elif args.address:
        try:
            address = parse_address_arg(args.address)
            result = decoder.parse_address(address)
            print(decoder.format_result(result))
        except ValueError:
            print(f"Error: Invalid address format: {args.address}")
            return 1
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
