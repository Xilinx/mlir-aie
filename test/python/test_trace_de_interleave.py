# test_trace_de_interleave.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python -m pytest %s -v

"""trace_pkts_de_interleave() must not attribute the payload of an invalid header.

Nonzero capture data is not evidence of a valid trace packet. An invalid first
header is an error. The payload of an invalid header after a valid one is
dropped with a warning, not filed under the previous packet's tile.
"""

import logging

import pytest
from aie.utils.trace.utils import trace_pkts_de_interleave


def test_invalid_first_packet_header_reports_word():
    with pytest.raises(ValueError, match="invalid packet header at word 0: deadbeef"):
        trace_pkts_de_interleave(["deadbeef"] + ["12345678"] * 7)


def test_valid_first_packet_header_is_de_interleaved():
    # Row 2, column 0, type 0, ID 1, with odd parity in bit 31.
    packets = trace_pkts_de_interleave(["80020001"] + ["12345678"] * 7)
    assert packets[0]["2,0"] == ["12345678"] * 7


def test_invalid_later_packet_header_drops_its_payload(caplog):
    # Valid headers for row 2 and row 3 of column 0, with an invalid (even
    # parity) one between them.
    stream = (
        ["80020001"]
        + ["11111111"] * 7
        + ["deadbeef"]
        + ["22222222"] * 7
        + ["00030001"]
        + ["33333333"] * 7
    )
    with caplog.at_level(logging.WARNING):
        packets = trace_pkts_de_interleave(stream)
    assert packets[0] == {"2,0": ["11111111"] * 7, "3,0": ["33333333"] * 7}
    assert "invalid header at word 8: deadbeef" in caplog.text
