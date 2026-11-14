# test_trace_de_interleave.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python -m pytest %s -v

"""trace_pkts_de_interleave() must reject a segment whose first header is invalid.

Nonzero capture data is not evidence of a valid trace packet. An invalid first
header used to leave the packet location empty, and the next payload word then
crashed with KeyError: ''.
"""

import pytest
from aie.utils.trace.utils import trace_pkts_de_interleave


def test_invalid_first_packet_header_reports_word():
    with pytest.raises(ValueError, match="invalid packet header at word 0: deadbeef"):
        trace_pkts_de_interleave(["deadbeef"] + ["12345678"] * 7)


def test_valid_first_packet_header_is_de_interleaved():
    # Row 2, column 0, type 0, ID 1, with odd parity in bit 31.
    packets = trace_pkts_de_interleave(["80020001"] + ["12345678"] * 7)
    assert packets[0]["2,0"] == ["12345678"] * 7
