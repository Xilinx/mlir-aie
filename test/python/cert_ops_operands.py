# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# The operand forms the `aiex.cert` builders accept: registers and wait handles
# as ints or prebuilt attributes, barriers as assembler names, ints or prebuilt
# attributes, and the diagnostics for values the ISA has no encoding for.

from aie.dialects.aie import device, AIEDevice
from aie.dialects.aiex import *
from aie.ir import (
    Attribute,
    Context,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
)

REG = "expected a CERT register 0..23, got "
LB = "expected a barrier lb0..lb15 or its index 0..15, got "
RB = "expected a barrier rb0..rb6 or its index 0..6, got "


def expect_error(kind, message, build):
    try:
        build()
    except kind as e:
        assert str(e) == message, str(e)
    else:
        raise AssertionError(f"expected {message!r}")


with Context(), Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):

        @device(AIEDevice.xcve3858)
        def device_body():
            @cert_page
            def _():
                @cert_job(0)
                def _():
                    # The generated builders' own forms still work: a prebuilt
                    # attribute is used as it is.
                    i32 = IntegerAttr.get(IntegerType.get_signless(32), 9)
                    # CHECK: aiex.cert.read32(9, 8192)
                    cert_read32(i32, 0x2000)
                    # CHECK: aiex.cert.wait_uc_dma(9)
                    cert_wait_uc_dma(i32)
                    # CHECK: aiex.cert.local_barrier(lb3, 2)
                    cert_local_barrier(Attribute.parse("3 : i32"), 2)
                    # CHECK: aiex.cert.remote_barrier(rb2, 1)
                    cert_remote_barrier(Attribute.parse("2 : i32"), 1)

                    # The op classes accept the same forms.
                    # CHECK: aiex.cert.read32_d(0, 23)
                    CertRead32DOp(0, 23)
                    # CHECK: aiex.cert.local_barrier(lb0, 1)
                    CertLocalBarrierOp("lb0", 1)

                    # Out-of-range operands are diagnosed where they are built.
                    err = expect_error
                    err(ValueError, REG + "24", lambda: cert_read32(24, 0x2000))
                    err(ValueError, REG + "-1", lambda: cert_read32(-1, 0x2000))
                    err(ValueError, REG + "99", lambda: cert_wait_uc_dma(99))
                    err(TypeError, REG + "True", lambda: cert_wait_uc_dma(True))
                    err(
                        ValueError, LB + "'lb16'", lambda: cert_local_barrier("lb16", 2)
                    )
                    err(ValueError, LB + "16", lambda: cert_local_barrier(16, 2))
                    err(ValueError, RB + "'rb7'", lambda: cert_remote_barrier("rb7", 1))
                    err(ValueError, RB + "'lb1'", lambda: cert_remote_barrier("lb1", 1))
                    err(ValueError, RB + "'rb'", lambda: cert_remote_barrier("rb", 1))
                    err(
                        ValueError,
                        RB + "'rb+1'",
                        lambda: cert_remote_barrier("rb+1", 1),
                    )
                    err(
                        ValueError,
                        RB + "'rb01'",
                        lambda: cert_remote_barrier("rb01", 1),
                    )
                    err(
                        ValueError,
                        LB + "'lb\xb2'",
                        lambda: cert_local_barrier("lb\xb2", 2),
                    )
                    err(
                        ValueError,
                        RB + "'rb\xb2'",
                        lambda: cert_remote_barrier("rb\xb2", 1),
                    )
                    err(TypeError, RB + "1.5", lambda: cert_remote_barrier(1.5, 1))

    assert module.operation.verify()
    print(module)
