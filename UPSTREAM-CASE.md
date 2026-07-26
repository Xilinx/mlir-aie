# Upstream case: field-level set-register-field helper over maskwrite32

## The hazard

Every lowering that pokes one bitfield of a shared control register (CORE_CONTROL,
a DMA channel CTRL word) has hand-coded that field's shift and mask as raw
constants. That is exactly how a write32-that-clobbers-a-sibling-field bug gets
in: nothing stops a lowering from mistyping a shift or citing the wrong
generation's mask, and a plain `npu.write32` (as opposed to `npu.maskwrite32`)
silently zeroes every other field packed into the same word on real hardware,
with no compile error and no runtime diagnostic.

## What this adds

`RegField`/`encodeRegisterField`/`createMaskWriteField`
(`lib/Dialect/AIEX/Utils/RegisterField.{h,cpp}`) is the single place a
`(register, field, value)` triple turns into an `npu.maskwrite32`. `RegField`
describes one bitfield the way aie-rt's `XAie_RegFldAttr` does (`Mask` plus the
register's own `RegOff`), read directly off an aie-rt reginit table.
`encodeRegisterField` derives the shift from the mask's trailing-zero count
instead of trusting a second, independently mistypeable lsb field, validates
the value fits the field's width, and rejects an unknown field (`mask == 0`).
`createMaskWriteField` always emits `npu.maskwrite32`, never `npu.write32`, so
a sibling field is preserved by construction, and diagnoses an out-of-width
value or unknown field on the caller's op instead of miscompiling silently.

`AIELowerCoreReset.cpp` and `AIELowerDmaChannelReset.cpp` are refactored onto
it. Both keep emitting byte-identical IR to what they hand-coded before, same
address/value/mask constants and op order, which is why their existing
`lower-core-reset` and `lower-dma-channel-reset` FileCheck tests pass
unchanged -- that equivalence is the correctness argument for the refactor,
and I re-ran both against a rebuilt `aie-opt` to confirm it.

## Why derive the shift instead of storing it

An earlier version of this header stored a separate `lsb` field and only
`assert()`-checked it against the mask's trailing-zero count. That is a real
gap: a copy-pasted `RegField` constant with the mask updated but the lsb
forgotten produces a value that passes the width check yet lands on the wrong
bits, and the header's whole claimed purpose is to make that unrepresentable,
not merely assert against it in builds that happen to keep assertions on.
`RegField` has no `lsb` member now; `encodeRegisterField` derives it from
`mask` every time, so the mismatch this header exists to prevent cannot occur.

## Test coverage

`test/CppTests/register_field_test.cpp` exercises `encodeRegisterField`
directly: the two 1-bit fields the shipped callers use, an 8-bit field to
cover the general width check, an out-of-width value, and the unknown-field
case. Neither existing caller feeds a caller-controlled value through a
shared field yet -- both only ever pass a fixed 0 or 1 into a 1-bit field --
so the diagnostic path is exercised through this CppTest, not through a
FileCheck test built on a synthetic op invented just for coverage.

## Ask

Land the helper plus the two refactored lowerings as one PR. It removes a
whole class of register-clobber bug from every future single-bitfield
lowering (BD length, cascade config, and anything else that pokes one field of
a shared control word), not just the two callers it ships with.
