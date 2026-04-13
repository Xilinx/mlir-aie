# Fix Buffer resolution in `inline_ops` and resnet RTP pattern (#3011)

## What

`InlineOpRuntimeTask.resolve()` now recursively resolves all `Buffer` instances
in its args before calling the user's function. A `Buffer` never given to any
`Worker` now raises `ValueError: Cannot resolve buffer until it has been placed`
instead of a confusing `AttributeError` from deep inside the callback.

The resnet `layers_conv2_x` RTP structure is cleaned up: replaced a 12-buffer
2D nested list (only 6 were ever used) with two named flat lists `rtp_conv1[i]`
and `rtp_conv1_skip[i]`, one buffer per worker that actually reads RTPs. Also
fixes a latent bug where the col-0 skip buffer was placed on `Tile(0,5)` instead
of `Tile(0,4)`. `test/python/localbuffer.py` is renamed to `buffer.py` to match
the current class name, and `test/python/buffer_resolution.py` adds three
regression tests that run in `check-aie` without requiring hardware.

## Why CI never caught it

The crash requires two conditions simultaneously: `Buffer.__setitem__` raising
instead of silently returning, and `set_rtps` writing to buffers that were never
given to a `Worker`. Commit `1d38a4c47b` introduced both the `raise` and the
fix to `set_rtps` in the **same commit**, so this broken state never existed in
main. CI was always green because it only ever ran against consistent states of
the repository.

The issue author must have been on a local branch or fork with only half of
`1d38a4c47b` applied — `buffer.py` with `raise` but `resnet.py` still writing
to all 12 buffers. That combination never got pushed to a branch CI ran against,
so it was invisible to automated testing.

The framework-level hazard itself (what happens when a `Buffer` is passed to
`inline_ops` but never given to a `Worker`) was never tested anywhere. The new
tests in `check-aie` close that gap without requiring a physical NPU.
