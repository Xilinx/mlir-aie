try:
    import pyxrt as xrt
except Exception as e:
    raise ImportError(f"Cannot import pyxrt (err={e})... is XRT installed?")
