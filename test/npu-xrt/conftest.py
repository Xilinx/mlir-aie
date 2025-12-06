import pytest


@pytest.fixture(scope="module", autouse=True)
def reset_xrt_host_runtime():
    try:
        from aie.iron.hostruntime.xrtruntime.hostruntime import XRTHostRuntime

        if XRTHostRuntime._instance:
            XRTHostRuntime._instance.reset()
    except ImportError:
        pass
