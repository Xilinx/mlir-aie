#!/bin/bash
# helper sourced at the top of each Claude-driven build shell.
# brings up the same env that quick_setup.sh produced, without re-creating ironenv.
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/ironenv/bin/activate"
_AIE_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
source "${_AIE_ROOT}/utils/env_setup.sh" "${_AIE_ROOT}/install" >/dev/null 2>&1
