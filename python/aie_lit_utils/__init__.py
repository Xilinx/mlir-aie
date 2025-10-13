# AIE Lit Test Utilities
"""
Shared utilities for AIE/AIR lit test configurations.

This package provides centralized hardware detection and configuration
for lit test suites across the AIE project.
"""

from .lit_config_helpers import LitConfigHelper, HardwareConfig

__all__ = ['LitConfigHelper', 'HardwareConfig']
