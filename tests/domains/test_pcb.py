"""
Tests for PCB domain module structure.

Verifies that the PCB domain packages are importable and correctly
structured as a domain pack.
"""

import importlib

import pytest


# Every PCB submodule should be importable
PCB_MODULES = [
    "domains.pcb",
    "domains.pcb.preprocessing",
    "domains.pcb.models",
    "domains.pcb.anomaly",
    "domains.pcb.risk",
    "domains.pcb.explainability",
    "domains.pcb.config",
]


@pytest.mark.parametrize("module_name", PCB_MODULES)
def test_pcb_module_importable(module_name):
    """Each PCB submodule should import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None
