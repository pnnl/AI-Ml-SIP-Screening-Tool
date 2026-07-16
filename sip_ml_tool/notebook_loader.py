"""Utilities for loading notebook workflow modules in order."""

from pathlib import Path


def load_notebook_module(module_name: str, namespace: dict) -> None:
    """Execute a workflow module inside the notebook namespace.

    The SIP QC notebook uses an ordered patch stack where later sections
    intentionally replace earlier functions. Executing these modules into the
    notebook globals preserves that behavior while keeping the notebook short.
    """
    module_path = Path(__file__).resolve().parent / module_name
    code = module_path.read_text(encoding="utf-8")
    exec(compile(code, str(module_path), "exec"), namespace)
