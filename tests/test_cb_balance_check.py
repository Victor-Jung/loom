"""Unit tests for split_kernel.check_cb_balance.

The checker is a lexical heuristic over generated device kernels, NOT a proof of
deadlock-freedom: flash attention trips it in both directions and runs correctly
on hardware. These tests pin the counting behaviour and, importantly, that it
only ever reports -- it must not gate compilation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPLIT = Path(__file__).resolve().parents[1] / "third_party" / "loom2ttkernel" / "split_kernel.py"
spec = importlib.util.spec_from_file_location("split_kernel", SPLIT)
sk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sk)


def test_balanced_kernel_reports_nothing() -> None:
    src = [
        "cb_wait_front(cb_id_binding0, 4);\n",
        "cb_pop_front(cb_id_binding0, 4);\n",
    ]
    errors, warnings = sk.check_cb_balance(src, "compute.cpp")
    assert errors == [] and warnings == []


def test_pop_without_wait_is_reported() -> None:
    errors, _ = sk.check_cb_balance(["cb_pop_front(cb_id_binding4, 2);\n"], "compute.cpp")
    assert len(errors) == 1
    assert "cb_id_binding4" in errors[0] and "pop without a matching wait" in errors[0]


def test_surplus_wait_is_a_warning_not_an_error() -> None:
    src = [
        "cb_wait_front(cb_id_binding0, 1);\n",
        "cb_wait_front(cb_id_binding0, 1);\n",
        "cb_pop_front(cb_id_binding0, 1);\n",
    ]
    errors, warnings = sk.check_cb_balance(src, "compute.cpp")
    assert errors == []
    assert len(warnings) == 1 and "never released" in warnings[0]


def test_helper_parameter_names_are_ignored() -> None:
    """in_cb/out_cb/cb_id are per-call aliases inside helpers; counting them
    would produce noise unrelated to any real buffer."""
    src = ["cb_wait_front(in_cb, tile_count);\n", "cb_pop_front(cb_id, tile_count);\n"]
    errors, warnings = sk.check_cb_balance(src, "compute.cpp")
    assert errors == [] and warnings == []


def test_checker_never_raises() -> None:
    """It is advisory: a pathological kernel must still return, not abort."""
    errors, warnings = sk.check_cb_balance(["cb_pop_front(cb_id_binding9, 1);\n"] * 5, "compute.cpp")
    assert len(errors) == 1
