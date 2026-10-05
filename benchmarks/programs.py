"""Registry of benchmark programs: kernel, configuration, and golden reference.

Every program is checked against its golden function on the board; the
reference is computed in float32 from the same random inputs the kernel sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

Reference = Callable[[list[torch.Tensor]], torch.Tensor]


@dataclass(frozen=True)
class Program:
    kernel: str          # kernel script, repo-relative
    config: str          # pipeline configuration, repo-relative
    reference: Reference # golden: inputs (in kernel argument order) -> output
    shape: str | None = None  # kernel-size override, e.g. "M256_K64_N256_P64"
    min_pcc: float = 0.99


def _softmax(ins: list[torch.Tensor]) -> torch.Tensor:
    return torch.softmax(ins[0].float(), dim=-1)


PROGRAMS: dict[str, Program] = {
    "toy_bmm": Program(
        kernel="kernels/toy_bmm.py",
        config="kernels/config_files/toy_bmm.json",
        reference=lambda ins: ins[0] @ ins[1],
    ),
    "toy_ewise3d": Program(
        kernel="kernels/toy_ewise3d.py",
        config="kernels/config_files/toy_ewise3d.json",
        reference=lambda ins: ins[0] + ins[1],
    ),
    "chain_fused": Program(
        kernel="kernels/chain_fused.py",
        config="kernels/config_files/chain_fused.json",
        reference=lambda ins: (ins[0] @ ins[1]) @ ins[2],
        shape="M256_K64_N256_P64",
    ),
    "shared_weight_bmm": Program(
        kernel="kernels/shared_weight_bmm.py",
        config="kernels/config_files/shared_weight_bmm.json",
        reference=lambda ins: ins[0] @ ins[1],
    ),
    "scaled_ewise": Program(
        kernel="kernels/scaled_ewise.py",
        config="kernels/config_files/scaled_ewise.json",
        reference=lambda ins: ins[0] * ins[1],
    ),
    "row_softmax": Program(
        kernel="kernels/row_softmax.py",
        config="kernels/config_files/row_softmax.json",
        reference=_softmax,
    ),
}
