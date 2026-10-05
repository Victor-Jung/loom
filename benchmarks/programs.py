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


def _col_softmax(ins: list[torch.Tensor]) -> torch.Tensor:
    return torch.softmax(ins[0].float(), dim=0)


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
    "gemm_bias_exp": Program(
        kernel="kernels/gemm_bias_exp.py",
        config="kernels/config_files/gemm_bias_exp.json",
        reference=lambda ins: torch.exp((ins[0] @ ins[1] + ins[2]) / 64.0),
    ),
    "softmax_twopass": Program(
        kernel="kernels/softmax_twopass.py",
        config="kernels/config_files/softmax_twopass.json",
        reference=_softmax,
    ),
    "batch_sum": Program(
        kernel="kernels/batch_sum.py",
        config="kernels/config_files/batch_sum.json",
        reference=lambda ins: ins[0].sum(dim=0),
    ),
    "transpose_matmul": Program(
        kernel="kernels/transpose_matmul.py",
        config="kernels/config_files/transpose_matmul.json",
        reference=lambda ins: ins[0].T @ ins[1],
    ),
    "row_center": Program(
        kernel="kernels/row_center.py",
        config="kernels/config_files/row_center.json",
        reference=lambda ins: ins[0] - ins[0].mean(dim=-1, keepdim=True),
    ),
    "attention_fullrow": Program(
        kernel="kernels/attention_fullrow.py",
        config="kernels/config_files/attention_fullrow.json",
        reference=lambda ins: torch.softmax(ins[0] @ ins[1].T, dim=-1) @ ins[2],
    ),
    # Full-array variants: 12x10 mesh, sizes where traffic matters.
    "toy_bmm_full": Program(
        kernel="kernels/toy_bmm.py",
        config="kernels/config_files/toy_bmm_full.json",
        reference=lambda ins: ins[0] @ ins[1],
        shape="B4_M2048_K1024_N2048",
    ),
    "shared_weight_bmm_full": Program(
        kernel="kernels/shared_weight_bmm.py",
        config="kernels/config_files/shared_weight_bmm_full.json",
        reference=lambda ins: ins[0] @ ins[1],
        shape="B8_M2048_K1024_N2048",
    ),
    "scaled_ewise_full": Program(
        kernel="kernels/scaled_ewise.py",
        config="kernels/config_files/scaled_ewise_full.json",
        reference=lambda ins: ins[0] * ins[1],
        shape="B16_M2048_N2048",
    ),
    "row_softmax_full": Program(
        kernel="kernels/row_softmax.py",
        config="kernels/config_files/row_softmax_full.json",
        reference=_softmax,
        shape="M8192_N1024",
    ),
    "gemm_bias_exp_full": Program(
        kernel="kernels/gemm_bias_exp.py",
        config="kernels/config_files/gemm_bias_exp_full.json",
        reference=lambda ins: torch.exp((ins[0] @ ins[1] + ins[2]) / 64.0),
        shape="M4096_K1024_N4096",
    ),
    "batch_sum_full": Program(
        kernel="kernels/batch_sum.py",
        config="kernels/config_files/batch_sum_full.json",
        reference=lambda ins: ins[0].sum(dim=0),
        shape="B16_M2048_N2048",
    ),
    "col_softmax_full": Program(
        kernel="kernels/col_softmax.py",
        config="kernels/config_files/col_softmax_full.json",
        reference=_col_softmax,
        shape="M1024_N8192",
    ),
    "row_center_full": Program(
        kernel="kernels/row_center.py",
        config="kernels/config_files/row_center_full.json",
        reference=lambda ins: ins[0] - ins[0].mean(dim=-1, keepdim=True),
        shape="M8192_N1024",
    ),
    "transpose_matmul_full": Program(
        kernel="kernels/transpose_matmul.py",
        config="kernels/config_files/transpose_matmul_full.json",
        reference=lambda ins: ins[0].T @ ins[1],
        shape="K1024_M4096_N4096",
    ),
    "chain_fused_full": Program(
        kernel="kernels/chain_fused.py",
        config="kernels/config_files/chain_fused_full.json",
        reference=lambda ins: (ins[0] @ ins[1]) @ ins[2],
        shape="M4096_K128_N4096_P128",
    ),
    "col_softmax": Program(
        kernel="kernels/col_softmax.py",
        config="kernels/config_files/col_softmax.json",
        reference=_col_softmax,
    ),
    "bias_add": Program(
        kernel="kernels/bias_add.py",
        config="kernels/config_files/bias_add.json",
        reference=lambda ins: ins[0] + ins[1],
    ),
    "outer_scale": Program(
        kernel="kernels/outer_scale.py",
        config="kernels/config_files/outer_scale.json",
        reference=lambda ins: ins[0] * ins[1] * ins[2],
    ),
    "weighted_batch_sum": Program(
        kernel="kernels/weighted_batch_sum.py",
        config="kernels/config_files/weighted_batch_sum.json",
        reference=lambda ins: (ins[0] * ins[1]).sum(dim=0),
    ),
    "bias_add_full": Program(
        kernel="kernels/bias_add.py",
        config="kernels/config_files/bias_add_full.json",
        reference=lambda ins: ins[0] + ins[1],
        shape="B16_M2048_N2048",
    ),
    "outer_scale_full": Program(
        kernel="kernels/outer_scale.py",
        config="kernels/config_files/outer_scale_full.json",
        reference=lambda ins: ins[0] * ins[1] * ins[2],
        shape="M4096_N4096",
    ),
    "weighted_batch_sum_full": Program(
        kernel="kernels/weighted_batch_sum.py",
        config="kernels/config_files/weighted_batch_sum_full.json",
        reference=lambda ins: (ins[0] * ins[1]).sum(dim=0),
        shape="B16_M2048_N2048",
    ),
    "softmax_twopass_full": Program(
        kernel="kernels/softmax_twopass.py",
        config="kernels/config_files/softmax_twopass_full.json",
        reference=_softmax,
        shape="M8192_N1024",
    ),
    "attention_fullrow_full": Program(
        kernel="kernels/attention_fullrow.py",
        config="kernels/config_files/attention_fullrow_full.json",
        reference=lambda ins: torch.softmax(ins[0] @ ins[1].T, dim=-1) @ ins[2],
        shape="M4096_N256_D64",
    ),
}
