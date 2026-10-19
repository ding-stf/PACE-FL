from __future__ import annotations

"""Development support for joint clipping under a fixed ellipsoidal norm.

The class in this module performs clipping only. It does not allocate noise,
account privacy loss, or make data-derived scales public. A caller is
responsible for the provenance and privacy status of ``group_scales``.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from spectral.spectral_groups import GROUPS


@dataclass(frozen=True)
class EllipsoidGroupDiagnostics:
    group_name: str
    scale: float
    raw_group_norm: float
    normalized_group_norm: float
    restored_group_norm_after_clip: float
    group_retention_ratio: float
    group_clipping_distortion: float
    ellipsoid_norm_contribution: float
    num_coefficients: int


@dataclass(frozen=True)
class EllipsoidalClipDiagnostics:
    normalized_joint_norm_preclip: float
    normalized_joint_norm_postclip: float
    joint_clip_factor: float
    raw_joint_norm: float
    post_clip_joint_norm: float
    clipping_distortion: float
    cosine_similarity: float
    group_diagnostics: tuple[EllipsoidGroupDiagnostics, ...]


def _validate_scales(scales: Mapping[str, float]) -> dict[str, float]:
    missing = set(GROUPS) - set(scales)
    extra = set(scales) - set(GROUPS)
    if missing or extra:
        raise ValueError(
            f"scales must contain exactly {list(GROUPS)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    values = {group: float(scales[group]) for group in GROUPS}
    invalid = [
        group
        for group, value in values.items()
        if not math.isfinite(value) or value <= 0.0
    ]
    if invalid:
        raise ValueError(f"scales must be finite and positive for {invalid}")
    return values


def normalize_scales(
    scales: Mapping[str, float], mode: str = "mean1"
) -> dict[str, float]:
    """Normalize fixed group scales without changing their relative geometry."""

    values = _validate_scales(scales)
    if mode == "none":
        return values
    if mode == "mean1":
        denominator = sum(values.values()) / len(values)
    elif mode == "geomean1":
        denominator = math.exp(
            sum(math.log(value) for value in values.values()) / len(values)
        )
    else:
        raise ValueError("normalization mode must be none, mean1, or geomean1")
    return {group: value / denominator for group, value in values.items()}


def _sum_squares(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    total: torch.Tensor | None = None
    for name, value in tensors.items():
        if not torch.is_floating_point(value):
            raise TypeError(f"ellipsoidal clipping requires floating tensor {name}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite tensor in ellipsoidal clipping: {name}")
        term = value.detach().double().square().sum()
        total = term if total is None else total + term
    return total


def _group_dot(
    first: Mapping[str, Mapping[str, torch.Tensor]],
    second: Mapping[str, Mapping[str, torch.Tensor]],
) -> float:
    total = 0.0
    for group in GROUPS:
        for name, value in first[group].items():
            total += float(
                (value.detach().double() * second[group][name].detach().double()).sum()
            )
    return total


class EllipsoidalSpectralClipper:
    """Apply one shared radial clip factor under a fixed ellipsoidal norm."""

    def __init__(
        self,
        group_scales: Mapping[str, float],
        joint_clip_norm: float = 1.0,
        *,
        normalization: str = "mean1",
        eps: float = 1e-12,
    ) -> None:
        if not math.isfinite(joint_clip_norm) or joint_clip_norm <= 0:
            raise ValueError("joint_clip_norm must be finite and positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.input_scales = _validate_scales(group_scales)
        self.scales = normalize_scales(group_scales, normalization)
        self.normalization = normalization
        self.joint_clip_norm = float(joint_clip_norm)
        self.eps = float(eps)

    def clip_groups(
        self, grouped_update: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> tuple[dict[str, dict[str, torch.Tensor]], EllipsoidalClipDiagnostics]:
        missing = set(GROUPS) - set(grouped_update)
        extra = set(grouped_update) - set(GROUPS)
        if missing or extra:
            raise ValueError(
                f"grouped update must contain exactly {list(GROUPS)}; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        group_sq: dict[str, torch.Tensor | None] = {}
        raw_norms: dict[str, float] = {}
        num_coefficients: dict[str, int] = {}
        joint_sq_tensor: torch.Tensor | None = None
        raw_sq_total = 0.0
        for group in GROUPS:
            group_sq[group] = _sum_squares(grouped_update[group])
            group_sq_value = (
                float(group_sq[group].detach().cpu()) if group_sq[group] is not None else 0.0
            )
            raw_sq_total += group_sq_value
            raw_norms[group] = math.sqrt(group_sq_value)
            num_coefficients[group] = sum(
                value.numel() for value in grouped_update[group].values()
            )
            if group_sq[group] is not None:
                normalized_term = group_sq[group] / (self.scales[group] ** 2)
                joint_sq_tensor = (
                    normalized_term
                    if joint_sq_tensor is None
                    else joint_sq_tensor + normalized_term
                )

        joint_sq = (
            float(joint_sq_tensor.detach().cpu()) if joint_sq_tensor is not None else 0.0
        )
        joint_norm = math.sqrt(joint_sq)
        alpha = min(1.0, self.joint_clip_norm / (joint_norm + self.eps))
        clipped = {
            group: {name: value * alpha for name, value in tensors.items()}
            for group, tensors in grouped_update.items()
        }

        raw_joint_norm = math.sqrt(raw_sq_total)
        post_joint_norm = raw_joint_norm * alpha
        if raw_joint_norm > self.eps:
            cosine = _group_dot(grouped_update, clipped) / (
                raw_joint_norm * post_joint_norm + self.eps
            )
            cosine = min(1.0, max(-1.0, cosine))
            distortion = abs(1.0 - alpha)
        else:
            cosine = 1.0
            distortion = 0.0

        group_diagnostics = []
        for group in GROUPS:
            raw_norm = raw_norms[group]
            normalized_norm = raw_norm / self.scales[group]
            contribution = (
                normalized_norm * normalized_norm / (joint_sq + self.eps)
                if joint_sq > 0
                else 0.0
            )
            group_diagnostics.append(
                EllipsoidGroupDiagnostics(
                    group_name=group,
                    scale=self.scales[group],
                    raw_group_norm=raw_norm,
                    normalized_group_norm=normalized_norm,
                    restored_group_norm_after_clip=raw_norm * alpha,
                    group_retention_ratio=alpha if raw_norm > self.eps else 1.0,
                    group_clipping_distortion=(1.0 - alpha) if raw_norm > self.eps else 0.0,
                    ellipsoid_norm_contribution=contribution,
                    num_coefficients=num_coefficients[group],
                )
            )

        return clipped, EllipsoidalClipDiagnostics(
            normalized_joint_norm_preclip=joint_norm,
            normalized_joint_norm_postclip=joint_norm * alpha,
            joint_clip_factor=alpha,
            raw_joint_norm=raw_joint_norm,
            post_clip_joint_norm=post_joint_norm,
            clipping_distortion=distortion,
            cosine_similarity=cosine,
            group_diagnostics=tuple(group_diagnostics),
        )
