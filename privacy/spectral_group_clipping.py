from __future__ import annotations

"""Public, fixed L2 clipping for architecture-defined spectral groups."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch

from spectral.spectral_groups import GROUPS


@dataclass(frozen=True)
class SpectralClipStats:
    group_name: str
    pre_clip_norm: float
    post_clip_norm: float
    clip_bound: float
    clip_factor: float
    was_clipped: bool
    num_coefficients: int

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def _group_norm(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.tensor(0.0)
    total = sum(value.double().square().sum() for value in tensors.values())
    return torch.sqrt(total).float()


class SpectralGroupClipper:
    """Clip every group with a fixed, public bound.

    Bounds are copied at construction and never inferred from an update.  Norm
    statistics are diagnostics only and do not feed back into the mechanism.
    """

    def __init__(self, clip_config: Mapping[str, float], eps: float = 1e-12) -> None:
        missing = set(GROUPS) - set(clip_config)
        extra = set(clip_config) - set(GROUPS)
        if missing or extra:
            raise ValueError(
                f"clip_config must contain exactly {list(GROUPS)}; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if any(float(value) <= 0 for value in clip_config.values()):
            raise ValueError("all spectral clipping bounds must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.clip_config = {name: float(clip_config[name]) for name in GROUPS}
        self.eps = float(eps)

    def clip_group(
        self,
        group_name: str,
        tensors: Mapping[str, torch.Tensor],
        clip_norm: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], SpectralClipStats]:
        if group_name not in GROUPS:
            raise ValueError(f"unknown spectral group: {group_name}")
        bound = self.clip_config[group_name] if clip_norm is None else float(clip_norm)
        if bound <= 0:
            raise ValueError("clip_norm must be positive")
        for name, value in tensors.items():
            if not torch.is_floating_point(value):
                raise TypeError(f"spectral clipping requires floating tensor {group_name}/{name}")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(
                    f"non-finite retained coefficient in {group_name}/{name}; refusing DP release"
                )
        norm_tensor = _group_norm(tensors)
        pre_norm = float(norm_tensor)
        factor = min(1.0, bound / (pre_norm + self.eps))
        clipped = {name: value * factor for name, value in tensors.items()}
        post_norm = pre_norm * factor
        stats = SpectralClipStats(
            group_name=group_name,
            pre_clip_norm=pre_norm,
            post_clip_norm=post_norm,
            clip_bound=bound,
            clip_factor=factor,
            was_clipped=factor < 1.0,
            num_coefficients=sum(value.numel() for value in tensors.values()),
        )
        return clipped, stats

    def clip_groups(
        self, grouped_update: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> tuple[dict[str, dict[str, torch.Tensor]], list[SpectralClipStats]]:
        missing = set(GROUPS) - set(grouped_update)
        if missing:
            raise ValueError(f"grouped update is missing: {sorted(missing)}")
        clipped: dict[str, dict[str, torch.Tensor]] = {}
        statistics: list[SpectralClipStats] = []
        for group_name in GROUPS:
            clipped[group_name], stats = self.clip_group(
                group_name, grouped_update[group_name]
            )
            statistics.append(stats)
        return clipped, statistics
