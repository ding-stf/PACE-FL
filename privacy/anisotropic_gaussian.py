from __future__ import annotations

"""Client-side joint anisotropic Gaussian mechanism for spectral groups.

The mechanism in this module protects the *individual update observed by the
server*.  It is therefore a local, client-update-level mechanism.  Projection
and grouping must be fixed, public and data independent before this mechanism
is called.

Each released group is clipped to a public L2 bound ``C_b``.  Independent
Gaussian noise with a public standard deviation ``s_b`` is then added to every
coordinate in that group.  Although the covariance is block diagonal, the
groups form one joint release.  For explicit per-block sensitivity bounds
``Delta_b`` (``C_b`` for add/remove or ``2*C_b`` for replacement), a rigorous
upper bound on the squared whitened sensitivity is

    sum_b (Delta_b / s_b) ** 2.

The bound is deliberately conservative: separate group clipping describes a
Cartesian product of L2 balls, so privacy accounting must not silently replace
the sum by one.  See :mod:`privacy.pjsdp_accountant` for multi-round RDP
composition and target-epsilon calibration.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch


Adjacency = Literal["add_remove", "replacement"]
GroupedTensors = Mapping[str, Mapping[str, torch.Tensor]]


@dataclass(frozen=True)
class AnisotropicGaussianResult:
    """A grouped DP release and diagnostics safe for the configured interface.

    ``diagnostics`` contains public mechanism parameters by default.  Private
    pre-clipping norms are returned only when ``include_private_diagnostics`` is
    explicitly enabled; those values must not be sent to the untrusted server
    or published without a separate privacy analysis.
    """

    groups: dict[str, dict[str, torch.Tensor]]
    dp_applied: bool
    diagnostics: dict[str, dict[str, float]]


def _validated_positive_map(
    name: str, values: Mapping[str, float]
) -> dict[str, float]:
    if not values:
        raise ValueError(f"{name} must contain at least one group")
    result = {group: float(value) for group, value in values.items()}
    invalid = [
        group
        for group, value in result.items()
        if not math.isfinite(value) or value <= 0.0
    ]
    if invalid:
        raise ValueError(f"{name} must be finite and positive for {sorted(invalid)}")
    return result


def _validate_same_groups(
    first_name: str,
    first: Mapping[str, object],
    second_name: str,
    second: Mapping[str, object],
) -> None:
    first_groups = set(first)
    second_groups = set(second)
    if first_groups != second_groups:
        missing = sorted(first_groups - second_groups)
        extra = sorted(second_groups - first_groups)
        raise ValueError(
            f"{second_name} groups must match {first_name}; "
            f"missing={missing}, extra={extra}"
        )


def group_l2_norm(tensors: Mapping[str, torch.Tensor]) -> float:
    """Return the L2 norm of all tensors in one group, accumulated in float64."""

    return math.sqrt(
        sum(float(value.detach().double().square().sum()) for value in tensors.values())
    )


def whitened_sensitivity_squared(
    group_sensitivities: Mapping[str, float],
    group_noise_stds: Mapping[str, float],
    *,
    adjacency: Adjacency = "add_remove",
) -> float:
    """Conservative squared Mahalanobis sensitivity of one joint release.

    ``group_sensitivities[b]`` is an L2 sensitivity bound for block ``b`` and
    ``group_noise_stds[b]`` is the per-coordinate Gaussian standard deviation
    for that block.  The caller must already supply ``C_b`` for add/remove or
    ``2*C_b`` for replacement; the adjacency label is validated metadata and
    does not rescale the supplied sensitivity a second time.
    """

    if adjacency not in {"add_remove", "replacement"}:
        raise ValueError("adjacency must be add_remove or replacement")
    sensitivities = _validated_positive_map(
        "group_sensitivities", group_sensitivities
    )
    noise_stds = _validated_positive_map("group_noise_stds", group_noise_stds)
    _validate_same_groups(
        "group_sensitivities",
        sensitivities,
        "group_noise_stds",
        noise_stds,
    )
    return float(
        sum(
            (sensitivities[group] / noise_stds[group]) ** 2
            for group in sensitivities
        )
    )


def gaussian_rdp(
    order: float,
    group_sensitivities: Mapping[str, float],
    group_noise_stds: Mapping[str, float],
    *,
    adjacency: Adjacency = "add_remove",
) -> float:
    """RDP of the joint block-diagonal Gaussian mechanism at one order."""

    alpha = float(order)
    if not math.isfinite(alpha) or alpha <= 1.0:
        raise ValueError("RDP order must be finite and greater than one")
    squared = whitened_sensitivity_squared(
        group_sensitivities, group_noise_stds, adjacency=adjacency
    )
    return alpha * squared / 2.0


class JointAnisotropicGaussianMechanism:
    """Clip public spectral groups and release them with fixed Gaussian noise.

    This class does not perform a spectral transform or choose a projection.  A
    caller must provide only the fixed projected groups that are intended to be
    released.  Requiring exact group-key equality prevents an unconfigured raw
    group from passing through accidentally.
    """

    aggregation_level = "local"

    def __init__(
        self,
        clip_bounds: Mapping[str, float],
        noise_stds: Mapping[str, float],
        generator: torch.Generator | None = None,
    ) -> None:
        self.clip_bounds = _validated_positive_map("clip_bounds", clip_bounds)
        self.noise_stds = _validated_positive_map("noise_stds", noise_stds)
        _validate_same_groups(
            "clip_bounds", self.clip_bounds, "noise_stds", self.noise_stds
        )
        self.generator = generator

    @property
    def released_groups(self) -> tuple[str, ...]:
        return tuple(self.clip_bounds)

    @property
    def whitened_sensitivity_squared(self) -> float:
        return whitened_sensitivity_squared(self.clip_bounds, self.noise_stds)

    def privatize(
        self,
        grouped_update: GroupedTensors,
        *,
        include_private_diagnostics: bool = False,
    ) -> AnisotropicGaussianResult:
        _validate_same_groups(
            "clip_bounds", self.clip_bounds, "grouped_update", grouped_update
        )

        released: dict[str, dict[str, torch.Tensor]] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for group in self.clip_bounds:
            tensors = grouped_update[group]
            for name, value in tensors.items():
                if not torch.is_floating_point(value):
                    raise TypeError(
                        f"Gaussian noise requires floating tensor {group}/{name}"
                    )
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(
                        f"non-finite clipped coefficient in {group}/{name}; refusing Gaussian release"
                    )

            before = group_l2_norm(tensors)
            clip_bound = self.clip_bounds[group]
            clip_factor = min(1.0, clip_bound / (before + 1e-12))
            noise_std = self.noise_stds[group]
            released[group] = {}
            for name, value in tensors.items():
                noise = torch.randn(
                    value.shape,
                    dtype=value.dtype,
                    device=value.device,
                    generator=self.generator,
                )
                released[group][name] = value * clip_factor + noise * noise_std

            public_stats = {
                "clip_bound": clip_bound,
                "noise_std": noise_std,
                "num_coefficients": float(
                    sum(value.numel() for value in tensors.values())
                ),
            }
            if include_private_diagnostics:
                public_stats.update(
                    {
                        "pre_clip_norm": before,
                        "post_clip_norm": before * clip_factor,
                        "clip_factor": clip_factor,
                        "was_clipped": float(clip_factor < 1.0),
                    }
                )
            diagnostics[group] = public_stats

        return AnisotropicGaussianResult(released, True, diagnostics)
