from __future__ import annotations

"""Projected Joint Spectral Differential Privacy (PJSDP) pipeline.

The server-visible client update follows exactly:

``raw -> spectral transform -> fixed projection -> fixed group clipping ->
joint anisotropic Gaussian noise -> reconstruction``.

Unreleased spectral coordinates are projected to zero both before and after
noise addition.  No raw private coefficient bypasses the mechanism.
"""

from collections.abc import Mapping

import torch

from spectral.spectral_groups import GROUPS, inverse_groups, transform_update

from .anisotropic_gaussian import (
    JointAnisotropicGaussianMechanism,
    group_l2_norm,
    whitened_sensitivity_squared,
)
from .dp_fedavg import DPUpdate
from .fixed_spectral_projection import FixedSpectralProjection
from .spectral_group_clipping import SpectralGroupClipper


class ProjectedJointSpectralDP:
    """Client-side DP release of one fixed projected spectral subspace."""

    aggregation_level = "local"
    requires_context = True
    client_side_release = True

    def __init__(
        self,
        projection: FixedSpectralProjection,
        clipper: SpectralGroupClipper,
        noise_mechanism: JointAnisotropicGaussianMechanism,
        *,
        group_sensitivities: Mapping[str, float] | None = None,
        record_private_diagnostics: bool = False,
    ) -> None:
        expected = set(GROUPS)
        if set(clipper.clip_config) != expected:
            raise ValueError("clipper must configure every spectral group")
        if set(noise_mechanism.clip_bounds) != expected:
            raise ValueError("noise mechanism must configure every spectral group")
        for group in GROUPS:
            if abs(
                clipper.clip_config[group] - noise_mechanism.clip_bounds[group]
            ) > 1e-12:
                raise ValueError("clipper and noise mechanism bounds must match")
        self.projection = projection
        self.clipper = clipper
        self.noise_mechanism = noise_mechanism
        sensitivity_source = group_sensitivities or clipper.clip_config
        if set(sensitivity_source) != expected:
            raise ValueError("group sensitivities must configure every spectral group")
        self._group_sensitivities = {
            group: float(sensitivity_source[group]) for group in GROUPS
        }
        if any(value <= 0 for value in self._group_sensitivities.values()):
            raise ValueError("group sensitivities must be positive")
        self.record_private_diagnostics = bool(record_private_diagnostics)
        self.privacy_group_metrics: list[dict[str, object]] = []

    @property
    def group_sensitivities(self) -> dict[str, float]:
        return dict(self._group_sensitivities)

    @property
    def group_noise_stds(self) -> dict[str, float]:
        return dict(self.noise_mechanism.noise_stds)

    def privatize(
        self,
        client_update: Mapping[str, torch.Tensor],
        round_idx: int = -1,
        client_id: int = -1,
    ) -> DPUpdate:
        spectral_groups = transform_update(client_update)
        projected_groups = self.projection.project_groups(spectral_groups)
        clipped_groups, clipping_stats = self.clipper.clip_groups(projected_groups)

        # The low-level mechanism treats all five blocks as one joint release.
        # Its second clipping pass is idempotent because the explicit clipper
        # above already enforces the same public bounds.
        noisy = self.noise_mechanism.privatize(
            clipped_groups, include_private_diagnostics=False
        )
        private_groups = self.projection.project_groups(noisy.groups)
        masks = self.projection.group_masks

        diagnostics: dict[str, float] = {
            "projection_ratio": self.projection.actual_retain_ratio,
            "released_coordinates": float(self.projection.retained_coordinates),
            "total_coordinates": float(self.projection.total_coordinates),
            "joint_whitened_sensitivity_squared": whitened_sensitivity_squared(
                self._group_sensitivities,self.noise_mechanism.noise_stds,
            ),
        }
        for stats in clipping_stats:
            group = stats.group_name
            released_coefficients = sum(
                int(mask.count_nonzero()) for mask in masks[group].values()
            )
            if self.record_private_diagnostics:
                effective_noise = {
                    name: private_groups[group][name] - clipped_groups[group][name]
                    for name in clipped_groups[group]
                }
                realized_noise_norm: float | str = group_l2_norm(effective_noise)
            else:
                # Revealing ||realized noise|| together with x+noise invalidates
                # the accounted Gaussian mechanism.  Formal artifacts expose
                # only public distribution parameters.
                realized_noise_norm = ""
            row: dict[str, object] = {
                "round": int(round_idx),
                "client_id": int(client_id),
                "group_name": group,
                "pre_clip_norm": (
                    stats.pre_clip_norm if self.record_private_diagnostics else ""
                ),
                "post_clip_norm": (
                    stats.post_clip_norm if self.record_private_diagnostics else ""
                ),
                "clip_bound": stats.clip_bound,
                "sensitivity": self._group_sensitivities[group],
                "clip_factor": (
                    stats.clip_factor if self.record_private_diagnostics else ""
                ),
                "was_clipped": (
                    stats.was_clipped if self.record_private_diagnostics else ""
                ),
                "noise_std": self.noise_mechanism.noise_stds[group],
                "noise_norm": realized_noise_norm,
                "expected_noise_squared_norm": (
                    released_coefficients
                    * self.noise_mechanism.noise_stds[group] ** 2
                ),
                "num_coefficients": released_coefficients,
            }
            self.privacy_group_metrics.append(row)
            diagnostics[f"{group}_clip_bound"] = stats.clip_bound
            diagnostics[f"{group}_noise_std"] = self.noise_mechanism.noise_stds[group]
            diagnostics[f"{group}_released_coordinates"] = float(
                released_coefficients
            )

        reconstructed = inverse_groups(private_groups, client_update)
        return DPUpdate(reconstructed, True, diagnostics)
