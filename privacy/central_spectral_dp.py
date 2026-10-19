from __future__ import annotations

"""Secure-aggregation compatible central spectral DP.

Clients locally apply only public, fixed spectral-group clipping.  A secure
aggregation service reveals the *mean* clipped update to the coordinator; the
coordinator then adds Gaussian noise once.  This is intentionally distinct
from local-update DP, where noise is injected into every client update.
"""

from collections.abc import Mapping, Sequence

import torch

from spectral.spectral_groups import GROUPS, inverse_groups, transform_update

from .clipping import clip_group
from .dp_fedavg import DPUpdate


class CentralSpectralDPMechanism:
    """Client-level central DP for a secure aggregate of clipped updates.

    The privacy unit is add/remove of one selected client.  For a round with
    ``m`` participants, the sensitivity of a released group mean is ``C_b/m``;
    noise is therefore ``sigma * C_b / m`` per coefficient.
    """

    aggregation_level = "central"

    def __init__(
        self,
        clip_norms: Mapping[str, float],
        noise_multiplier: float,
        generator: torch.Generator | None = None,
    ) -> None:
        missing = set(GROUPS) - set(clip_norms)
        if missing:
            raise ValueError(f"missing clip norms: {sorted(missing)}")
        if any(value <= 0 for value in clip_norms.values()):
            raise ValueError("all clip norms must be positive")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        self.clip_norms = dict(clip_norms)
        self.noise_multiplier = noise_multiplier
        self.generator = generator

    @property
    def composed_group_count(self) -> int:
        return len(GROUPS)

    def clip_for_secure_aggregation(
        self, update: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        groups = transform_update(update)
        clipped = {
            group: clip_group(tensors, self.clip_norms[group])[0]
            for group, tensors in groups.items()
        }
        return inverse_groups(clipped, update)

    def privatize_secure_mean(
        self, clipped_updates: Sequence[Mapping[str, torch.Tensor]]
    ) -> DPUpdate:
        if not clipped_updates:
            raise ValueError("secure aggregation requires at least one update")
        count = len(clipped_updates)
        mean = {
            name: sum(update[name] for update in clipped_updates) / count
            for name in clipped_updates[0]
        }
        groups = transform_update(mean)
        private: dict[str, dict[str, torch.Tensor]] = {}
        diagnostics: dict[str, float] = {"secure_aggregation_clients": float(count)}
        for group, tensors in groups.items():
            std = self.noise_multiplier * self.clip_norms[group] / count
            private[group] = {
                name: value
                + torch.randn(
                    value.shape,
                    dtype=value.dtype,
                    device=value.device,
                    generator=self.generator,
                )
                * std
                for name, value in tensors.items()
            }
            diagnostics[f"{group}_central_noise_std"] = std
        return DPUpdate(inverse_groups(private, mean), True, diagnostics)
