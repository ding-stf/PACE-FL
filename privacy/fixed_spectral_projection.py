from __future__ import annotations

"""Fixed, architecture-only projection of model updates in spectral space.

The projection mask is constructed once from parameter names and shapes.  It
never inspects parameter values, client data, gradients, losses, or spectral
magnitudes.  The default policy first preserves the public low-band and
``other`` coordinates, then fills the requested budget from the two high-band
groups using a seed-keyed, platform-independent hash ordering.

The projected model-shaped tensors returned by :meth:`project` are generally
dense because temporal coefficients are transformed back with an inverse DCT.
The zero guarantee applies to their spectral representation: transforming the
result again yields zero at every unreleased coordinate (up to floating-point
round-off in the DCT round trip).
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from spectral.spectral_groups import GROUPS, inverse_groups, transform_update


DEFAULT_PREFERRED_GROUPS = ("graph_low", "temporal_low", "other")
MASK_FORMAT_VERSION = 1


def _validate_retain_ratio(retain_ratio: float) -> float:
    ratio = float(retain_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("retain_ratio must be in (0, 1]")
    return ratio


def _architecture_groups(
    template: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Create group-shaped zeros without reading any template value."""

    if not template:
        raise ValueError("architecture template must contain at least one tensor")
    architecture_zeros = {
        name: torch.zeros(tuple(value.shape), dtype=torch.float32)
        for name, value in template.items()
    }
    return transform_update(architecture_zeros)


def _coordinate_score(seed: int, group: str, name: str, index: int) -> bytes:
    identity = f"{int(seed)}\0{group}\0{name}\0{index}".encode("utf-8")
    return hashlib.sha256(identity).digest()


class FixedSpectralProjection:
    """A reusable fixed mask for all clients and communication rounds.

    Parameters
    ----------
    architecture_template:
        A model state/update mapping.  Only keys and tensor shapes are used.
    retain_ratio:
        Requested fraction of all spectral coordinates.  The retained count is
        ``ceil(total_coordinates * retain_ratio)``.
    seed:
        Public seed used only to break coordinate-selection ties through a
        deterministic SHA-256 ordering.  It is not a DP randomness seed.
    preferred_groups:
        Public groups retained before the remaining budget is assigned to the
        other groups.  Defaults to low graph/temporal bands and ``other``.
    """

    def __init__(
        self,
        architecture_template: Mapping[str, torch.Tensor],
        retain_ratio: float = 0.90,
        seed: int = 42,
        preferred_groups: Sequence[str] = DEFAULT_PREFERRED_GROUPS,
    ) -> None:
        self.retain_ratio = _validate_retain_ratio(retain_ratio)
        self.seed = int(seed)
        preferred = tuple(dict.fromkeys(preferred_groups))
        unknown = set(preferred) - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown preferred groups: {sorted(unknown)}")
        self.preferred_groups = preferred

        architecture_groups = _architecture_groups(architecture_template)
        self._shapes = {
            group: {name: tuple(value.shape) for name, value in values.items()}
            for group, values in architecture_groups.items()
        }
        self.total_coordinates = sum(
            math.prod(shape)
            for values in self._shapes.values()
            for shape in values.values()
        )
        if self.total_coordinates <= 0:
            raise ValueError("architecture template has no spectral coordinates")
        self.retained_coordinates = min(
            self.total_coordinates,
            math.ceil(self.total_coordinates * self.retain_ratio),
        )
        self._masks = self._build_masks()
        self.mask_hash = self._compute_mask_hash()

    @property
    def actual_retain_ratio(self) -> float:
        return self.retained_coordinates / self.total_coordinates

    @property
    def group_masks(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return defensive CPU copies so callers cannot mutate the mask."""

        return {
            group: {name: mask.clone() for name, mask in values.items()}
            for group, values in self._masks.items()
        }

    def _candidates(self, groups: Sequence[str]) -> list[tuple[str, str, int]]:
        candidates: list[tuple[str, str, int]] = []
        for group in groups:
            for name in sorted(self._shapes[group]):
                candidates.extend(
                    (group, name, index)
                    for index in range(math.prod(self._shapes[group][name]))
                )
        return candidates

    def _select(
        self, candidates: Sequence[tuple[str, str, int]], count: int
    ) -> list[tuple[str, str, int]]:
        if count <= 0:
            return []
        if count >= len(candidates):
            return list(candidates)
        return sorted(
            candidates,
            key=lambda item: (_coordinate_score(self.seed, *item), item),
        )[:count]

    def _build_masks(self) -> dict[str, dict[str, torch.Tensor]]:
        masks = {
            group: {
                name: torch.zeros(shape, dtype=torch.bool)
                for name, shape in values.items()
            }
            for group, values in self._shapes.items()
        }
        preferred = self._candidates(self.preferred_groups)
        remaining_groups = tuple(
            group for group in GROUPS if group not in self.preferred_groups
        )
        remaining = self._candidates(remaining_groups)

        if self.retained_coordinates <= len(preferred):
            selected = self._select(preferred, self.retained_coordinates)
        else:
            selected = preferred + self._select(
                remaining, self.retained_coordinates - len(preferred)
            )
        if len(selected) != self.retained_coordinates:
            raise RuntimeError("failed to allocate the requested projection budget")
        for group, name, index in selected:
            masks[group][name].reshape(-1)[index] = True
        return masks

    def _compute_mask_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"fixed-spectral-mask-v{MASK_FORMAT_VERSION}\n".encode())
        for group in GROUPS:
            for name in sorted(self._masks[group]):
                mask = self._masks[group][name].reshape(-1)
                shape = ",".join(str(part) for part in self._shapes[group][name])
                digest.update(f"{group}\0{name}\0{shape}\0".encode("utf-8"))
                digest.update(bytes(mask.to(torch.uint8).tolist()))
        return digest.hexdigest()

    def _validate_groups(
        self, groups: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> None:
        if set(groups) != set(GROUPS):
            raise ValueError("spectral groups do not match the architecture template")
        for group in GROUPS:
            if set(groups[group]) != set(self._shapes[group]):
                raise ValueError(
                    f"tensor names in {group!r} do not match the architecture template"
                )
            for name, value in groups[group].items():
                if tuple(value.shape) != self._shapes[group][name]:
                    raise ValueError(
                        f"shape of {group}::{name} does not match the architecture template"
                    )

    def project_groups(
        self, groups: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Apply the cached mask; every omitted spectral coordinate is zero."""

        self._validate_groups(groups)
        return {
            group: {
                name: torch.where(
                    self._masks[group][name].to(device=value.device),
                    value,
                    torch.zeros_like(value),
                )
                for name, value in values.items()
            }
            for group, values in groups.items()
        }

    def project(
        self, tensors: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Transform, project, and reconstruct model-shaped tensors."""

        groups = transform_update(tensors)
        return inverse_groups(self.project_groups(groups), tensors)

    def save(self, path: str | Path) -> Path:
        """Persist ``spectral_mask.pt`` plus canonical reproducibility metadata."""

        destination = Path(path)
        if destination.suffix != ".pt":
            destination = destination / "spectral_mask.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": MASK_FORMAT_VERSION,
            "retain_ratio": self.retain_ratio,
            "actual_retain_ratio": self.actual_retain_ratio,
            "seed": self.seed,
            "preferred_groups": self.preferred_groups,
            "total_coordinates": self.total_coordinates,
            "retained_coordinates": self.retained_coordinates,
            "mask_hash": self.mask_hash,
            "group_masks": self.group_masks,
        }
        torch.save(payload, destination)
        return destination
