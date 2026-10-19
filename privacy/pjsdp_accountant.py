from __future__ import annotations

"""RDP accounting for repeated joint anisotropic spectral Gaussian releases.

This accountant intentionally omits client-sampling amplification.  The
``sampling_rate`` is accepted and recorded so experiment artifacts state the
selection protocol, but it does not reduce epsilon.  This is the conservative
default for a server that observes selected client identities and each
individual DP update.  Callers may pass a rigorously enforced public upper
bound on releases per client instead of the number of global protocol rounds;
that is bounded sequential composition, not hidden-subsampling amplification.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .anisotropic_gaussian import whitened_sensitivity_squared


def _default_orders() -> np.ndarray:
    return np.asarray(
        [
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            *range(2, 65),
            96,
            128,
            192,
            256,
            384,
            512,
            768,
            1024,
        ],
        dtype=float,
    )


def _validate_orders(orders: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(orders, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("orders must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(result)) or np.any(result <= 1.0):
        raise ValueError("all RDP orders must be finite and greater than one")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("RDP orders must be strictly increasing")
    return result


def _validate_delta(delta: float) -> float:
    value = float(delta)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("delta must be finite and in (0, 1)")
    return value


def _validate_sampling_rate(sampling_rate: float) -> float:
    value = float(sampling_rate)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("sampling_rate must be finite and in (0, 1]")
    return value


@dataclass(frozen=True)
class PJSDPEvent:
    rounds: int
    sampling_rate: float
    sampling_amplification: bool
    whitened_sensitivity_squared: float


@dataclass(frozen=True)
class NoiseCalibration:
    target_epsilon: float
    achieved_epsilon: float
    delta: float
    global_scale: float
    group_noise_stds: dict[str, float]
    rounds: int
    sampling_rate: float


class PJSDPAccountant:
    """Compose repeated joint Gaussian releases for one protected client."""

    def __init__(self, orders: Sequence[float] | np.ndarray | None = None) -> None:
        self.orders = _validate_orders(
            _default_orders() if orders is None else orders
        )
        self.rdp = np.zeros_like(self.orders)
        self.events: list[PJSDPEvent] = []

    def step(
        self,
        group_sensitivities: Mapping[str, float],
        group_noise_stds: Mapping[str, float],
        *,
        rounds: int = 1,
        sampling_rate: float = 1.0,
        sampling_amplification: bool = False,
    ) -> None:
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
            raise ValueError("rounds must be a positive integer")
        rate = _validate_sampling_rate(sampling_rate)
        if sampling_amplification:
            raise NotImplementedError(
                "subsampled Gaussian amplification is not implemented; "
                "use sampling_amplification=False"
            )
        # ``group_sensitivities`` already contains the adjacency-specific
        # bound (C_b for add/remove or 2*C_b for replacement).
        squared = whitened_sensitivity_squared(
            group_sensitivities, group_noise_stds
        )
        self.rdp += rounds * self.orders * squared / 2.0
        self.events.append(PJSDPEvent(rounds, rate, False, squared))

    def epsilon(self, delta: float) -> float:
        value = _validate_delta(delta)
        epsilons = self.rdp + math.log(1.0 / value) / (self.orders - 1.0)
        return float(np.min(epsilons))

    @classmethod
    def calibrate_global_scale(
        cls,
        target_epsilon: float,
        delta: float,
        rounds: int,
        group_sensitivities: Mapping[str, float],
        noise_weights: Mapping[str, float],
        *,
        sampling_rate: float = 1.0,
        sampling_amplification: bool = False,
        orders: Sequence[float] | np.ndarray | None = None,
        iterations: int = 100,
    ) -> NoiseCalibration:
        """Calibrate ``s_b = global_scale * noise_weights[b]``.

        ``noise_weights`` and sensitivities must be public and fixed before the
        formal run.  The returned ``achieved_epsilon`` is recomputed from the
        final absolute standard deviations; callers must log it separately from
        ``target_epsilon``.
        """

        target = float(target_epsilon)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("target_epsilon must be finite and positive")
        _validate_delta(delta)
        rate = _validate_sampling_rate(sampling_rate)
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        template = cls(orders)
        privacy_floor = float(
            np.min(math.log(1.0 / float(delta)) / (template.orders - 1.0))
        )
        if target <= privacy_floor:
            raise ValueError(
                "target_epsilon is below the finite-order accountant floor; "
                "supply a larger order grid"
            )

        # Validate keys and positive weights through the common sensitivity
        # routine.  Unit scale is arbitrary and used only for validation here.
        whitened_sensitivity_squared(group_sensitivities, noise_weights)

        def evaluate(scale: float) -> float:
            stds = {group: float(scale) * float(weight) for group, weight in noise_weights.items()}
            accountant = cls(orders)
            accountant.step(
                group_sensitivities,
                stds,
                rounds=rounds,
                sampling_rate=rate,
                sampling_amplification=sampling_amplification,
            )
            return accountant.epsilon(delta)

        low = 0.0
        high = 1.0
        while evaluate(high) > target:
            high *= 2.0
            if not math.isfinite(high) or high > 1e15:
                raise RuntimeError("failed to bracket a valid Gaussian noise scale")

        for _ in range(iterations):
            midpoint = (low + high) / 2.0
            if evaluate(midpoint) > target:
                low = midpoint
            else:
                high = midpoint

        stds = {
            group: high * float(weight) for group, weight in noise_weights.items()
        }
        achieved = evaluate(high)
        return NoiseCalibration(
            target,
            achieved,
            float(delta),
            high,
            stds,
            rounds,
            rate,
        )
