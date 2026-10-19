from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class ForecastingTotals:
    absolute_error: float = 0.0
    squared_error: float = 0.0
    absolute_percentage_error: float = 0.0
    valid_count: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> None:
        mask = torch.isfinite(pred) & torch.isfinite(target)
        if valid is not None:
            mask &= valid.bool()
        if not mask.any():
            return
        error = pred[mask] - target[mask]
        denominator = target[mask].abs().clamp_min(1e-5)
        self.absolute_error += float(error.abs().double().sum())
        self.squared_error += float(error.double().square().sum())
        self.absolute_percentage_error += float((error.abs() / denominator).double().sum())
        self.valid_count += int(mask.sum())

    def merge(self, other: "ForecastingTotals") -> None:
        self.absolute_error += other.absolute_error
        self.squared_error += other.squared_error
        self.absolute_percentage_error += other.absolute_percentage_error
        self.valid_count += other.valid_count

    def compute(self) -> dict[str, float]:
        if self.valid_count == 0:
            return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
        count = float(self.valid_count)
        return {
            "MAE": self.absolute_error / count,
            "RMSE": (self.squared_error / count) ** 0.5,
            "MAPE": self.absolute_percentage_error / count * 100.0,
        }

def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.bool() & torch.isfinite(pred) & torch.isfinite(target)
    if not mask.any():
        return (pred * 0.0).sum()
    return (pred[mask] - target[mask]).abs().mean()

def forecasting_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, float]:
    totals = ForecastingTotals()
    totals.update(pred, target, valid)
    return totals.compute()
