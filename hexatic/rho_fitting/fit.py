"""Workflow orchestration for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RhoFittingConfig


@dataclass(frozen=True)
class RhoFittingResult:
    case_id: str
    status: str
    nd: int

    def summary(self) -> str:
        return f"[rho_fitting] case={self.case_id} status={self.status} nd={self.nd}"


def run(config: RhoFittingConfig) -> RhoFittingResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return RhoFittingResult(
        case_id=config.case_id,
        status="setup-ready",
        nd=config.settings.nd,
    )
