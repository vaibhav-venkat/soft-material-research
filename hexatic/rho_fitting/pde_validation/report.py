"""Text reports for PDE validation rollouts."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from .model import ValidationResult


REPORT_STEPS = (1, 2, 5, 10, 20, 50, 100)


def pde_validation_report_lines(
    *,
    case: str,
    cache_path: Path,
    results: dict[str, ValidationResult],
) -> list[str]:
    """Build tab-separated text lines summarizing validation metrics at fixed steps."""
    lines = [
        f"Rho fitting PDE validation report: {case}",
        "",
        f"cache: {cache_path}",
        "metrics: RMSE and R^2 at selected rollout steps",
        "",
    ]
    for mode, result in results.items():
        lines.extend(
            [
                f"[{mode}]",
                "step\tframe_index\ttime\trmse\tr2",
            ]
        )
        for step in REPORT_STEPS:
            frame_index = min(step, result.times.size - 1)
            lines.append(
                f"{step}\t{frame_index}\t{result.times[frame_index]:.10g}\t"
                f"{result.rmse_t[frame_index]:.10g}\t{result.r2_t[frame_index]:.10g}"
            )
        lines.append("")
    return lines


def write_pde_validation_report(
    path: Path,
    *,
    case: str,
    cache_path: Path,
    results: dict[str, ValidationResult],
    overwrite: bool,
) -> None:
    """Write the PDE validation text report through a temporary file."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(pde_validation_report_lines(case=case, cache_path=cache_path, results=results)) + "\n"
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
