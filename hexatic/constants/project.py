from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path = Path(__file__).resolve().parents[1]
    output_dir: Path = project_dir / "output"
    sphere_output_dir: Path = output_dir / "sphere"
    cylinder_output_dir: Path = output_dir / "cylinder"
    image_output_dir: Path = output_dir / "images"


PATHS = ProjectPaths()

PROJECT_DIR = PATHS.project_dir
OUTPUT_DIR = PATHS.output_dir
SPHERE_OUTPUT_DIR = PATHS.sphere_output_dir
CYLINDER_OUTPUT_DIR = PATHS.cylinder_output_dir
IMAGE_OUTPUT_DIR = PATHS.image_output_dir
