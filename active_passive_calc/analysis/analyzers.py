from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.dtype[np.float64]]


def interpolate_periodic(
    values: npt.NDArray[np.float64],
    coords: npt.NDArray[np.float64],
    period: float,
    n_points: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if n_points is None:
        n_points = len(values)

    start = coords[0]
    new_grid = np.linspace(start, start + period, n_points, endpoint=False)

    ext_coords = np.append(coords, coords[0] + period)
    ext_values = np.append(values, values[0])

    interpolated_values = np.interp(new_grid, ext_coords, ext_values)

    return interpolated_values, new_grid


class ModeAnalyzer(ABC):
    @abstractmethod
    def extract(
        self,
        positions: np.ndarray,
        box: np.ndarray,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        pass

    @abstractmethod
    def compute_n_points(
        self, coords_sorted: npt.NDArray[np.float64], period: float
    ) -> int:
        pass

    @abstractmethod
    def compute_k(self, n_points: int, period: float) -> np.ndarray:
        pass


class Vesicle(ModeAnalyzer):
    def __init__(self, sigma_vertex: float):
        self.sigma_vertex = sigma_vertex

    def extract(
        self,
        positions: np.ndarray,
        box: np.ndarray,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        x_pos = positions[:, 0]
        y_pos = positions[:, 1]

        centered_x = x_pos - np.mean(x_pos)
        centered_y = y_pos - np.mean(y_pos)

        r = np.sqrt(centered_x**2 + centered_y**2)

        vals_sorted = r
        coords_sorted = np.linspace(0, 2 * np.pi, len(r), endpoint=False)
        period = 2 * np.pi

        return vals_sorted, coords_sorted, period

    def compute_n_points(
        self, coords_sorted: npt.NDArray[np.float64], period: float
    ) -> int:
        return len(coords_sorted)

    def compute_k(self, n_points: int, period: float) -> np.ndarray:
        a = 4 * self.sigma_vertex
        return np.fft.fftfreq(n_points, a)


class Membrane(ModeAnalyzer):
    def extract(
        self,
        positions: np.ndarray,
        box: np.ndarray,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        x_pos = positions[:, 0]
        period = float(box[1])

        vals_sorted = x_pos
        coords_sorted = np.linspace(0, period, len(x_pos), endpoint=False)

        return vals_sorted, coords_sorted, period

    def compute_n_points(
        self, coords_sorted: npt.NDArray[np.float64], period: float
    ) -> int:
        dy = np.diff(coords_sorted)
        positive_dy = dy[dy > 1e-6]
        if len(positive_dy) == 0:
            return len(coords_sorted)
        min_dy = np.min(positive_dy)
        return int(np.ceil(period / min_dy))

    def compute_k(self, n_points: int, period: float) -> np.ndarray:
        spacing = period / n_points
        return np.fft.fftfreq(n_points, spacing)


ANALYZERS: dict[str, type[ModeAnalyzer]] = {
    "vesicle": Vesicle,
    "membrane": Membrane,
}


def get_analyzer(mode: str, sigma_vertex: float) -> ModeAnalyzer:
    if mode == "vesicle":
        return Vesicle(sigma_vertex)
    if mode == "membrane":
        return Membrane()
    raise ValueError(f"Unknown mode: {mode}. Available: {list(ANALYZERS.keys())}")
