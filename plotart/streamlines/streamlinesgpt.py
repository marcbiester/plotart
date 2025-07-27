from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Union

import numpy as np
import numba

@numba.njit(inline='always')
def squared_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return dx*dx + dy*dy

@numba.njit(inline='always')
def point_in_canvas(point: np.ndarray, x_start: float, x_end: float, y_start: float, y_end: float) -> bool:
    return (x_start <= point[0] <= x_end) and (y_start <= point[1] <= y_end)

@numba.njit
def check_sink_proximity(p: np.ndarray, sinks: np.ndarray, maxdist: float) -> int:
    # Returns 1 if close to a sink, else 0
    # Compare squared distances to avoid sqrt
    maxdist2 = maxdist * maxdist
    for i in range(sinks.shape[0]):
        if squared_distance(p, sinks[i]) < maxdist2:
            return 1
    return 0

@numba.njit
def bilinear_interpolate(x_arr: np.ndarray, y_arr: np.ndarray, F: np.ndarray, x: float, y: float) -> float:
    # Bilinear interpolation on a structured grid.
    # Assumes x_arr and y_arr are sorted.
    # Find indices i, j such that x_arr[i] <= x < x_arr[i+1], y_arr[j] <= y < y_arr[j+1].
    
    # Binary search for i in x direction
    i = np.searchsorted(x_arr, x)
    if i == 0:
        i = 0
    elif i >= x_arr.size:
        i = x_arr.size - 2
    else:
        i -= 1
    
    # Binary search for j in y direction
    j = np.searchsorted(y_arr, y)
    if j == 0:
        j = 0
    elif j >= y_arr.size:
        j = y_arr.size - 2
    else:
        j -= 1

    x1, x2 = x_arr[i], x_arr[i+1]
    y1, y2 = y_arr[j], y_arr[j+1]

    Q11 = F[j, i]
    Q12 = F[j+1, i]
    Q21 = F[j, i+1]
    Q22 = F[j+1, i+1]

    denom = (x2 - x1)*(y2 - y1)
    if denom == 0.0:
        return Q11  # Degenerate case

    return (
        Q11 * (x2 - x) * (y2 - y) +
        Q21 * (x - x1) * (y2 - y) +
        Q12 * (x2 - x) * (y - y1) +
        Q22 * (x - x1) * (y - y1)
    ) / denom

@numba.njit
def compute_trace_numba(
    p: np.ndarray,
    maxiter: int,
    max_sink_dist: float,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dt: float,
    sinks: np.ndarray,
    x_start: float, x_end: float,
    y_start: float, y_end: float,
    limit: int,
    max_iter_sink_proximity: int
) -> np.ndarray:
    ps = np.empty((maxiter + 1, 2), dtype=np.float64)
    n = 0
    sink_soak = 0

    for iteration in range(maxiter + 1):
        if not point_in_canvas(p, x_start, x_end, y_start, y_end):
            break
        if sink_soak > max_iter_sink_proximity:
            break

        ps[n, 0] = p[0]
        ps[n, 1] = p[1]

        # Interpolate velocities
        u_val = bilinear_interpolate(x_arr, y_arr, u, p[0], p[1])
        v_val = bilinear_interpolate(x_arr, y_arr, v, p[0], p[1])

        p[0] += u_val * dt
        p[1] += v_val * dt
        sink_soak += check_sink_proximity(p, sinks, max_sink_dist)
        n += 1

    # Slice and downsample (if needed)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
    step = max(1, n // limit)
    return ps[:n:step, :]

class StreamLines:
    def __init__(self, grid: Dict[str, Union[int, float]]):
        self.x, self.y, self.dx, self.dy = [], [], [], []
        self.X, self.Y, self.psis, self.streamtraces = [], [], [], []
        self.source_sink, self.vortices, self.seeds, self.sl_config = [], [], [], {}
        self.grid = grid
        grid_shape = (grid["no_points_y"], grid["no_points_x"])
        self.psi = np.zeros(grid_shape)
        self.u = np.zeros(grid_shape)
        self.v = np.zeros(grid_shape)
        self.generate_mesh()

    def generate_mesh(self):
        grid = self.grid
        self.x = np.linspace(grid["x_start"], grid["x_end"], grid["no_points_x"])
        self.y = np.linspace(grid["y_start"], grid["y_end"], grid["no_points_y"])
        self.dx = (grid["x_end"] - grid["x_start"]) / grid["no_points_x"]
        self.dy = (grid["y_end"] - grid["y_start"]) / grid["no_points_y"]
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def add_source_sink(self, strength: float, x: float, y: float):
        self.source_sink.append([x, y, strength])
        psi = strength / (2 * np.pi) * np.arctan2((self.Y - y), (self.X - x))
        self.psis.append(psi)
        self.psi += psi
        self.u += (strength/(2*np.pi))*((self.X - x)/((self.X - x)**2+(self.Y - y)**2))
        self.v += (strength/(2*np.pi))*((self.Y - y)/((self.X - x)**2+(self.Y - y)**2))

    def add_free_stream(self, u_inf: float = 0, v_inf: float = 0):
        psi = u_inf * self.Y - v_inf * self.X
        self.psis.append(psi)
        self.psi += psi
        self.u += u_inf
        self.v += v_inf

    def add_vortex(self, strength: float, x: float, y: float):
        self.vortices.append([x, y, strength])
        psi = strength/(4*np.pi)*np.log((self.X - x)**2+(self.Y - y)**2)
        self.psis.append(psi)
        self.psi += psi
        self.u += (strength/(2*np.pi))*((self.Y - y)/((self.X - x)**2+(self.Y - y)**2))
        self.v -= (strength/(2*np.pi))*((self.X - x)/((self.X - x)**2+(self.Y - y)**2))

    def reset_psi(self):
        self.psi = sum(self.psis)

    def _circle_points(self, p: List[float], r: float, n_points: int) -> List[List[float]]:
        return [
            [p[0] + np.cos(alpha)*r, p[1] + np.sin(alpha)*r]
            for alpha in np.linspace(0, 2*np.pi, n_points+1)
        ]

    def calc_streamtraces(
        self,
        n_streamtraces: int = 10,
        dt: float = 0.005,
        maxiter: int = 500,
        radius: float = 0.1,
        max_sink_dist: float = 0.1,
        max_iter_sink_proximity: int = 0,
        seeds: Union[str, List[float]] = ["random"],
        n_cpu: int = 1,
        limit: int = 1500
    ) -> None:

        self.sl_config = {
            "n_streamtraces": n_streamtraces,
            "dt": dt,
            "maxiter": maxiter,
            "radius": radius,
            "max_sink_dist": max_sink_dist,
            "seeds": seeds,
        }

        if not isinstance(seeds, list):
            raise ValueError("seeds argument must be a list")

        seed_handlers = {
            "random": self._handle_random_seeds,
            "sources": self._handle_sources_seeds,
            "grid": self._handle_grid_seeds,
        }

        sinks = np.array([[x[0], x[1]] for x in self.source_sink if x[2] < 0], dtype=np.float64)

        # Generate seeds if strings provided
        for item in seeds:
            if isinstance(item, str):
                seed_handler = seed_handlers.get(item)
                if seed_handler:
                    seed_handler(n_streamtraces, radius)
            else:
                # item assumed to be a coordinate pair
                self.seeds.append(item)

        # Convert seeds to numpy array
        seeds_arr = np.array(self.seeds, dtype=np.float64)

        # Prepare partial function
        args = (
            self.grid["x_start"],
            self.grid["x_end"],
            self.grid["y_start"],
            self.grid["y_end"],
        )

        # Using multiprocessing pool if requested
        if n_cpu > 1:
            with Pool(n_cpu) as pool:
                results = pool.map(
                    partial(self._compute_trace_wrapper,
                            maxiter=maxiter,
                            max_sink_dist=max_sink_dist,
                            dt=dt,
                            sinks=sinks,
                            limit=limit,
                            max_iter_sink_proximity=max_iter_sink_proximity),
                    seeds_arr
                )
        else:
            results = [
                self._compute_trace_wrapper(s,
                                            maxiter=maxiter,
                                            max_sink_dist=max_sink_dist,
                                            dt=dt,
                                            sinks=sinks,
                                            limit=limit,
                                            max_iter_sink_proximity=max_iter_sink_proximity)
                for s in seeds_arr
            ]

        self.streamtraces = results

    def _compute_trace_wrapper(self, seed: np.ndarray, maxiter: int, max_sink_dist: float, dt: float,
                               sinks: np.ndarray, limit: int, max_iter_sink_proximity: int):
        return compute_trace_numba(
            seed.copy(),
            maxiter,
            max_sink_dist,
            self.x, self.y,
            self.u, self.v,
            dt,
            sinks,
            self.grid["x_start"], self.grid["x_end"],
            self.grid["y_start"], self.grid["y_end"],
            limit,
            max_iter_sink_proximity
        )

    def _handle_random_seeds(self, n_streamtraces: int, radius: float):
        xs = np.random.uniform(self.grid["x_start"], self.grid["x_end"], size=n_streamtraces)
        ys = np.random.uniform(self.grid["y_start"], self.grid["y_end"], size=n_streamtraces)
        self.seeds.extend(np.column_stack((xs, ys)).tolist())

    def _handle_sources_seeds(self, n_streamtraces: int, radius: float):
        seeds_center = [[x[0], x[1]] for x in self.source_sink if x[2] > 0]
        if len(seeds_center) == 0:
            return
        n_streamstraces_per_source = max(1, int(n_streamtraces / len(seeds_center)))
        for p in seeds_center:
            self.seeds.extend(self._circle_points(p, radius, n_streamstraces_per_source))

    def _handle_grid_seeds(self, n_streamtraces: int):
        grid_points = np.column_stack((self.X.flatten(), self.Y.flatten()))
        step = max(1, int(self.X.size / n_streamtraces))
        self.seeds.extend(grid_points[::step].tolist())
