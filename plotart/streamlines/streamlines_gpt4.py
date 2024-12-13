from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Union

import numpy as np
from scipy import interpolate


def point_distance(p1: np.ndarray, p2: np.ndarray) -> float:  # pylint: disable=C0103
    """Calculate Euclidean distance between two points.

    Args:
        p1 (np.ndarray): First point.
        p2 (np.ndarray): Second point.

    Returns:
        float: Euclidean distance between two points.
    """
    return np.linalg.norm(p2 - p1)


def check_sink_proximity(  # pylint: disable=C0103
    p: np.ndarray, sinks: List[np.ndarray], maxdist: float = 0.1
) -> bool:
    """Check if point p is within maxdist of any sink.

    Args:
        p (np.ndarray): Point to check.
        sinks (List[np.ndarray]): List of sink points.
        maxdist (float, optional): Maximum distance to consider a point close to a sink. Defaults to 0.1.

    Returns:
        bool: True if point p is close to any sink, False otherwise.
    """
    return any(point_distance(p, sink) < maxdist for sink in sinks)


def compute_trace(  # pylint: disable=C0103
    p: np.ndarray,
    maxiter: int,
    max_sink_dist: float,
    grid: Dict[str, Union[float, int]],
    u_i: Callable,
    v_i: Callable,
    dt: float,
    sinks: List[np.ndarray],
    limit: int = 1500,
    max_iter_sink_proximity: int = 0,
) -> List[np.ndarray]:
    """Compute a streamline trace starting from point p.

    Args:
        p (np.ndarray): Starting point.
        maxiter (int): Maximum number of iterations.
        max_sink_dist (float): Maximum distance to consider a point close to a sink.
        grid (Dict[str, Union[float, int]]): Grid parameters.
        u_i (callable): Function to compute horizontal component of velocity at a point.
        v_i (callable): Function to compute vertical component of velocity at a point.
        dt (float): Time step for streamline computation.
        sinks (List[np.ndarray]): List of sink points.
        limit (int, optional): Limit on length of trace. Defaults to 1500.
        max_iter_sink_proximity (int, optional): Maximum iterations when close to a sink. Defaults to 0.

    Returns:
        List[np.ndarray]: List of points along the computed streamline trace.
    """
    ps = np.empty((maxiter + 1, 2))
    n = 0
    sink_soak = 0
    while (
        point_in_canvas(p, grid)
        and n <= maxiter
        and sink_soak <= max_iter_sink_proximity
    ):
        ps[n] = p
        dx, dy = u_i(p[0], p[1])[0] * dt, v_i(p[0], p[1])[0] * dt
        p += np.array([dx, dy])
        n += 1
        sink_soak += check_sink_proximity(p, sinks, maxdist=max_sink_dist)

    return ps[: n : max(1, n // limit), :]


def point_in_canvas(point: np.ndarray, grid: Dict[str, Union[float, int]]) -> bool:
    """Check if a point is inside the grid canvas.

    Args:
        point (np.ndarray): Point to check.
        grid (Dict[str, Union[float, int]]): Grid parameters.

    Returns:
        bool: True if point is inside canvas, False otherwise.
    """
    x, y = point  # pylint: disable=C0103
    return (
        grid["x_start"] <= x <= grid["x_end"] and grid["y_start"] <= y <= grid["y_end"]
    )


class StreamLines:
    """
    A class used to represent the Streamlines in a grid.

    ...

    Attributes
    ----------
    x, y, dx, dy, X, Y, psis, streamtraces, source_sink, vortices, seeds : list
        lists to store respective attributes.
    sl_config : dict
        dictionary to store the streamlines configuration.
    grid : dict
        grid dictionary passed as a parameter.
    psi, u, v : np.array
        numpy arrays to store stream function and velocities respectively.

    Methods
    -------
    generate_mesh():
        Generates the grid mesh.
    add_source_sink(strength: float, x: float, y: float):
        Adds a source or a sink to the grid.
    add_free_stream(u_inf=0, v_inf=0):
        Adds a free stream to the grid.
    add_vortex(strength: float, x: float, y: float):
        Adds a vortex to the grid.
    reset_psi():
        Resets the stream function.
    _circle_points(p: List[float], r: float, n_points: int):
        Returns points in a circular arc.
    calc_streamtraces(
        n_streamtraces: int, dt: float, maxiter: int, radius: float,
        max_sink_dist: float, max_iter_sink_proximity: int, seeds: Union[str, List[float]],
        n_cpu: int):
        Calculates the streamlines.
    """

    def __init__(self, grid: Dict[str, Union[int, float]]):
        self.x, self.y, self.dx, self.dy = [], [], [], []  # pylint: disable=C0103
        self.X, self.Y, self.psis, self.streamtraces = (
            [],
            [],
            [],
            [],
        )  # pylint: disable=C0103
        self.source_sink, self.vortices, self.seeds, self.sl_config = [], [], [], {}
        self.grid = grid
        grid_shape = (grid["no_points_x"], grid["no_points_y"])
        self.psi, self.u, self.v = (  # pylint: disable=C0103
            np.zeros(grid_shape),
            np.zeros(grid_shape),
            np.zeros(grid_shape),
        )
        self.generate_mesh()

    def generate_mesh(self):
        """
        Generates the grid mesh.
        """
        grid = self.grid
        self.x, self.y = np.linspace(
            grid["x_start"], grid["x_end"], grid["no_points_x"]
        ), np.linspace(grid["y_start"], grid["y_end"], grid["no_points_y"])
        self.dx, self.dy = (grid["x_end"] - grid["x_start"]) / grid["no_points_x"], (
            grid["y_end"] - grid["y_start"]
        ) / grid["no_points_y"]
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def add_source_sink(
        self, strength: float, x: float, y: float
    ):  # pylint: disable=C0103
        """
        Adds a source or a sink to the grid.

        Parameters
        ----------
        strength : float
            Strength of the source/sink.
        x : float
            x-coordinate of the source (or sink).
        y : float
            y-coordinate of the source (or sink).
        """
        self.source_sink.append([x, y, strength])
        psi = strength / (2 * np.pi) * np.arctan2((self.Y - y), (self.X - x))
        self.psis.append(psi)
        self.psi += psi
        self.u += (
            strength
            / (2 * np.pi)
            * (self.X - x)
            / ((self.X - x) ** 2 + (self.Y - y) ** 2)
        )
        self.v += (
            strength
            / (2 * np.pi)
            * (self.Y - y)
            / ((self.X - x) ** 2 + (self.Y - y) ** 2)
        )

    def add_free_stream(
        self, u_inf: float = 0, v_inf: float = 0
    ):  # pylint: disable=C0103
        """
        Adds a free stream to the grid.

        Parameters
        ----------
        u_inf : float, optional
            Stream velocity in x direction (default is 0).
        v_inf : float, optional
            Stream velocity in y direction (default is 0).
        """
        psi = u_inf * self.Y - v_inf * self.X
        self.psis.append(psi)
        self.psi += psi
        self.u += u_inf
        self.v += v_inf

    def add_vortex(self, strength: float, x: float, y: float):  # pylint: disable=C0103
        """
        Adds a vortex to the grid.

        Parameters
        ----------
        strength : float
            Strength of the vortex.
        x : float
            x-coordinate of the vortex.
        y : float
            y-coordinate of the vortex.
        """
        self.vortices.append([x, y, strength])
        psi = strength / (4 * np.pi) * np.log((self.X - x) ** 2 + (self.Y - y) ** 2)
        self.psis.append(psi)
        self.psi += psi
        self.u += (
            strength
            / (2 * np.pi)
            * (self.Y - y)
            / ((self.X - x) ** 2 + (self.Y - y) ** 2)
        )
        self.v -= (
            strength
            / (2 * np.pi)
            * (self.X - x)
            / ((self.X - x) ** 2 + (self.Y - y) ** 2)
        )

    def reset_psi(self):
        """
        Resets the stream function.
        """
        self.psi = sum(self.psis)

    def _circle_points(  # pylint: disable=C0103
        self, p: List[float], r: float, n_points: int
    ) -> List[List[float]]:
        """
        Returns points in a circular arc.

        Parameters
        ----------
        p : list
            Center of the circle.
        r : float
            Radius of the circle.
        n_points : int
            Number of points in the circle.

        Returns
        -------
        list
            List of points in the circle.
        """
        return [
            [p[0] + np.cos(alpha) * r, p[1] + np.sin(alpha) * r]
            for alpha in np.linspace(0, 2 * np.pi, n_points + 1)
        ]

    def calc_streamtraces(  # pylint: disable=C0103
        self,
        n_streamtraces: int = 10,
        dt: float = 0.005,
        maxiter: int = 500,
        radius: float = 0.1,
        max_sink_dist: float = 0.1,
        max_iter_sink_proximity: int = 0,
        seeds: Union[str, List[float]] = ["random"],
        n_cpu: int = 1,
    ) -> None:
        """
        Calculates the streamlines.

        Parameters
        ----------
        n_streamtraces : int, optional
            Number of streamtraces (default is 10).
        dt : float, optional
            Time step (default is 0.005).
        maxiter : int, optional
            Maximum number of iterations (default is 500).
        radius : float, optional
            Radius (default is 0.1).
        max_sink_dist : float, optional
            Maximum sink distance (default is 0.1).
        max_iter_sink_proximity : int, optional
            Maximum sink proximity iterations (default is 0).
        seeds : str or list, optional
            Seeds (default is "random").
        n_cpu : int, optional
            Number of CPUs (default is 1).
        """
        self.sl_config = {
            "n_streamtraces": n_streamtraces,
            "dt": dt,
            "maxiter": maxiter,
            "radius": radius,
            "max_sink_dist": max_sink_dist,
            "seeds": seeds,
        }

        u_i = interpolate.interp2d(self.x, self.y, self.u)
        v_i = interpolate.interp2d(self.x, self.y, self.v)

        if not isinstance(seeds, list):
            raise ValueError("seeds argument must be list")

        seed_handlers = {
            "random": self._handle_random_seeds,
            "sources": self._handle_sources_seeds,
            "grid": self._handle_grid_seeds,
        }

        sinks = [[x[0], x[1]] for x in self.source_sink if x[2] < 0]
        for item in seeds:
            seed_handler = seed_handlers.get(item)
            if seed_handler:
                seed_handler(n_streamtraces, radius)

        compute_trace_partial = partial(
            compute_trace,
            maxiter=maxiter,
            max_sink_dist=max_sink_dist,
            max_iter_sink_proximity=max_iter_sink_proximity,
            sinks=sinks,
            grid=self.grid,
            u_i=u_i,
            v_i=v_i,
            dt=dt,
        )

        if n_cpu > 1:
            with Pool(n_cpu) as pool:
                self.streamtraces = pool.map(compute_trace_partial, self.seeds)
        else:
            self.streamtraces = list(map(compute_trace_partial, self.seeds))

    def _handle_random_seeds(self, n_streamtraces: int):
        xs = np.random.uniform(  # pylint: disable=C0103
            self.grid["x_start"], self.grid["x_end"], size=n_streamtraces
        )
        ys = np.random.uniform(  # pylint: disable=C0103
            self.grid["y_start"], self.grid["y_end"], size=n_streamtraces
        )
        self.seeds.extend(np.dstack((xs, ys))[0].tolist())

    def _handle_sources_seeds(self, n_streamtraces: int, radius: float):
        seeds_center = [[x[0], x[1]] for x in self.source_sink if x[2] > 0]
        n_streamstraces_per_source = int(n_streamtraces / len(seeds_center))
        for p in seeds_center:  # pylint: disable=C0103
            self.seeds.extend(
                self._circle_points(p, radius, n_streamstraces_per_source)
            )

    def _handle_grid_seeds(self, n_streamtraces: int):
        grid_points = np.dstack((self.X.flatten(), self.Y.flatten()))[0]
        gridskip = int(
            self.grid["no_points_x"] * self.grid["no_points_y"] / n_streamtraces
        )
        self.seeds.extend(grid_points[::gridskip].tolist())
