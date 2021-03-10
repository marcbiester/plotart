import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from multiprocessing import Pool
from functools import partial
import functools


def point_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def check_sink_proximity(p, sinks, maxdist=0.1):
    f = functools.partial(point_distance, p2=p)
    dist = list(map(f, sinks))
    return any(x < maxdist for x in dist)


def compute_trace(p, maxiter, max_sink_dist, grid, u_i, v_i, dt, sinks, limit=1500):
    ps = []
    n = 0
    sink_soak = 0
    while (
        point_in_canvas(p, grid)
        and n <= maxiter
        and sink_soak < 30  # allow only 20 more steps if getting close to sink
    ):
        ps.append(p.copy())
        dx = u_i(p[0], p[1])[0] * dt
        dy = v_i(p[0], p[1])[0] * dt
        p[0] = p[0] + dx
        p[1] = p[1] + dy
        n += 1
        if check_sink_proximity(p, sinks, maxdist=max_sink_dist):
            sink_soak += 1
    # limit length of points returned
    skips = max(1, int(len(ps) / limit))
    return ps[::skips]


def point_in_canvas(point, grid):
    if (
        point[0] < grid["x_start"]
        or point[0] > grid["x_end"]
        or point[1] < grid["y_start"]
        or point[1] > grid["y_end"]
    ):
        return False
    else:
        return True


class streamLines:
    def __init__(self, grid):
        self.x = []
        self.y = []
        self.dx = []
        self.dy = []
        self.X = []
        self.Y = []
        self.psis = []
        self.streamtraces = []
        self.source_sink = []
        self.seeds = []
        self.grid = grid
        grid_shape = (grid["no_points_x"], grid["no_points_y"])
        self.psi = np.zeros(grid_shape)
        self.u = np.zeros(grid_shape)
        self.v = np.zeros(grid_shape)
        self.generate_mesh()

    def generate_mesh(self):
        self.x = np.linspace(
            self.grid["x_start"], self.grid["x_end"], self.grid["no_points_x"]
        )
        self.y = np.linspace(
            self.grid["y_start"], self.grid["y_end"], self.grid["no_points_y"]
        )
        self.dx = (self.grid["x_end"] - self.grid["x_start"]) / self.grid["no_points_x"]
        self.dy = (self.grid["y_end"] - self.grid["y_start"]) / self.grid["no_points_y"]
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def add_source_sink(self, strength, x, y):
        """
        Returns the stream-function generated by a source/sink.

        Parameters
        ----------
        strength: float
            Strength of the source/sink.
        x: float
            x-coordinate of the source (or sink).
        y: float
            y-coordinate of the source (or sink).

        Returns
        -------
        psi: 2D Numpy array of floats
            The stream-function.
        """
        self.source_sink.append([x, y, strength])
        psi = strength / (2 * np.pi) * np.arctan2((self.Y - y), (self.X - x))
        self.psis.append(psi)
        self.psi = np.add(self.psi, psi)
        self.u = self.u + (
            strength
            / (2 * np.pi)
            * (self.X - x)
            / ((self.X - x) ** 2 + (self.Y - y) ** 2)
        )
        self.v = self.v + (
            strength
            / (2 * np.pi)
            * (self.Y - y)
            / ((self.X - x) ** 2 + (self.Y - y) ** 2)
        )

    def add_free_stream(self, u_inf=0, v_inf=0):
        psi = u_inf * self.Y - v_inf * self.X
        self.psis.append(psi)
        self.psi = np.add(self.psi, psi)
        self.u = self.u + u_inf
        self.v = self.v + v_inf

    def add_doublet(self, strength, x, y):
        """
        Returns the stream-function generated by a doublet.

        Parameters
        ----------
        strength: float
            Strength of the doublet.
        x: float
            x-coordinate of the doublet.
        y: float
            y-coordinate of the doublet.

        Returns
        -------
        psi: 2D Numpy array of floats
            The stream-function.
        """
        psi = (
            -strength
            / (2 * np.pi)
            * (self.Y - y)
            / ((self.X - x) ** 2 + (self.Y - y) ** 2)
        )
        self.psis.append(psi)
        self.psi = np.add(self.psi, psi)
        self.u = self.u - (
            strength
            / (2 * np.pi)
            * ((self.X - x) ** 2 - (self.Y - y) ** 2)
            / ((self.X - x) ** 2 + (self.Y - y) ** 2) ** 2
        )
        self.v = self.v - (
            strength
            / (2 * np.pi)
            * 2
            * (self.X - x)
            * (self.Y - y)
            / ((self.X - x) ** 2 + (self.Y - y) ** 2) ** 2
        )

    def add_vortex(self, strength, x, y):
        """
        Returns the velocity field generated by a vortex.

        Parameters
        ----------
        strength: float
            Strength of the vortex.
        xv: float
            x-coordinate of the vortex.
        yv: float
            y-coordinate of the vortex.
        X: 2D Numpy array of floats
            x-coordinate of the mesh points.
        Y: 2D Numpy array of floats
            y-coordinate of the mesh points.
        """
        psi = strength / (4 * np.pi) * np.log((self.X - x) ** 2 + (self.Y - y) ** 2)
        self.psis.append(psi)
        self.psi = np.add(self.psi, psi)
        self.u = self.u + strength / (2 * np.pi) * (self.Y - y) / (
            (self.X - x) ** 2 + (self.Y - y) ** 2
        )
        self.v = self.v - strength / (2 * np.pi) * (self.X - x) / (
            (self.X - x) ** 2 + (self.Y - y) ** 2
        )

    def reset_psi(self):
        self.psi = self.psis[0]
        if len(self.psis > 0):
            for i in range(1, len(self.psis)):
                self.psi = np.add(self.psi, self.psis[i])

    def _circle_points(self, p, r, n_points):
        points = []
        for alpha in np.linspace(0, 2 * np.pi, n_points + 1):
            points.append([p[0] + np.cos(alpha) * r, p[1] + np.sin(alpha) * r])
        return points

    def calc_streamtraces(
        self,
        n_streamtraces=10,
        dt=0.005,
        maxiter=500,
        radius=0.1,
        seeds=["random"],
        n_cpu=1,
    ):
        u_i = interpolate.interp2d(self.x, self.y, self.u)
        v_i = interpolate.interp2d(self.x, self.y, self.v)

        if not isinstance(seeds, list):
            raise Exception("seeds argument must be list")

        for item in seeds:
            if item == "random":
                xs = np.random.uniform(
                    self.grid["x_start"], self.grid["x_end"], size=n_streamtraces
                )
                ys = np.random.uniform(
                    self.grid["y_start"], self.grid["y_end"], size=n_streamtraces
                )
                self.seeds.extend(np.dstack((xs, ys))[0].tolist())

            elif item == "sources":
                seeds_center = [[x[0], x[1]] for x in self.source_sink if x[2] > 0]
                n_streamstraces_per_source = int(n_streamtraces / len(seeds_center))
                for p in seeds_center:
                    self.seeds.extend(
                        self._circle_points(p, radius, n_streamstraces_per_source)
                    )

            elif item == "grid":
                grid_points = np.dstack((self.X.flatten(), self.Y.flatten()))[0]
                gridskip = int(
                    self.grid["no_points_x"] * self.grid["no_points_y"] / n_streamtraces
                )
                self.seeds.extend(grid_points[::gridskip].tolist())

            sinks = [[x[0], x[1]] for x in self.source_sink if x[2] < 0]
            if n_cpu > 1:
                with Pool(n_cpu) as pool:
                    self.streamtraces = pool.map(
                        partial(
                            compute_trace,
                            maxiter=maxiter,
                            max_sink_dist=0.1,
                            sinks=sinks,
                            grid=self.grid,
                            u_i=u_i,
                            v_i=v_i,
                            dt=dt,
                        ),
                        self.seeds,
                    )
            else:
                for seed in self.seeds:
                    self.streamtraces.append(
                        compute_trace(
                            seed,
                            maxiter=maxiter,
                            max_sink_dist=0.1,
                            sinks=sinks,
                            grid=self.grid,
                            u_i=u_i,
                            v_i=v_i,
                            dt=dt,
                        )
                    )

    def plot(self, num_level=25, legend=True):
        fig, ax = plt.subplots(figsize=(12, 10))
        c = ax.contour(self.x, self.y, self.psi, num_level)
        if legend:
            fig.colorbar(c, ax=ax, shrink=0.9)