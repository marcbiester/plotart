import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Optional
from multiprocessing import Pool, cpu_count
from scipy.interpolate import RegularGridInterpolator

_EPS = 1e-12

# -----------------------------
# Potential / Velocity Field
# -----------------------------

@dataclass
class Grid:
    x_start: float
    x_end: float
    no_points_x: int
    y_start: float
    y_end: float
    no_points_y: int

    def linspaces(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.x_start, self.x_end, self.no_points_x)
        y = np.linspace(self.y_start, self.y_end, self.no_points_y)
        return x, y

    def contains(self, p: np.ndarray) -> bool:
        return (
            self.x_start <= p[0] <= self.x_end
            and self.y_start <= p[1] <= self.y_end
        )


class PotentialField:
    """
    Builds a potential flow field (psi, u, v) on a Cartesian grid.
    Supports sources/sinks, doublets, vortices, and uniform free stream.

    Parameters
    ----------
    grid : dict
        Grid definition with bounds and resolution.
    core_radius : float, optional
        Soft-core regularization length [same units as x/y]. If > 0, all
        singular contributions use r^2 := (dx^2 + dy^2 + core_radius^2).
        Default 0.0 (no regularization beyond a tiny epsilon).
    """

    def __init__(self, grid: dict, core_radius: float = 0.0):
        self.grid = Grid(**grid)
        self.x, self.y = self.grid.linspaces()
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")

        # Storage
        self._psis: List[np.ndarray] = []
        self.sources_sinks: List[Tuple[float, float, float]] = []
        self.doublets: List[Tuple[float, float, float]] = []
        self.vortices: List[Tuple[float, float, float]] = []

        # Fields
        self.psi = np.zeros_like(self.X, dtype=float)
        self.u = np.zeros_like(self.X, dtype=float)
        self.v = np.zeros_like(self.X, dtype=float)

        # Regularization (soft core)
        self.core_radius: float = float(core_radius)

    # ---- helpers ----
    def _r2_dx_dy(self, x0: float, y0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx = self.X - x0
        dy = self.Y - y0
        r2 = dx*dx + dy*dy + (self.core_radius * self.core_radius)
        # As an ultimate guard, keep r2 from underflowing to 0
        r2 = np.maximum(r2, _EPS)
        return r2, dx, dy

    def reset(self) -> None:
        """Clear the field to zero and re-accumulate all contributions."""
        self.psi.fill(0.0)
        self.u.fill(0.0)
        self.v.fill(0.0)
        for (x0, y0, s) in self.sources_sinks:
            self._acc_source_sink(x0, y0, s)
        for (x0, y0, s) in self.doublets:
            self._acc_doublet(x0, y0, s)
        for (x0, y0, s) in self.vortices:
            self._acc_vortex(x0, y0, s)

    # ---- add elements ----
    def add_source_sink(self, strength: float, x: float, y: float) -> None:
        self.sources_sinks.append((x, y, strength))
        self._acc_source_sink(x, y, strength)

    def _acc_source_sink(self, x: float, y: float, s: float) -> None:
        r2, dx, dy = self._r2_dx_dy(x, y)
        self.psi += s/(2*np.pi) * np.arctan2(dy, dx)
        self.u   += s/(2*np.pi) * dx/r2
        self.v   += s/(2*np.pi) * dy/r2

    def add_doublet(self, strength: float, x: float, y: float) -> None:
        self.doublets.append((x, y, strength))
        self._acc_doublet(x, y, strength)

    def _acc_doublet(self, x: float, y: float, s: float) -> None:
        r2, dx, dy = self._r2_dx_dy(x, y)
        self.psi += -s/(2*np.pi) * dy/r2
        self.u   += -s/(2*np.pi) * ((dx*dx - dy*dy) / (r2*r2))
        self.v   += -s/(2*np.pi) * (2*dx*dy / (r2*r2))

    def add_vortex(self, strength: float, x: float, y: float) -> None:
        self.vortices.append((x, y, strength))
        self._acc_vortex(x, y, strength)

    def _acc_vortex(self, x: float, y: float, g: float) -> None:
        r2, dx, dy = self._r2_dx_dy(x, y)
        self.psi += g/(4*np.pi) * np.log(r2)
        self.u   +=  g/(2*np.pi) * dy/r2
        self.v   += -g/(2*np.pi) * dx/r2

    def add_free_stream(self, u_inf: float = 0.0, v_inf: float = 0.0) -> None:
        self.psi += u_inf*self.Y - v_inf*self.X
        self.u   += u_inf
        self.v   += v_inf

    # ---- interpolators ----
    def build_interpolators(self):
        """Create RegularGridInterpolators for u and v."""
        u_i = RegularGridInterpolator(
            (self.y, self.x), self.u, bounds_error=False, fill_value=np.nan
        )
        v_i = RegularGridInterpolator(
            (self.y, self.x), self.v, bounds_error=False, fill_value=np.nan
        )
        return u_i, v_i


# -----------------------------
# Streamline Tracer (RK4)
# -----------------------------

@dataclass
class TraceConfig:
    dt: float = 0.01
    max_steps: int = 2000
    max_points: int = 1500
    max_sink_dist: float = 0.1
    cutoff_radius: float = 0.0        # NEW: hard stop near any singularity
    speed_epsilon: float = 1e-6       # stop if |v| < eps
    n_jobs: int = 1


class StreamTracer:
    """
    Compute streamlines by integrating the velocity field (u, v).
    Clean separation from PotentialField: you can also pass any callable
    velocity field via interpolators with the same interface.
    """

    def __init__(self, field: PotentialField):
        self.field = field
        self.u_i, self.v_i = field.build_interpolators()
        self.traces: List[np.ndarray] = []

        # sinks are useful termination targets
        self.sinks: List[Tuple[float, float]] = [
            (x, y) for (x, y, s) in self.field.sources_sinks if s < 0
        ]

    # ---- seeding strategies ----
    def seeds_random(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        xs = rng.uniform(self.field.grid.x_start, self.field.grid.x_end, n)
        ys = rng.uniform(self.field.grid.y_start, self.field.grid.y_end, n)
        return np.column_stack((xs, ys))

    def seeds_from_sources(self, n_per_source: int = 10, radius: float = 0.1) -> np.ndarray:
        centers = [(x, y) for (x, y, s) in self.field.sources_sinks if s > 0]
        pts = []
        for (cx, cy) in centers:
            alphas = np.linspace(0, 2*np.pi, n_per_source, endpoint=False)
            for a in alphas:
                pts.append([cx + radius*np.cos(a), cy + radius*np.sin(a)])
        return np.array(pts) if pts else np.empty((0, 2))

    def seeds_from_grid(self, n_total: int) -> np.ndarray:
        grid_pts = np.column_stack((self.field.X.ravel(), self.field.Y.ravel()))
        if n_total >= len(grid_pts):
            return grid_pts
        stride = max(1, len(grid_pts) // n_total)
        return grid_pts[::stride]

    # ---- core RK4 integrator ----
    def _vel(self, p: np.ndarray) -> np.ndarray:
        # Interpolators expect (y, x)
        u = self.u_i([p[1], p[0]]).item()
        v = self.v_i([p[1], p[0]]).item()

        if not np.isfinite(u) or not np.isfinite(v):
            return np.array([np.nan, np.nan])
        return np.array([u, v], dtype=float)

    def _near_sink(self, p: np.ndarray, maxd: float) -> bool:
        if not self.sinks:
            return False
        sx = np.array([s[0] for s in self.sinks])
        sy = np.array([s[1] for s in self.sinks])
        d2 = (p[0] - sx)**2 + (p[1] - sy)**2
        return np.any(d2 <= maxd*maxd)

    def _rk4_step(self, p: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._vel(p)
        if not np.all(np.isfinite(k1)):
            return np.array([np.nan, np.nan])
        k2 = self._vel(p + 0.5*dt*k1)
        k3 = self._vel(p + 0.5*dt*k2)
        k4 = self._vel(p + dt*k3)
        if not (np.all(np.isfinite(k2)) and np.all(np.isfinite(k3)) and np.all(np.isfinite(k4))):
            return np.array([np.nan, np.nan])
        return p + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

    def _rk45_step(self, p, t, dt, rtol=1e-4, atol=1e-6, dt_min=1e-4, dt_max=0.1):
        """Adaptive RK45 integration step for streamline tracing.
        Returns: new_point, new_time, new_dt, success
        """
        def f(point):
            u = self.u_i([point[1], point[0]]).item()
            v = self.v_i([point[1], point[0]]).item()
            return np.array([u, v])

        # Butcher tableau for Dormand–Prince RK45
        c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        ]
        b4 = [5179/57600, 0, 7571/16695, 393/640,
            -92097/339200, 187/2100, 1/40]   # 4th order
        b5 = [35/384, 0, 500/1113, 125/192,
            -2187/6784, 11/84, 0]             # 5th order

        k = []
        k.append(f(p))
        k.append(f(p + dt * (a[1][0] * k[0])))
        k.append(f(p + dt * (a[2][0] * k[0] + a[2][1] * k[1])))
        k.append(f(p + dt * (a[3][0] * k[0] + a[3][1] * k[1] + a[3][2] * k[2])))
        k.append(f(p + dt * (a[4][0] * k[0] + a[4][1] * k[1] + a[4][2] * k[2] + a[4][3] * k[3])))
        k.append(f(p + dt * (a[5][0] * k[0] + a[5][1] * k[1] + a[5][2] * k[2] + a[5][3] * k[3] + a[5][4] * k[4])))
        k.append(f(p + dt * (a[6][0] * k[0] + a[6][2] * k[2] + a[6][3] * k[3] + a[6][4] * k[4] + a[6][5] * k[5])))

        y4 = p + dt * sum(bi * ki for bi, ki in zip(b4, k))
        y5 = p + dt * sum(bi * ki for bi, ki in zip(b5, k))

        # error estimate
        err = np.linalg.norm(y5 - y4, ord=np.inf)

        tol = atol + rtol * max(np.linalg.norm(p, ord=np.inf), np.linalg.norm(y5, ord=np.inf))

        if err <= tol:  # accept step
            # new dt suggestion
            dt_new = min(dt_max, max(dt_min, dt * min(5, 0.9 * (tol / err)**0.2)))
            return y5, t + dt, dt_new, True
        else:  # reject step
            dt_new = max(dt_min, dt * max(0.1, 0.9 * (tol / err)**0.25))
            return p, t, dt_new, False


    def _within_cutoff(self, p: np.ndarray, r: float) -> bool:
        if r <= 0.0:
            return False
        # Build array of singularity positions: sources/sinks, vortices, doublets
        items = self.field.sources_sinks + self.field.vortices + self.field.doublets
        if not items:
            return False
        pts = np.array([[x, y] for (x, y, _) in items], dtype=float)
        d2 = np.sum((pts - p)**2, axis=1)
        return np.any(d2 <= r*r)

    def _trace_one(self, seed: np.ndarray, cfg: TraceConfig) -> np.ndarray:
        p = np.array(seed, dtype=float)
        pts = [p.copy()]
        steps = 0
        while steps < cfg.max_steps and len(pts) < cfg.max_points and self.field.grid.contains(p):
            # cutoff near any singularity (hard stop)
            if self._within_cutoff(p, cfg.cutoff_radius):
                break

            v = self._vel(p)
            if not np.all(np.isfinite(v)) or np.linalg.norm(v) < cfg.speed_epsilon:
                break

            # terminate when close to sink (classic behaviour)
            if self._near_sink(p, cfg.max_sink_dist):
                break

            p = self._rk4_step(p, cfg.dt)
            if not np.all(np.isfinite(p)):
                break
            pts.append(p.copy())
            steps += 1
        return np.array(pts)

    # Public API
    def trace(self, seeds: np.ndarray, cfg: Optional[TraceConfig] = None) -> List[np.ndarray]:
        if cfg is None:
            cfg = TraceConfig()
        n_jobs = min(max(1, cfg.n_jobs), cpu_count())
        seeds = np.asarray(seeds, dtype=float).reshape(-1, 2)

        if n_jobs == 1:
            self.traces = [self._trace_one(s, cfg) for s in seeds]
        else:
            with Pool(processes=n_jobs) as pool:
                self.traces = pool.starmap(_trace_one_helper, [(s, cfg, self.field, self.sinks) for s in seeds])
        return self.traces

    def plot(self, show=True, figsize: Tuple[int, int]=(10, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        # Only the integrated streamlines (no psi contour background)
        for tr in self.traces:
            if len(tr) >= 2:
                ax.plot(tr[:,0], tr[:,1], lw=1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.field.grid.x_start, self.field.grid.x_end)
        ax.set_ylim(self.field.grid.y_start, self.field.grid.y_end)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Streamlines")
        if show:
            plt.show()
        return fig, ax
    
    def write_svg(self, file, height=5, width=5, offset_x=0, offset_y=0):
        """Export streamlines to SVG (self.traces)."""
        gx = self.field.grid
        dx = gx.x_end - gx.x_start
        dy = gx.y_end - gx.y_start
        fx = width / dx
        fy = height / dy

        with open(file, "w") as f:
            f.write(f'<svg height="{height}" width="{width}" xmlns="http://www.w3.org/2000/svg">\n')
            for streamline in self.traces:
                if len(streamline) >= 2:
                    path = f'M {streamline[0,0]*fx + offset_x} {streamline[0,1]*fy + offset_y} '
                    path += " ".join(f"L {p[0]*fx + offset_x} {p[1]*fy + offset_y}" for p in streamline[1:])
                    f.write(f'<path d="{path}" stroke="black" fill="none" />\n')
            f.write("</svg>\n")


# Helper for multiprocessing (top-level function, picklable)
def _trace_one_helper(seed: np.ndarray, cfg: TraceConfig, field: PotentialField, sinks: List[Tuple[float,float]]) -> np.ndarray:
    tracer = StreamTracer(field)  # build local interpolators in each process
    tracer.sinks = sinks  # share precomputed sinks
    return tracer._trace_one(seed, cfg)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    grid = {
        "x_start": 0, "x_end": 10, "no_points_x": 100,
        "y_start": 0, "y_end": 10, "no_points_y": 100
    }

    # Soft-core regularization to tame singularities,
    # set to 0.0 for exact math
    pf = PotentialField(grid, core_radius=0.1)
    pf.add_source_sink(+5.0, x=2.0, y=1.0)    # source
    pf.add_source_sink(-5.0, x=8.0, y=8.0)    # sink
    pf.add_vortex(2.0,  x=5.0, y=5.0)
    pf.add_vortex(20.0, x=8.0, y=8.0)         # strong vortex at the sink
    pf.add_free_stream(u_inf=0.5, v_inf=2.0)

    tracer = StreamTracer(pf)
    seeds = np.vstack([
        #tracer.seeds_from_sources(n_per_source=12, radius=0.15),
        tracer.seeds_from_grid(180),
        #tracer.seeds_random(100)
    ])

    # Hard cutoff near singularities so we stop before jittering
    cfg = TraceConfig(dt=0.005, max_steps=8000, cutoff_radius=0.15, n_jobs=min(8, cpu_count()))
    tracer.trace(seeds, cfg)
    tracer.plot()
    tracer.write_svg("streamlines.svg", height=600, width=800)