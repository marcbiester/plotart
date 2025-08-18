import numpy as np
from tqdm import tqdm
import math
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
    cutoff_radius: float = 0.0
    speed_epsilon: float = 1e-6
    n_jobs: int = 1

    method: str = "rk45"     # options: "rk45", "rk4"
    rtol: float = 1e-4
    atol: float = 1e-6
    dt_min: float = 1e-4
    dt_max: float = 0.1
    max_rejects: int = 25    # max consecutive rejects before aborting a trace
    step_factor: float = 0.2   # fraction of local length scale per step




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

    def _rk45_step(self, p: np.ndarray, dt: float):
        """
        One adaptive Dormand–Prince RK45 step with error control and a velocity-based ceiling.
        Returns: (new_point, new_dt, accepted: bool)
        Uses tolerances and limits from self.cfg (must be set by caller).
        """
        cfg = self.cfg  # set by _trace_one before calling us

        def f(point: np.ndarray):
            # velocity at (x,y) via interpolators; returns None if invalid
            u = self.u_i([point[1], point[0]]).item()
            v = self.v_i([point[1], point[0]]).item()
            if not (np.isfinite(u) and np.isfinite(v)):
                return None
            return np.array([u, v], dtype=float)

        # Dormand–Prince coefficients (7 stages; 5th order with 4th order embedded)
        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        a71, a73, a74, a75, a76 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

        b4 = np.array([5179/57600, 0.0, 7571/16695, 393/640,
                    -92097/339200, 187/2100, 1/40])  # 4th order
        b5 = np.array([35/384,     0.0, 500/1113,  125/192,
                    -2187/6784, 11/84,         0.0])  # 5th order

        k1 = f(p)
        if k1 is None:
            return p, max(cfg.dt_min, 0.5*dt), False

        k2 = f(p + dt*(a21*k1))
        if k2 is None:
            return p, max(cfg.dt_min, 0.5*dt), False

        k3 = f(p + dt*(a31*k1 + a32*k2))
        if k3 is None:
            return p, max(cfg.dt_min, 0.5*dt), False

        k4 = f(p + dt*(a41*k1 + a42*k2 + a43*k3))
        if k4 is None:
            return p, max(cfg.dt_min, 0.5*dt), False

        k5 = f(p + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        if k5 is None:
            return p, max(cfg.dt_min, 0.5*dt), False

        k6 = f(p + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        if k6 is None:
            return p, max(cfg.dt_min, 0.5*dt), False

        k7 = f(p + dt*(a71*k1 + a73*k3 + a74*k4 + a75*k5 + a76*k6))
        if k7 is None:
            return p, max(cfg.dt_min, 0.5*dt), False

        K = [k1, k2, k3, k4, k5, k6, k7]
        y4 = p + dt * sum(bi * ki for bi, ki in zip(b4, K))
        y5 = p + dt * sum(bi * ki for bi, ki in zip(b5, K))

        err = np.linalg.norm(y5 - y4, ord=np.inf)
        if not np.isfinite(err):
            return p, max(cfg.dt_min, 0.5*dt), False

        # Tolerance based on state magnitude (classic embedded controller)
        tol = cfg.atol + cfg.rtol * max(np.linalg.norm(p, ord=np.inf),
                                        np.linalg.norm(y5, ord=np.inf))

        # Propose new dt (limit growth/shrink)
        if err == 0.0:
            fac = 5.0
            accepted = True
        else:
            accepted = err <= tol
            # use exponent 1/5 for 5th order
            fac = 0.9 * (tol / err)**0.2
            fac = min(5.0, max(0.1, fac))
        dt_new = dt * fac

        # --- velocity-based ceiling (limit *physical* step length) ---
        v_here = k1  # velocity at p
        vnorm = np.linalg.norm(v_here)
        if vnorm > 1e-12:
            max_phys_dt = cfg.step_factor / vnorm
            dt_new = min(dt_new, max_phys_dt)

        # Clamp to [dt_min, dt_max]
        dt_new = min(cfg.dt_max, max(cfg.dt_min, dt_new))

        # Return the higher-order (y5) solution on accept, otherwise unchanged p
        return (y5 if accepted else p), dt_new, accepted





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
        # let _rk45_step see tolerances/limits
        self.cfg = cfg

        p = np.array(seed, dtype=float)
        pts = [p.copy()]
        steps = 0

        # RK45 state
        t = 0.0
        dt = cfg.dt
        consecutive_rejects = 0

        while steps < cfg.max_steps and len(pts) < cfg.max_points and self.field.grid.contains(p):
            # hard stop near any singularity
            if self._within_cutoff(p, cfg.cutoff_radius):
                break

            v = self._vel(p)
            if not np.all(np.isfinite(v)) or np.linalg.norm(v) < cfg.speed_epsilon:
                break

            # classic termination near sinks
            if self._near_sink(p, cfg.max_sink_dist):
                break

            if cfg.method.lower() == "rk4":
                p_new = self._rk4_step(p, cfg.dt)
                if not np.all(np.isfinite(p_new)):
                    break
                p = p_new
                t += cfg.dt
                pts.append(p.copy())
                steps += 1
                continue

            # ---- RK45 adaptive step (default) ----
            attempt = 0
            while True:
                p_candidate, dt_new, ok = self._rk45_step(p, dt)
                if ok:
                    # accepted: advance
                    t += dt
                    p = p_candidate
                    dt = dt_new
                    pts.append(p.copy())
                    steps += 1
                    consecutive_rejects = 0
                    break
                else:
                    # rejected: shrink dt and retry
                    attempt += 1
                    consecutive_rejects += 1
                    dt = dt_new
                    if dt <= cfg.dt_min + 1e-15 or attempt >= cfg.max_rejects:
                        # cannot make progress; end this streamline gracefully
                        return np.array(pts)

        return np.array(pts)


    # Public API
    def trace(self, seeds: np.ndarray, cfg: Optional[TraceConfig] = None) -> List[np.ndarray]:
        if cfg is None:
            cfg = TraceConfig()
        n_jobs = min(max(1, cfg.n_jobs), cpu_count())
        seeds = np.asarray(seeds, dtype=float).reshape(-1, 2)

        # --- Single-process: full tqdm ---
        if n_jobs == 1:
            iterator = seeds
            if tqdm is not None:
                iterator = tqdm(seeds, desc="Tracing", unit="seed")
            self.traces = [self._trace_one(s, cfg) for s in iterator]
            return self.traces

        # --- Multi-process: preserve order with starmap + rough % after batches ---
        batch_size = 50  # change here if you want different granularity
        total = len(seeds)
        self.traces = []

        with Pool(processes=n_jobs) as pool:
            # process in ordered batches so we can print coarse progress without reordering results
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = seeds[start:end]
                args = [(s, cfg, self.field, self.sinks) for s in batch]
                # starmap preserves input order
                batch_traces = pool.starmap(_trace_one_helper, args)
                self.traces.extend(batch_traces)

                # rough progress print (works even without tqdm)
                done = end
                pct = int(done * 100 / total)
                print(f"[trace] {done}/{total} ({pct}%)")

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
    
    def write_svg(
        self,
        file,
        target_width_px: int = 1200,
        target_height_px: int = None,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        stroke: str = "black",
        stroke_width: float = 0.06,
        non_scaling_stroke: bool = True,
    ):
        """
        Export streamlines to SVG.

        - Coordinates are in data units (1 unit = 1 viewBox unit).
        - The on-screen size is controlled by width/height in pixels.
        - If target_height_px is None, it’s computed to preserve aspect.
        """
        gx = self.field.grid
        dx = gx.x_end - gx.x_start
        dy = gx.y_end - gx.y_start
        if dx <= 0 or dy <= 0:
            raise ValueError("Invalid grid extents")

        if target_height_px is None:
            target_height_px = int(round(target_width_px * dy / dx))

        def x_u(x):  # user units (viewBox space)
            return (x - gx.x_start) + offset_x

        def y_u(y):
            # flip Y so 'up' remains up in the rendered image
            return (gx.y_end - y) + offset_y

        vector_effect = ' vector-effect="non-scaling-stroke"' if non_scaling_stroke else ""

        with open(file, "w", encoding="utf-8") as f:
            f.write(
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{target_width_px}" height="{target_height_px}" '
                f'viewBox="0 0 {dx} {dy}" preserveAspectRatio="xMidYMid meet">\n'
            )
            f.write(f'<g fill="none" stroke="{stroke}" stroke-width="{stroke_width}"{vector_effect}>\n')

            for streamline in self.traces:
                if len(streamline) < 2:
                    continue
                parts = [f'M {x_u(streamline[0,0]):.6g} {y_u(streamline[0,1]):.6g}']
                for p in streamline[1:]:
                    parts.append(f'L {x_u(p[0]):.6g} {y_u(p[1]):.6g}')
                f.write(f'  <path d="{" ".join(parts)}" />\n')

            f.write('</g>\n</svg>\n')




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
    cfg = TraceConfig(
        dt=0.002,
        rtol=1e-6,
        atol=1e-8,
        dt_min=1e-5,
        dt_max=0.02,
        max_steps=8000,
        cutoff_radius=0.15,
        step_factor=0.2,
        n_jobs=min(8, cpu_count())
    )

    tracer.trace(seeds, cfg)
    tracer.plot()
    tracer.write_svg("streamlines.svg", height=600, width=600)