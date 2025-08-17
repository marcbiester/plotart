import numpy as np
import matplotlib.pyplot as plt

class PotentialField:
    def __init__(self, grid):
        """
        Initialize the PotentialField on a Cartesian mesh.
        
        Parameters
        ----------
        grid : dict
            Must contain keys 'x_start', 'x_end', 'no_points_x',
                              'y_start', 'y_end', 'no_points_y'
        """
        self.grid = grid
        self._generate_mesh()
        # initialize fields
        self.psis = []
        self.sources_sinks = []
        self.doublets = []
        self.vortices = []
        # start with zero stream-function, velocities
        zero_psi = np.zeros_like(self.X)
        self.psis.append(zero_psi)
        self.psi = zero_psi.copy()
        self.u = np.zeros_like(self.X)
        self.v = np.zeros_like(self.X)

    def _generate_mesh(self):
        """Create x, y arrays and meshgrid X, Y."""
        gx = self.grid
        self.x = np.linspace(gx["x_start"], gx["x_end"], gx["no_points_x"])
        self.y = np.linspace(gx["y_start"], gx["y_end"], gx["no_points_y"])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        # spacings (unused in current ops but available)
        self.dx = (gx["x_end"] - gx["x_start"]) / (gx["no_points_x"] - 1)
        self.dy = (gx["y_end"] - gx["y_start"]) / (gx["no_points_y"] - 1)

    def reset_psi(self):
        """Recompute total ψ by summing all individual ψ contributions."""
        self.psi = sum(self.psis)

    def add_source_sink(self, strength, x, y):
        """Add a source (strength>0) or sink (strength<0)."""
        self.sources_sinks.append((x, y, strength))
        r2 = (self.X - x)**2 + (self.Y - y)**2
        psi = strength/(2*np.pi)*np.arctan2(self.Y - y, self.X - x)
        self.psis.append(psi)
        self.psi += psi
        self.u +=  strength/(2*np.pi)*(self.X - x)/r2
        self.v +=  strength/(2*np.pi)*(self.Y - y)/r2

    def add_free_stream(self, u_inf=0.0, v_inf=0.0):
        """Add a uniform free-stream flow."""
        psi = u_inf*self.Y - v_inf*self.X
        self.psis.append(psi)
        self.psi += psi
        self.u += u_inf
        self.v += v_inf

    def add_doublet(self, strength, x, y):
        """Add a doublet of given strength at (x,y)."""
        self.doublets.append((x, y, strength))
        dx = self.X - x
        dy = self.Y - y
        r2 = dx**2 + dy**2
        psi = -strength/(2*np.pi)*dy/r2
        self.psis.append(psi)
        self.psi += psi
        self.u += -strength/(2*np.pi)*((dx**2 - dy**2)/r2**2)
        self.v += -strength/(2*np.pi)*(2*dx*dy/r2**2)

    def add_vortex(self, strength, x, y):
        """Add a vortex (positive = counter‑clockwise) at (x,y)."""
        self.vortices.append((x, y, strength))
        dx = self.X - x
        dy = self.Y - y
        r2 = dx**2 + dy**2
        psi = strength/(4*np.pi)*np.log(r2)
        self.psis.append(psi)
        self.psi += psi
        self.u +=  strength/(2*np.pi)* dy/r2
        self.v += -strength/(2*np.pi)* dx/r2

    def plot_streamlines(self, n_stream=200, linewidth=1, density=1.5):
        """
        Plot ψ contours and velocity streamlines.
        
        Parameters
        ----------
        n_stream : int
            Number of contour levels for ψ.
        linewidth : float
            Base linewidth for streamlines.
        density : float or 2‐tuple
            Controls closeness of streamlines.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        # ψ contours
        cnt = ax.contour(self.X, self.Y, self.psi, levels=n_stream, linewidths=0.5)
        ax.clabel(cnt, inline=1, fontsize=8)
        # streamlines
        strm = ax.streamplot(
            self.X, self.Y, self.u, self.v,
            density=density, linewidth=linewidth,
            arrowsize=1.0
        )
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Potential‐Flow Streamlines & ψ contours')
        plt.tight_layout()
        plt.show()
