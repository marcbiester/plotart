import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from multiprocessing import Pool
from functools import partial
from xml.dom import minidom
import re
from svgpathtools import parse_path


def point_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def check_sink_proximity(p, sinks, maxdist=0.1):
    f = partial(point_distance, p2=p)
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
        self.vortices=[]
        self.seeds = []
        self.sl_config = {}
        self.grid = grid
        grid_shape = (grid["no_points_x"], grid["no_points_y"])
        self.psi = np.zeros(grid_shape)
        self.u = np.zeros(grid_shape)
        self.v = np.zeros(grid_shape)
        self.generate_mesh()

    def init_from_svg(
        self,
        file,
        config={
            "source_color": "000000",  # black
            "sink_color": "ff0000",  # red
            "seeds_color": "00ff00",  # green
            "vortex_color_cw": "0000ff",  # blue
            "vortex_color_ccw": "000080",  # navy
            "source_strength_default": 0.1,
            "sink_strength_default": -0.1,
            "vortex_default_strength": 0.1,
            "points_per_unit_length": 1,
            "scale": 5,
        },
        plot=False,
    ):

        sources = []
        sinks = []
        seeds = []
        vortices = []
        vortices_strength = []
        sinks_strength = []
        sources_strength = []

        svg_xml = minidom.parse("test.svg")
        paths = svg_xml.getElementsByTagName("path")
        circles = svg_xml.getElementsByTagName("circle")
        for path in paths:
            path = path.toxml()
            color = re.findall(r"#([a-f,0-9]{6})", path)[0]
            d = re.findall(r'\sd="([A-Z,a-z,0-9,\s,\.,\-]*)"', path)[0]
            path = parse_path(d)
            num_samples = int(path.length() * config["points_per_unit_length"])
            if color == config["sink_color"]:
                for j in range(num_samples):
                    sinks.append(path.point(j / num_samples))
                    sinks_strength.append(config["sink_strength_default"])
            elif color == config["source_color"]:
                for j in range(num_samples):
                    sources.append(path.point(j / num_samples))
                    sources_strength.append(config["source_strength_default"])
            elif color == config["seeds_color"]:
                for j in range(num_samples):
                    seeds.append(path.point(j / num_samples))
            elif color == config["vortex_color"]:
                for j in range(num_samples):
                    vortices.append(path.point(j / num_samples))
                    vortices_strength.append(config["vortex_default_strength"])
            else:
                raise Exception("Please make sure features are definded for colors")

        for circle in circles:
            circle = circle.toxml()
            cx = float(re.findall(r'\scx="([0-9,\.,\-]*)"', circle)[0])
            cy = float(re.findall(r'\scy="([0-9,\.,\-]*)"', circle)[0])
            r = float(re.findall(r'\sr="([0-9,\.]*)"', circle)[0])
            color = re.findall(r"#([a-f,0-9]{6})", circle)[0]
            c = complex(cx, cy)

            if color == config["sink_color"]:
                sinks.append(c)
                sinks_strength.append(-r * config["scale"])
            elif color == config["source_color"]:
                sources.append(c)
                sources_strength.append(r * config["scale"])
            elif color == config["seeds_color"]:
                seeds.append(c)
            elif color == config["vortex_color_cw"]:
                vortices.append(c)
                vortices_strength.append(r * config["scale"])
            elif color == config["vortex_color_ccw"]:
                vortices.append(c)
                vortices_strength.append(-r * config["scale"])

        if plot:
            _, ax1 = plt.subplots()
            ax1.scatter(
                [x.real for x in sinks],
                [x.imag for x in sinks],
                color="#" + config["sink_color"],
            )
            ax1.scatter(
                [x.real for x in sources],
                [x.imag for x in sources],
                color="#" + config["source_color"],
            )
            ax1.scatter(
                [x.real for x in seeds],
                [x.imag for x in seeds],
                color="#" + config["seeds_color"],
            )
            ax1.scatter(
                [x.real for x in vortices],
                [x.imag for x in vortices],
                color="#" + config["vortex_color_cw"],
            )

            ax1.invert_yaxis()

        self.seeds = [[x.real, x.imag] for x in seeds]
        for i in range(len(sources)):
            self.add_source_sink(sources_strength[i], sources[i].real, sources[i].imag)

        for i in range(len(sinks)):
            self.add_source_sink(sinks_strength[i], sinks[i].real, sinks[i].imag)

        for i in range(len(vortices)):
            self.add_vortex(vortices_strength[i], vortices[i].real, vortices[i].imag)

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
        self.vortices.append([x, y, strength])
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
        max_sink_dist=0.1,
        seeds=["random"],
        n_cpu=1,
    ):

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
                            max_sink_dist=max_sink_dist,
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
                            max_sink_dist=max_sink_dist,
                            sinks=sinks,
                            grid=self.grid,
                            u_i=u_i,
                            v_i=v_i,
                            dt=dt,
                        )
                    )

    def write_svg_init(
        self,
        file,
        config={
            "source_color": "000000",  # black
            "sink_color": "ff0000",  # red
            "seeds_color": "00ff00",  # green
            "vortex_color_cw": "0000ff",  # blue
            "vortex_color_ccw": "000080",  # navy)
            "scale": 5,
        },
    ):
        sources = [x for x in self.source_sink if x[2] >=0]
        sinks = [x for x in self.source_sink if x[2] <0]


        f = open(file, "w")
        f.write(
            f"""<svg height="{self.grid['y_end'] - self.grid['y_start']}" width="{self.grid['x_end'] - self.grid['x_start']}">"""
        )

        for source in sources:
            f.write(
                f"""<circle cx="{source[0]}" cy="{source[1]}" r="{source[2] / config['scale']}" fill="none" stroke="#{config['source_color']}" />\n"""
            )
        for sink in sinks:
            f.write(
                f"""<circle cx="{sink[0]}" cy="{sink[1]}" r="{abs(sink[2])/ config['scale']}" fill="none" stroke="#{config['sink_color']}" />\n"""
            )
        for vortex in self.vortices:
            if vortex[0] >= 0:
                color = config["vortex_color_cw"]
            else:
                color = config["vortex_color_ccw"]
            f.write(
                f"""<circle cx="{vortex[0]}" cy="{vortex[1]}" r="{abs(vortex[2])/ config['scale']}" fill="none" stroke="#{color}" />\n"""
            )
        for seed in self.seeds:
            f.write(
                f"""<circle cx="{seed[0]}" cy="{seed[1]}" r="{config['scale']}" fill="none" stroke="#{config['seeds_color']}" />\n"""
            )

        y = -100
        for key in self.sl_config.keys():
            f.write(f"""<text x="0" y="{y}">{key}:{self.sl_config[key]}</text>""")
            y += 10

        f.write("</svg>")
        f.close()

    def write_svg(
        self, file, height=297, width=420, offset_x=420 / 2, offset_y=297 / 2
    ):
        print("writing to svg ...", end="")
        dx = self.grid["x_end"] - self.grid["x_start"]
        fx = width / dx
        dy = self.grid["y_end"] - self.grid["y_start"]
        fy = height / dy

        f = open(file, "w")
        f.write(f'<svg height="{height}" width="{width}">')

        d = ""

        for streamline in self.streamtraces:
            # d += f'<circle cx="{seed[1]}" cy="{seed[0]}" r="5" fill="none" stroke="black" />\n'
            if len(streamline) > 2:
                d += f'<path d="M {streamline[0][0]*fx + offset_x} {streamline[0][1]*fy + offset_y} '
                for i in range(len(streamline) - 1):
                    d += f"L {streamline[i+1][0]*fx + offset_x} {streamline[i+1][1]*fy + offset_y} "
                d += '" stroke="black" fill="none" />\n'
        f.write(d)
        f.write("</svg>")
        f.close()

    def plot(self, num_level=25, legend=True):
        fig, ax = plt.subplots(figsize=(12, 10))
        c = ax.contour(self.x, self.y, self.psi, num_level)
        if legend:
            fig.colorbar(c, ax=ax, shrink=0.9)
