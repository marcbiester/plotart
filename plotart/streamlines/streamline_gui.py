#!/usr/bin/env python3
"""
Minimal GUI for interactively building a potential flow field and plotting streamlines.

- Click to place: Source, Sink, Vortex, Doublet
- Set strengths and global parameters in the right-hand panel
- Choose seed strategy and integration method (rk45 default)
- Calculate to render; Save SVG to export

Requires: tkinter (stdlib), numpy, matplotlib, scipy, tqdm
Must be in the same folder as streamlines_gpt5.py
"""

import tkinter as tk
import json, os
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import your engine
import streamlines_gpt5 as eng  # <-- your file

class VerticalScrolledFrame(ttk.Frame):
    """
    A scrollable container: use `self.interior` as the parent for your widgets.
    Works on Linux/Windows/macOS mouse wheels.
    """
    def __init__(self, master, width=360, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=width)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # The interior frame that holds content
        self.interior = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")

        # Update scrollregion when interior size changes
        self.interior.bind("<Configure>", self._on_interior_configure)
        # Keep window width matched to canvas width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel support (Linux/Win/macOS)
        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.interior)

    def _on_interior_configure(self, event):
        # Update scrollable region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Fit the interior window to canvas width
        self.canvas.itemconfig(self._win, width=event.width)

    def _bind_mousewheel(self, widget):
        # Windows / macOS
        widget.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        # Linux (X11)
        widget.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        widget.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")
    
    def _on_mousewheel(self, event):
        # Windows: event.delta is +/-120 multiples; macOS uses small deltas
        delta = int(-1 * (event.delta/120)) if event.delta != 0 else 0
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(+1, "units")


# ---------- Data model for placed elements ----------

@dataclass
class SourceSink:
    x: float
    y: float
    strength: float  # + source, - sink

@dataclass
class Vortex:
    x: float
    y: float
    gamma: float

@dataclass
class Doublet:
    x: float
    y: float
    strength: float

# ---------- GUI App ----------

class StreamlinesApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Streamlines Builder — potential flow playground")
        self.geometry("1200x800")

        # State
        self.mode = tk.StringVar(value="source")  # tool mode: source/sink/vortex/doublet
        self.sources_sinks: List[SourceSink] = []
        self.vortices: List[Vortex] = []
        self.doublets: List[Doublet] = []
        self.free_stream = (0.0, 0.0)

        # Defaults (grid + numerics)
        self.x_start = tk.DoubleVar(value=0.0)
        self.x_end   = tk.DoubleVar(value=10.0)
        self.y_start = tk.DoubleVar(value=0.0)
        self.y_end   = tk.DoubleVar(value=10.0)
        self.nx      = tk.IntVar(value=120)
        self.ny      = tk.IntVar(value=120)
        self.core_radius = tk.DoubleVar(value=0.1)

        # element strengths
        self.source_strength = tk.DoubleVar(value=5.0)
        self.sink_strength   = tk.DoubleVar(value=5.0)
        self.vortex_gamma    = tk.DoubleVar(value=10.0)
        self.doublet_strength = tk.DoubleVar(value=2.0)

        # free stream
        self.u_inf = tk.DoubleVar(value=0.5)
        self.v_inf = tk.DoubleVar(value=0.0)

        # seeding (multi-select)
        self.seed_use_grid = tk.BooleanVar(value=True)
        self.seed_use_random = tk.BooleanVar(value=False)
        self.seed_use_sources = tk.BooleanVar(value=False)

        self.seed_count_grid = tk.IntVar(value=200)    # grid: total sampled points
        self.seed_count_random = tk.IntVar(value=0)    # random seeds
        self.seed_n_per_source = tk.IntVar(value=12)   # per positive source
        self.source_seed_radius = tk.DoubleVar(value=0.15)

        self.rng_seed = tk.IntVar(value=42)            # (keep) random reproducibility


        # integrator config
        self.method = tk.StringVar(value="rk45")  # rk45 or rk4
        self.dt = tk.DoubleVar(value=0.005)
        self.rtol = tk.DoubleVar(value=1e-5)
        self.atol = tk.DoubleVar(value=1e-7)
        self.dt_min = tk.DoubleVar(value=1e-5)
        self.dt_max = tk.DoubleVar(value=0.3)
        self.max_steps = tk.IntVar(value=8000)
        self.cutoff_radius = tk.DoubleVar(value=0.15)
        self.speed_eps = tk.DoubleVar(value=1e-6)
        self.step_factor = tk.DoubleVar(value=0.2)
        self.n_jobs = tk.IntVar(value=max(1, eng.cpu_count() // 2))

        self.rng_seed = tk.IntVar(value=42)  # used for random seeding reproducibility

        # build UI
        self._build_ui()

        # storage for last computed tracer & seeds
        self._last_tracer: Optional[eng.StreamTracer] = None
        self._last_seeds: Optional[np.ndarray] = None

    # ----- UI layout -----
    def _build_ui(self):
        # Left: Matplotlib canvas
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Click to place elements • Right panel = parameters")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self._update_axes_limits()

        self.canvas = FigureCanvasTkAgg(fig, master=left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bind click
        self.canvas.mpl_connect("button_press_event", self._on_click)

        # Right: Controls
        right_sf = VerticalScrolledFrame(self, width=380)
        right_sf.pack(side=tk.RIGHT, fill=tk.Y)
        right = right_sf.interior  # <- use this as the parent for all the controls below

        # --- Modes (tools) ---
        tool_grp = ttk.LabelFrame(right, text="Place element (click canvas)")
        tool_grp.pack(fill=tk.X, pady=4)

        for text, val in [("Source (+)", "source"),
                          ("Sink (−)", "sink"),
                          ("Vortex (Γ)", "vortex"),
                          ("Doublet", "doublet")]:
            ttk.Radiobutton(tool_grp, text=text, variable=self.mode, value=val).pack(anchor="w")

        # Strengths
        str_grp = ttk.LabelFrame(right, text="Element parameters")
        str_grp.pack(fill=tk.X, pady=4)
        self._labeled_entry(str_grp, "Source strength (+):", self.source_strength)
        self._labeled_entry(str_grp, "Sink strength (magnitude):", self.sink_strength)
        self._labeled_entry(str_grp, "Vortex Γ:", self.vortex_gamma)
        self._labeled_entry(str_grp, "Doublet strength:", self.doublet_strength)

        # Free stream
        fs_grp = ttk.LabelFrame(right, text="Free-stream (U∞, V∞)")
        fs_grp.pack(fill=tk.X, pady=4)
        self._labeled_entry(fs_grp, "U∞:", self.u_inf)
        self._labeled_entry(fs_grp, "V∞:", self.v_inf)
        ttk.Button(fs_grp, text="Apply Free-Stream", command=self._apply_free_stream).pack(fill=tk.X, pady=2)
        ttk.Button(fs_grp, text="Remove Free-Stream",
           command=lambda: (setattr(self, 'free_stream', (0.0, 0.0)),
                            self._sync_element_list(),
                            self._redraw_elements())
          ).pack(fill=tk.X, pady=2)


        # Grid & field
        grid_grp = ttk.LabelFrame(right, text="Grid & Field")
        grid_grp.pack(fill=tk.X, pady=4)
        self._labeled_entry(grid_grp, "x_start:", self.x_start)
        self._labeled_entry(grid_grp, "x_end:", self.x_end)
        self._labeled_entry(grid_grp, "y_start:", self.y_start)
        self._labeled_entry(grid_grp, "y_end:", self.y_end)
        self._labeled_entry(grid_grp, "nx:", self.nx)
        self._labeled_entry(grid_grp, "ny:", self.ny)
        self._labeled_entry(grid_grp, "Core radius:", self.core_radius)

        # Seeding (multi-select)
        seed_grp = ttk.LabelFrame(right, text="Seeding")
        seed_grp.pack(fill=tk.X, pady=4)

        ttk.Checkbutton(seed_grp, text="Grid (sampled)", variable=self.seed_use_grid).pack(anchor="w")
        self._labeled_entry(seed_grp, "Grid count:", self.seed_count_grid)

        ttk.Checkbutton(seed_grp, text="Random", variable=self.seed_use_random).pack(anchor="w")
        self._labeled_entry(seed_grp, "Random count:", self.seed_count_random)
        self._labeled_entry(seed_grp, "Random seed:", self.rng_seed)

        ttk.Checkbutton(seed_grp, text="Around positive sources", variable=self.seed_use_sources).pack(anchor="w")
        self._labeled_entry(seed_grp, "n per source:", self.seed_n_per_source)
        self._labeled_entry(seed_grp, "Source ring radius:", self.source_seed_radius)



        # Integrator
        integ_grp = ttk.LabelFrame(right, text="Integrator")
        integ_grp.pack(fill=tk.X, pady=4)
        for text, val in [("RK45 (adaptive)", "rk45"), ("RK4 (fixed)", "rk4")]:
            ttk.Radiobutton(integ_grp, text=text, variable=self.method, value=val).pack(anchor="w")
        self._labeled_entry(integ_grp, "dt (start):", self.dt)
        self._labeled_entry(integ_grp, "rtol:", self.rtol)
        self._labeled_entry(integ_grp, "atol:", self.atol)
        self._labeled_entry(integ_grp, "dt_min:", self.dt_min)
        self._labeled_entry(integ_grp, "dt_max:", self.dt_max)
        self._labeled_entry(integ_grp, "max_steps:", self.max_steps)
        self._labeled_entry(integ_grp, "cutoff_radius:", self.cutoff_radius)
        self._labeled_entry(integ_grp, "speed_epsilon:", self.speed_eps)
        self._labeled_entry(integ_grp, "step_factor:", self.step_factor)
        self._labeled_entry(integ_grp, "n_jobs:", self.n_jobs)

        # --- Element list + remove ---
        ttk.Label(right, text="Elements").pack(anchor="w", pady=(10, 2))

        self.element_list = tk.Listbox(right, height=8)
        self.element_list.pack(fill="x", pady=2)

        btn_remove = ttk.Button(right, text="Remove Selected", command=self._remove_selected_element)
        btn_remove.pack(fill="x", pady=(2, 5))

        # Actions
        act_grp = ttk.LabelFrame(right, text="Actions")
        act_grp.pack(fill=tk.X, pady=6)
        ttk.Button(act_grp, text="Calculate", command=self._calculate).pack(fill=tk.X, pady=2)
        ttk.Button(act_grp, text="Undo last", command=self._undo).pack(fill=tk.X, pady=2)
        ttk.Button(act_grp, text="Clear all", command=self._clear_all).pack(fill=tk.X, pady=2)
        ttk.Button(act_grp, text="Save SVG", command=self._save_svg).pack(fill=tk.X, pady=2)

        ttk.Button(act_grp, text="Save Config", command=self._save_config).pack(fill=tk.X, pady=2)
        ttk.Button(act_grp, text="Load Config", command=self._load_config).pack(fill=tk.X, pady=2)



        # Legend / help
        help_grp = ttk.LabelFrame(right, text="Legend")
        help_grp.pack(fill=tk.BOTH, pady=6, expand=True)
        ttk.Label(help_grp, text="Source: red ●\nSink: blue ×\nVortex: purple ○\nDoublet: green ◆",
                  justify="left").pack(anchor="w")

        # Initial draw of elements layer
        self._redraw_elements()

    def _on_add_source(self):
        x = float(self.entry_x.get())
        y = float(self.entry_y.get())
        strength = float(self.entry_strength.get())
        self.sources_sinks.append(SourceSink(x, y, strength))
        self._redraw_elements()
        self.element_list.insert(tk.END, f"Source ({x:.2f},{y:.2f}), s={strength}")

    def _current_config_dict(self) -> dict:
        """Collect current UI state + elements into a JSON-serializable dict."""
        cfg = {
            "grid": {
                "x_start": float(self.x_start.get()),
                "x_end": float(self.x_end.get()),
                "y_start": float(self.y_start.get()),
                "y_end": float(self.y_end.get()),
                "nx": int(self.nx.get()),
                "ny": int(self.ny.get()),
                "core_radius": float(self.core_radius.get()),
            },
            "elements": {
                "sources_sinks": [
                    {"x": e.x, "y": e.y, "strength": e.strength}
                    for e in self.sources_sinks
                ],
                "vortices": [
                    {"x": v.x, "y": v.y, "gamma": v.gamma}
                    for v in self.vortices
                ],
                "doublets": [
                    {"x": d.x, "y": d.y, "strength": d.strength}
                    for d in self.doublets
                ],
            },
            "freestream": {"u_inf": float(self.u_inf.get()), "v_inf": float(self.v_inf.get())},
            "seeding": {
                "use_grid": bool(self.seed_use_grid.get()),
                "grid_count": int(self.seed_count_grid.get()),
                "use_random": bool(self.seed_use_random.get()),
                "random_count": int(self.seed_count_random.get()),
                "rng_seed": int(self.rng_seed.get()),
                "use_sources": bool(self.seed_use_sources.get()),
                "n_per_source": int(self.seed_n_per_source.get()),
                "source_seed_radius": float(self.source_seed_radius.get()),
            },
            "integrator": {
                "method": self.method.get().lower(),
                "dt": float(self.dt.get()),
                "rtol": float(self.rtol.get()),
                "atol": float(self.atol.get()),
                "dt_min": float(self.dt_min.get()),
                "dt_max": float(self.dt_max.get()),
                "max_steps": int(self.max_steps.get()),
                "cutoff_radius": float(self.cutoff_radius.get()),
                "speed_epsilon": float(self.speed_eps.get()),
                "step_factor": float(self.step_factor.get()),
                "n_jobs": int(self.n_jobs.get()),
            },
            "meta": {
                "app": "streamlines_gui",
                "version": 1,
            }
        }
        return cfg

    def _apply_config_dict(self, cfg: dict, run_calculate: bool = True):
        """Apply a previously saved config dict to UI + internal lists."""
        try:
            g = cfg["grid"]
            self.x_start.set(float(g["x_start"])); self.x_end.set(float(g["x_end"]))
            self.y_start.set(float(g["y_start"])); self.y_end.set(float(g["y_end"]))
            self.nx.set(int(g["nx"])); self.ny.set(int(g["ny"]))
            self.core_radius.set(float(g.get("core_radius", 0.0)))

            # elements
            self.sources_sinks.clear()
            for e in cfg["elements"]["sources_sinks"]:
                self.sources_sinks.append(SourceSink(float(e["x"]), float(e["y"]), float(e["strength"])))
            self.vortices.clear()
            for v in cfg["elements"]["vortices"]:
                self.vortices.append(Vortex(float(v["x"]), float(v["y"]), float(v["gamma"])))
            self.doublets.clear()
            for d in cfg["elements"]["doublets"]:
                self.doublets.append(Doublet(float(d["x"]), float(d["y"]), float(d["strength"])))

            # free-stream
            fs = cfg.get("freestream", {})
            self.u_inf.set(float(fs.get("u_inf", 0.0)))
            self.v_inf.set(float(fs.get("v_inf", 0.0)))
            self.free_stream = (self.u_inf.get(), self.v_inf.get())

            # seeding (multi-select) with backward compatibility
            s = cfg["seeding"]
            # new keys
            if "use_grid" in s or "use_random" in s or "use_sources" in s:
                self.seed_use_grid.set(bool(s.get("use_grid", True)))
                self.seed_count_grid.set(int(s.get("grid_count", 200)))
                self.seed_use_random.set(bool(s.get("use_random", False)))
                self.seed_count_random.set(int(s.get("random_count", 0)))
                self.rng_seed.set(int(s.get("rng_seed", 42)))
                self.seed_use_sources.set(bool(s.get("use_sources", False)))
                self.seed_n_per_source.set(int(s.get("n_per_source", 12)))
                self.source_seed_radius.set(float(s.get("source_seed_radius", 0.15)))
            else:
                # legacy single-mode configs
                mode = s.get("mode", "grid")
                self.seed_use_grid.set(mode == "grid")
                self.seed_use_random.set(mode == "random")
                self.seed_use_sources.set(mode == "sources")
                self.seed_count_grid.set(int(s.get("seed_count", 200)) if mode == "grid" else 0)
                self.seed_count_random.set(int(s.get("seed_count", 200)) if mode == "random" else 0)
                self.seed_n_per_source.set(12)
                self.source_seed_radius.set(float(s.get("source_seed_radius", 0.15)))
                self.rng_seed.set(int(s.get("rng_seed", 42)))


            # integrator
            it = cfg["integrator"]
            self.method.set(it.get("method", "rk45"))
            self.dt.set(float(it.get("dt", 0.005)))
            self.rtol.set(float(it.get("rtol", 1e-5)))
            self.atol.set(float(it.get("atol", 1e-7)))
            self.dt_min.set(float(it.get("dt_min", 1e-5)))
            self.dt_max.set(float(it.get("dt_max", 0.03)))
            self.max_steps.set(int(it.get("max_steps", 8000)))
            self.cutoff_radius.set(float(it.get("cutoff_radius", 0.15)))
            self.speed_eps.set(float(it.get("speed_epsilon", 1e-6)))
            self.step_factor.set(float(it.get("step_factor", 0.2)))
            self.n_jobs.set(int(it.get("n_jobs", max(1, eng.cpu_count() // 2))))

            # refresh visuals
            self._redraw_elements()
            if run_calculate:
                self._calculate()
        except Exception as ex:
            messagebox.showerror("Load Config", f"Invalid config format:\n{ex}")

    def _save_config(self):
        """Save current configuration to a JSON file."""
        cfg = self._current_config_dict()
        path = filedialog.asksaveasfilename(
            title="Save Config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            messagebox.showinfo("Save Config", f"Saved to:\n{path}")
        except Exception as ex:
            messagebox.showerror("Save Config", f"Failed to save:\n{ex}")

    def _load_config(self):
        """Load configuration from a JSON file and apply it."""
        path = filedialog.askopenfilename(
            title="Load Config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._apply_config_dict(cfg, run_calculate=True)
            messagebox.showinfo("Load Config", "Configuration loaded.")
        except Exception as ex:
            messagebox.showerror("Load Config", f"Failed to load:\n{ex}")

    def _remove_selected_element(self):
        sel = self.element_list.curselection()
        if not sel:
            return
        idx = sel[0]

        # If free-stream exists, it's the first row (index 0)
        if self._has_free_stream() and idx == 0:
            self.free_stream = (0.0, 0.0)
            self._sync_element_list()
            self._redraw_elements()
            return

        # Adjust index if free-stream occupies row 0
        offset = 1 if self._has_free_stream() else 0
        k = idx - offset

        # sources/sinks
        if 0 <= k < len(self.sources_sinks):
            del self.sources_sinks[k]
            self._sync_element_list()
            self._redraw_elements()
            return
        k -= len(self.sources_sinks)

        # vortices
        if 0 <= k < len(self.vortices):
            del self.vortices[k]
            self._sync_element_list()
            self._redraw_elements()
            return
        k -= len(self.vortices)

        # doublets
        if 0 <= k < len(self.doublets):
            del self.doublets[k]
            self._sync_element_list()
            self._redraw_elements()
            return


    def _sync_element_list(self):
        # Create the listbox once in _build_ui (see previous message), then:
        self.element_list.delete(0, tk.END)

        # Free-stream first (if present)
        if self._has_free_stream():
            u, v = self.free_stream
            self.element_list.insert(tk.END, f"Free-stream U∞={u:.3g}, V∞={v:.3g}")

        # Then sources/sinks
        for s in self.sources_sinks:
            kind = "Source" if s.strength > 0 else "Sink"
            self.element_list.insert(tk.END, f"{kind} ({s.x:.2f},{s.y:.2f}) s={s.strength:.3g}")

        # Vortices
        for v in self.vortices:
            self.element_list.insert(tk.END, f"Vortex ({v.x:.2f},{v.y:.2f}) Γ={v.gamma:.3g}")

        # Doublets
        for d in self.doublets:
            self.element_list.insert(tk.END, f"Doublet ({d.x:.2f},{d.y:.2f}) μ={d.strength:.3g}")



    def _labeled_entry(self, parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        e = ttk.Entry(row, textvariable=var, width=12)
        e.pack(side=tk.LEFT)
        return e

    # ----- Canvas interaction -----

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = float(event.xdata), float(event.ydata)
        mode = self.mode.get()
        if mode == "source":
            s = float(self.source_strength.get())
            self.sources_sinks.append(SourceSink(x, y, +abs(s)))
        elif mode == "sink":
            s = float(self.sink_strength.get())
            self.sources_sinks.append(SourceSink(x, y, -abs(s)))
        elif mode == "vortex":
            g = float(self.vortex_gamma.get())
            self.vortices.append(Vortex(x, y, g))
        elif mode == "doublet":
            s = float(self.doublet_strength.get())
            self.doublets.append(Doublet(x, y, s))
        self._redraw_elements()

    def _apply_free_stream(self):
        self.free_stream = (float(self.u_inf.get()), float(self.v_inf.get()))
        self._sync_element_list()
        self._redraw_elements()
        messagebox.showinfo("Free-stream", f"Applied U∞={self.free_stream[0]}, V∞={self.free_stream[1]}")

    def _undo(self):
        # Undo from the last modified list that has entries
        for lst in (self.doublets, self.vortices, self.sources_sinks):
            if lst:
                lst.pop()
                break
        self._redraw_elements()

    def _clear_all(self):
        self.sources_sinks.clear()
        self.vortices.clear()
        self.doublets.clear()
        self._redraw_elements()
        # Clear plot except axes
        self.ax.cla()
        self._update_axes_limits()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Click to place elements • Right panel = parameters")
        self.canvas.draw_idle()

    # ----- Drawing -----

    def _update_axes_limits(self):
        self.ax.set_xlim(self.x_start.get(), self.x_end.get())
        self.ax.set_ylim(self.y_start.get(), self.y_end.get())

    def _redraw_elements(self):
        """
        Redraw element markers (sources, sinks, vortices, doublets) on top of the plot.
        Keeps any existing streamlines (lines tagged with _is_streamline).
        """
        # Remove all non-streamline lines
        for line in list(self.ax.lines):
            if not getattr(line, "_is_streamline", False):
                line.remove()

        # Remove any scatter/collection/patch artists
        for coll in list(self.ax.collections):
            coll.remove()
        for patch in list(self.ax.patches):
            patch.remove()

        # Keep axes limits / labels
        self._update_axes_limits()

        # Re-plot element markers
        xs_p = [e.x for e in self.sources_sinks if e.strength > 0]
        ys_p = [e.y for e in self.sources_sinks if e.strength > 0]
        xs_n = [e.x for e in self.sources_sinks if e.strength < 0]
        ys_n = [e.y for e in self.sources_sinks if e.strength < 0]
        if xs_p:
            self.ax.scatter(xs_p, ys_p, s=30, c="red", marker="o", label="source")
        if xs_n:
            self.ax.scatter(xs_n, ys_n, s=30, c="blue", marker="x", label="sink")

        if self.vortices:
            xv = [v.x for v in self.vortices]
            yv = [v.y for v in self.vortices]
            # hollow circle to distinguish
            self.ax.scatter(xv, yv, s=30, c="purple", marker="o", facecolors="none", label="vortex")

        if self.doublets:
            xd = [d.x for d in self.doublets]
            yd = [d.y for d in self.doublets]
            self.ax.scatter(xd, yd, s=40, c="green", marker="D", label="doublet")

        # --- free-stream indicator (arrow) ---
        u, v = self.free_stream
        if self._has_free_stream():
            # place arrow near top-left; scale length to ~10% of width
            x0, x1 = self.x_start.get(), self.x_end.get()
            y0, y1 = self.y_start.get(), self.y_end.get()
            W = x1 - x0
            H = y1 - y0
            norm = (u**2 + v**2) ** 0.5
            if norm > 0:
                ux, uy = u / norm, v / norm
                L = 0.12 * W
                ax0 = x0 + 0.06 * W
                ay0 = y1 - 0.08 * H
                self.ax.arrow(ax0, ay0, ux * L, uy * L,
                            head_width=0.03 * H, head_length=0.04 * W,
                            fc="gray", ec="gray", length_includes_head=True, linewidth=1.2)
                # label
                self.ax.text(ax0, ay0, "U∞", fontsize=9, color="gray",
                            verticalalignment="top", horizontalalignment="left")


        self._sync_element_list()
        self.canvas.draw_idle()

    def _has_free_stream(self) -> bool:
        u, v = self.free_stream
        return abs(u) + abs(v) > 0

    # ----- Calculation & plotting -----

    def _calculate(self):
        try:
            grid = {
                "x_start": float(self.x_start.get()),
                "x_end":   float(self.x_end.get()),
                "no_points_x": int(self.nx.get()),
                "y_start": float(self.y_start.get()),
                "y_end":   float(self.y_end.get()),
                "no_points_y": int(self.ny.get()),
            }
            pf = eng.PotentialField(grid, core_radius=float(self.core_radius.get()))
            # elements
            for e in self.sources_sinks:
                pf.add_source_sink(e.strength, e.x, e.y)
            for v in self.vortices:
                pf.add_vortex(v.gamma, v.x, v.y)
            for d in self.doublets:
                pf.add_doublet(d.strength, d.x, d.y)
            # free stream
            u_inf, v_inf = self.free_stream
            if abs(u_inf) + abs(v_inf) > 0:
                pf.add_free_stream(u_inf=u_inf, v_inf=v_inf)

            tracer = eng.StreamTracer(pf)

            # seeds (combine selected strategies)
            seed_arrays = []

            if self.seed_use_grid.get() and int(self.seed_count_grid.get()) > 0:
                seed_arrays.append(tracer.seeds_from_grid(int(self.seed_count_grid.get())))

            if self.seed_use_random.get() and int(self.seed_count_random.get()) > 0:
                rng = np.random.default_rng(int(self.rng_seed.get()))
                seed_arrays.append(tracer.seeds_random(int(self.seed_count_random.get()), rng=rng))

            if self.seed_use_sources.get():
                npos = sum(1 for (x, y, sgn) in tracer.field.sources_sinks if sgn > 0)
                if npos > 0 and int(self.seed_n_per_source.get()) > 0:
                    seed_arrays.append(
                        tracer.seeds_from_sources(
                            n_per_source=int(self.seed_n_per_source.get()),
                            radius=float(self.source_seed_radius.get())
                        )
                    )

            if not seed_arrays:
                messagebox.showinfo("Seeding", "No seeding strategy selected or counts are zero.")
                return

            seeds = np.vstack(seed_arrays)


            # cfg
            cfg = eng.TraceConfig(
                dt=float(self.dt.get()),
                rtol=float(self.rtol.get()),
                atol=float(self.atol.get()),
                dt_min=float(self.dt_min.get()),
                dt_max=float(self.dt_max.get()),
                max_steps=int(self.max_steps.get()),
                cutoff_radius=float(self.cutoff_radius.get()),
                speed_epsilon=float(self.speed_eps.get()),
                step_factor=float(self.step_factor.get()),
                n_jobs=max(1, int(self.n_jobs.get())),
                method=self.method.get().lower()
            )

            # compute
            tracer.trace(seeds, cfg)

            # plot
            self._plot_streamlines(tracer)
            self._last_tracer = tracer
            self._last_seeds = seeds

        except Exception as e:
            messagebox.showerror("Error", f"Computation failed:\n{e}")

    def _plot_streamlines(self, tracer: eng.StreamTracer):
        # Clear previous streamlines but keep element markers
        self.ax.cla()
        self._update_axes_limits()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Streamlines")

        # draw the streamlines
        for tr in tracer.traces:
            if len(tr) >= 2:
                line, = self.ax.plot(tr[:,0], tr[:,1], lw=1.0)
                line._is_streamline = True  # tag so we can preserve on overlay clear

        leg = self.ax.get_legend()
        if leg is not None:
            leg.remove()

        # redraw element markers on top
        self._redraw_elements()
        self.canvas.draw_idle()

    def _save_svg(self):
        if self._last_tracer is None:
            messagebox.showinfo("Save SVG", "No streamlines yet. Press Calculate first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save SVG",
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            # optional: tweak stroke / stroke_width
            self._last_tracer.write_svg(
                path,
                target_width_px=1600,           # choose your preferred output width
                target_height_px=None,          # auto-compute to preserve aspect
                stroke="black",
                stroke_width=0.06,              # this is in viewBox units; visually stable via non-scaling-stroke
                non_scaling_stroke=True
            )
            messagebox.showinfo("Save SVG", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save SVG", f"Failed to save:\n{e}")


# ---------- main ----------

if __name__ == "__main__":
    app = StreamlinesApp()
    app.mainloop()
