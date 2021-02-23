grid = {
    "x_start": -5,
    "x_end": 5,
    "y_start": -3.5,
    "y_end": 3.5,
    "no_points_x": 200,
    "no_points_y": 200,
}

self = streamLines(grid)
self.add_free_stream(v_inf=0.4)
# self.add_doublet(15,0,0)
for i in range(100):
    x, y = np.random.uniform(-5, 5), np.random.uniform(-3.5, 3.5)
    s = -1 if np.random.randint(0, 2) == 0 else 1
    f = np.random.uniform(1, 3)
    self.add_vortex(f * 4 * s, x, y)
    self.add_source_sink(f * s, x, y)


# self.calc_velocity()
plt.figure(figsize=(420 / 25.4, 297 / 25.4))
plt.streamplot(self.x, self.y, self.u, self.v, density=3, arrowstyle="-")
plt.axis("equal")
plt.axis("off")
