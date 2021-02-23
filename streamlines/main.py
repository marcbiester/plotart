from streamlines import streamLines
import numpy as np
import matplotlib.pyplot as plt

grid = {
    "x_start": -5,
    "x_end": 5,
    "y_start": -3.5,
    "y_end": 3.5,
    "no_points_x": 100,
    "no_points_y": 100,
}

self = streamLines(grid)
#self.add_free_stream(v_inf=0.4)
# self.add_doublet(15,0,0)
for i in range(100):
    x, y = np.random.uniform(-5, 5), np.random.uniform(-3.5, 3.5)
    s = -1 if np.random.randint(0, 2) == 0 else 1
    f = np.random.uniform(1, 4)
    self.add_vortex(f * 4 * s, x, y)
    self.add_source_sink(f * s, x, y)


plt.figure(figsize=(420 / 25.4, 297 / 25.4))
self.calc_streamtraces(n_streamtraces=500, maxiter=20000, dt=0.001)
# self.calc_velocity()
for ps in self.streamtraces:
    plt.plot([x[0] for x in ps], [x[1] for x in ps], color='black')
plt.axis('equal')
plt.axis("off")



plt.figure(figsize=(420 / 25.4, 297 / 25.4))
plt.streamplot(self.x, self.y, self.u, self.v, density=3, arrowstyle="-")
plt.axis("equal")
plt.axis('off')
