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

for i in range(500):
    x, y = np.random.uniform(-5, 5), np.random.uniform(-3.5, 3.5)
    self.add_vortex(np.random.choice([-1,1])*4, x, y) 
    self.add_source_sink(-0.1, x, y)
    
for i in range(200):
    x, y = np.random.uniform(-5, 5), np.random.uniform(-3.5, 3.5)
    self.add_source_sink(2, x, y)


plt.figure(figsize=(420 / 25.4, 297 / 25.4))
self.calc_streamtraces(n_streamtraces=1000, seeds=['grid'], dt=0.001, maxiter=2000, radius=0.1)
for ps in self.streamtraces:
    plt.plot([x[0] for x in ps], [x[1] for x in ps], color='black', alpha=0.5)
plt.axis('equal')
plt.axis("off")