from blobs import Blob
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt


circle = Point(2.5, 2.5).buffer(2)
rectangle = Polygon([(0, 0), (0, 21), (29, 21), (29, 0)])
rectangle = rectangle.buffer(4)

self = Blob(r_p=3, delta=0.1, size=30, n_low=300, n_up=300, l_min=0.05, mask=rectangle)
self._create_blobs()
self._grow_blobs()
ax = self._plot_blobs()
plt.savefig(
    "Blob_size"
    + str(self.size)
    + "_rp_"
    + str(self.r_p)
    + "_nlow_"
    + str(self.n_low)
    + "_nup_"
    + str(self.n_up)
    + "_delta_"
    + str(self.delta).replace(".", "")
    + ".svg"
)
print(
    "Blob_size"
    + str(self.size)
    + "_rp_"
    + str(self.r_p)
    + "_nlow_"
    + str(self.n_low)
    + "_nup_"
    + str(self.n_up)
    + "_delta_"
    + str(self.delta).replace(".", "")
    + ".svg"
)
