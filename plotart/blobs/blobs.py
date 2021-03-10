import numpy as np
from scipy.spatial import distance_matrix
from shapely.geometry import Polygon, LineString, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as patch_poly
from PIL import Image


class Blob:
    def __init__(
        self,
        r_p=20,
        l_min=0.1,
        n_low=6,
        n_up=6,
        delta=0.1,
        size=10,
        clearance=0,
        pic=[],
        pic_shape=[],
        max_iter=0,
        mask=[],
    ):
        """Class to generate Blob-like patterns. If pic is set, pic will be used for seeding, size will limit size of blobs. If mask is set, mask is used for seeding ans as border for blobs"""
        self.r_p = r_p
        if r_p < 2:
            raise Exception("r_p must be at least 2")
        self.l_min = l_min
        self.n_low = n_low
        if n_up < self.n_low:
            raise Exception("n_up must be larger than or equal to n_low")
        else:
            self.n_up = n_up
        self.clearance = clearance
        self.delta = delta
        self.size = size
        self.pic = pic
        self.max_iter = max_iter
        self.mask = mask
        if self.mask and self.pic:
            raise Exception(
                "either mask or pic can be set. if both are set, mask will be ignored (currently)"
            )
        self.seeds = []
        self.blobs = []

    def _seeds_from_pic(self):
        img = Image.open(self.pic).convert("L")
        img = np.array(img)
        self.pic_shape=np.array(img.shape)/img.shape[0]*self.size
        img = 255 - img
        img[img == 0] = 15
        img = img / np.sum(img)

        num_loops = 0
        random_points=[]
        flat = img.flatten()

        while len(random_points) < self.r_p and num_loops < 1000:
            sample_index = np.random.choice(
                a=flat.size, p=flat, size=self.r_p - len(random_points)
            )
            adjusted_index = np.unravel_index(sample_index, img.shape)
            adjusted_index = list(zip(*adjusted_index))
            if num_loops == 0:
                        random_points = [(x[0] / img.shape[0]*self.size, x[1] / img.shape[0]*self.size) for x in adjusted_index]
            else:
                random_points = np.append(
                    random_points,
                    [(x[0] / img.shape[0], x[1] / img.shape[0]) for x in adjusted_index],
                    axis=0)
            random_points = self._check_distance(random_points)
            num_loops += 1
        if num_loops == 1000:
            print(
                f"Warning. Exceeded max number of seeding iterations ({num_loops}). Only having {len(random_points)} seeds. Reduce l_min to get more seeding."
            )
        self.seeds = random_points

    def _check_distance(self, points):
        # calculate distance matrix
        distance_triu = np.triu(distance_matrix(points, points))
        # identify points being too close to each other (i.e. 2 * l_min)
        close_points = np.argwhere(
            (distance_triu > 0) & (distance_triu < 2 * self.l_min + self.clearance)
        )
        # remove points being too close and return remaining
        return np.delete(points, close_points[:, 0], 0)

    def _seed(self):
        # create random points
        random_points = np.random.uniform(
            self.l_min + self.clearance, self.size - self.l_min - self.clearance
        )
        # remove points closer togesther as 2*l_min
        random_points = self._check_distance(random_points)
        # fill up array until number of points equals r_p
        num_loops = 0
        while len(random_points) < self.r_p and num_loops < 1000:
            random_points = np.append(
                random_points,
                np.random.uniform(
                    self.l_min + self.clearance, self.size - self.l_min - self.clearance
                ),
                axis=0,
            )
            random_points = self._check_distance(random_points)
            num_loops += 1
        if num_loops == 1000:
            print(
                f"Warning. Exceeded max number of seeding iterations ({num_loops}). Only having {len(random_points)} seeds. Reduce l_min to get more seeding."
            )
        self.seeds = random_points

    def _seed_from_polygon(self):
        # from https://gis.stackexchange.com/questions/207731/generating-random-coordinates-in-multipolygon-in-python
        random_points = []
        minx, miny, maxx, maxy = self.mask.bounds
        minx += self.l_min + self.clearance
        maxx -= self.l_min + self.clearance
        miny += self.l_min + self.clearance
        maxy -= self.l_min + self.clearance
        num_loops = 0
        while len(random_points) < self.r_p:
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if self.mask.contains(pnt):
                random_points.append([pnt.x, pnt.y])
            random_points = self._check_distance(
                random_points
            ).tolist()  # TODO: Fix this more elegantly
            num_loops += 1
        if num_loops == 1000:
            print(
                f"Warning. Exceeded max number of seeding iterations ({num_loops}). Only having {len(random_points)} seeds. Reduce l_min to get more seeding."
            )
        print(num_loops)
        self.seeds = random_points

    def _create_blobs(self):
        if self.pic:
            self._seeds_from_pic()
        elif self.mask:
            self._seed_from_polygon()
        else:
            self._seed()

        # create blobs. Blob structure is x,y,l_1, ..., l_n, exhausted-flag. l_1 meaning distance of point to center for equally distributed points
        for p in self.seeds:
            x = p[0]
            y = p[1]
            # make things more interesting by adding a random turning
            a = np.random.rand() * np.pi / 2
            # set number of edges of body
            n_edges = self.n_low
            # if bounds are given, reset edges
            if self.n_low != self.n_up:
                n_edges = np.random.randint(self.n_low, self.n_up)

            blob = []
            blob.extend([x, y, a])
            blob.extend(n_edges * [self.l_min])
            blob.append(False)
            self.blobs.append(blob)

    def _blob2coords(self, blob):
        n_edges = len(blob) - 4
        dalpha = 2 * np.pi / n_edges
        coords = []
        for edge in range(n_edges):
            alpha = edge * dalpha + blob[2]
            x, y = (
                blob[0] + blob[3 + edge] * np.cos(alpha),
                blob[1] + blob[3 + edge] * np.sin(alpha),
            )
            coords.append([x, y])
        return coords

    def _check_collision(self, blob, polygon_blobs):
        blob = self._blob2coords(blob)
        blob = Polygon(blob)
        blob = blob.buffer(self.clearance / 2, join_style=2)

        if self.mask:
            borders = self.mask
        elif self.pic:
            borders = Polygon(
                [(0, 0), (0, self.pic_shape[1]), (self.pic_shape[0], self.pic_shape[1]), (self.pic_shape[0], 0)]
            )
        else:
            borders = Polygon(
                [(0, 0), (0, self.size), (self.size, self.size), (self.size, 0)]
            )

            
        collision = False
        for other_blob in polygon_blobs:
            if blob.intersects(other_blob) or not blob.within(borders):
                collision = True
                break

        return collision

    def _grow_blobs(self, iter2png=False):
        n = 0
        max_iter = 999999 if self.max_iter == 0 else self.max_iter
        while sum([x[-1] for x in self.blobs]) != len(self.blobs) and n < max_iter:
            n += 1
            print(n)
            for i in range(len(self.blobs)):
                blob = self.blobs[i]
                if not blob[-1]:
                    other_blobs = np.delete(self.blobs, i, 0)
                    other_blobs_polygon = []
                    for b in other_blobs:
                        other_blobs_polygon.append(Polygon(self._blob2coords(b)))
                    n_edges = len(blob) - 3
                    col_edges = 0
                    for edge in range(n_edges):
                        # avoid too large geometries
                        if (
                            blob[2 + edge] + self.delta <= self.size
                        ):  # / np.sqrt(self.r_p):
                            blob[2 + edge] += self.delta
                            if self._check_collision(blob, other_blobs_polygon):
                                col_edges += 1
                                blob[2 + edge] -= self.delta
                            # check if all changes on edges lead to collision (exhausted)
                        else:
                            col_edges += 1
                        if n_edges == col_edges:
                            blob[-1] = True
                self.blobs[i] = blob
            if iter2png:
                self.blobs2png(n)

    def _plot_blobs(self):
        _, ax = plt.subplots(figsize=(12, 12))

        for blob in self.blobs:
            ax.add_patch(patch_poly(self._blob2coords(blob), closed=True, fill=False))

        ax.axis("equal")
        ax.autoscale(enable=True, axis="both")
        ax.axis("off")
        return ax

    def blobs2png(self, n):
        ax = self._plot_blobs()
        plt.savefig(f"pixel_{n:05d}.png")
        plt.savefig(f"vector_{n:05d}.svg")
        plt.close("all")
