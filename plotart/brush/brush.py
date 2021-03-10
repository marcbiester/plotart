import numpy as np
from PIL import Image
from scipy.spatial import distance_matrix

class brushLines:
    def __init__(self, pic, no_points=500, min_dist=0.001):
        self.pic = pic
        self.r_p = no_points
        self.l_min=min_dist
        self.seeds=[]
        self.pic_shape=[]

    def seed(self, replace_zeros_with = 5):
        img = Image.open(self.pic).convert("L")
        img = np.array(img)
        img=img
        self.pic_shape=np.array(img.shape)
        img = 255 - img
        img[img == 0] = replace_zeros_with
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
                        random_points = [(x[0], x[1]) for x in adjusted_index]
            else:
                random_points = np.append(
                    random_points,
                    [(x[0], x[1]) for x in adjusted_index],
                    axis=0)
            random_points = self._check_distance(random_points)
            num_loops += 1
        if num_loops == 1000:
            print(
                f"Warning. Exceeded max number of seeding iterations ({num_loops}). Only having {len(random_points)} seeds. Reduce l_min to get more seeding."
            )
        self.seeds = random_points

    
    def write_svg(self, line_length=4):
        print("writing to svg ...", end="")

        f = open(self.pic+"_out.svg", "w")
        f.write(f'<svg height="{self.pic_shape[0]}" width="{self.pic_shape[1]}">')

        d=""

        for seed in self.seeds:
            #d += f'<circle cx="{seed[1]}" cy="{seed[0]}" r="5" fill="none" stroke="black" />\n'
            d +=  f'<line x1="{seed[1]}" y1="{seed[0]}" x2="{seed[1]}" y2="{seed[0] - line_length}" stroke="black" />'
        f.write(d)
        f.write("</svg>")
        f.close()
        print("... done", end="")



    def _check_distance(self, points):
        # calculate distance matrix
        distance_triu = np.triu(distance_matrix(points, points))
        # identify points being too close to each other (i.e. 2 * l_min)
        close_points = np.argwhere(
            (distance_triu > 0) & (distance_triu < 2 * self.l_min)
        )
        # remove points being too close and return remaining
        return np.delete(points, close_points[:, 0], 0)