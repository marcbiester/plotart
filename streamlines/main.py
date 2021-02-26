from streamlines import streamLines
import numpy as np
import matplotlib.pyplot as plt
import datetime

grid = {
    "x_start": -5,
    "x_end": 5,
    "y_start": -3.5,
    "y_end": 3.5,
    "no_points_x": 100,
    "no_points_y": 100,
}
pic = 5
while True:
    try:
        t1 = datetime.datetime.now()
        self = streamLines(grid)

        for i in range(np.random.randint(1, 200)):
            x, y = np.random.uniform(-5, 5), np.random.uniform(-3.5, 3.5)
            self.add_vortex(np.random.uniform(-10, 10), x, y)
            self.add_source_sink(np.random.uniform(-3, 3), x, y)

        for i in range(np.random.randint(1, 200)):
            x, y = np.random.uniform(-5, 5), np.random.uniform(-3.5, 3.5)
            self.add_source_sink(np.random.uniform(-3, 3), x, y)

        plt.figure(figsize=(420 / 25.4, 297 / 25.4))
        random_seed_setting = np.random.choice(['random','sources','grid'], np.random.randint(1,4)).tolist()
        self.calc_streamtraces(
            n_streamtraces=np.random.randint(100, 1000),
            seeds=random_seed_setting,
            dt=0.001,
            maxiter=np.random.randint(200, 8000),
            radius=np.random.uniform(0.02, 0.2),
            n_cpu=4,
        )
        t2 = datetime.datetime.now()
        print(
            f"it took {t2-t1} to calculate the flow field and the stream traces for {pic}"
        )

        for ps in self.streamtraces:
            plt.plot([x[0] for x in ps], [x[1] for x in ps], color="black", alpha=0.5)
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{pic}.svg")
        plt.savefig(f"{pic}.png")

        t3 = datetime.datetime.now()
        print(f"it took {t3-t2} to create plot {pic}")
    except Exception as ex:
        print(ex)
    pic += 1
