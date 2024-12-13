import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RectanglePacker:
    def __init__(self, rect_target: Dict[str, int]):
        self.rect_target = rect_target

    def generate_random_rectangles(self, num_rectangles: int, h_range: List[int] = [1,10], w_range: List[int] = [1,10]) -> Dict[str, Dict[str, int]]:
        rect_sources = {}
        for i in range(num_rectangles):
            h = random.randint(h_range[0], h_range[1])
            w = random.randint(w_range[0], w_range[1])
            rect_sources[f"rect{i+1}"] = {"x": 0, "y": 0, "h": h, "w": w}
        return rect_sources

    @staticmethod
    def overlap(r1: Dict[str, int], r2: Dict[str, int]) -> bool:
        return (r1["x"] < r2["x"] + r2["w"] and
                r1["x"] + r1["w"] > r2["x"] and
                r1["y"] < r2["y"] + r2["h"] and
                r1["y"] + r1["h"] > r2["y"])

    def can_fit(self, r: Dict[str, int], rects: List[Dict[str, int]]) -> bool:
        for x in range(self.rect_target["w"] - r["w"] + 1):
            for y in range(self.rect_target["h"] - r["h"] + 1):
                temp_r = r.copy()
                temp_r["x"] = x
                temp_r["y"] = y
                if not any(self.overlap(temp_r, rect) for rect in rects):
                    r["x"] = x
                    r["y"] = y
                    return True
        return False


    def optimize_rectangles(self, rect_sources: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        rects = list(rect_sources.values())
        rects.sort(key=lambda r: r["h"] * r["w"], reverse=True)

        solution = []
        for r in rects:
            if self.can_fit(r, solution):
                solution.append(r)

        return {k: v for k, v in rect_sources.items() if v in solution}

    def plot_rectangles(self, solution: Dict[str, Dict[str, int]]) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, self.rect_target["w"])
        ax.set_ylim(0, self.rect_target["h"])

        target_rect_patch = patches.Rectangle((self.rect_target["x"], self.rect_target["y"]),
                                              self.rect_target["w"], self.rect_target["h"],
                                              linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(target_rect_patch)

        for i, rect in enumerate(solution.values()):
            rect_patch = patches.Rectangle((rect["x"], rect["y"]), rect["w"], rect["h"],
                                           linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3)
            ax.add_patch(rect_patch)
            ax.text(rect["x"] + rect["w"] / 2, rect["y"] + rect["h"] / 2, f"R{i + 1}",
                    horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')

        plt.show()

# rect_target = {"x": 0, "y": 0, "h": 21, "w": 29}
# packer = RectanglePacker(rect_target)

# rect_sources = packer.generate_random_rectangles(100)
# print("Source rectangles:", rect_sources)
# print("Target rectangle:", rect_target)

# solution = packer.optimize_rectangles(rect_sources)
# print("Solution:", solution)

# packer.plot_rectangles(solution)

