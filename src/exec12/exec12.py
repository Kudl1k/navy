# PRI0192
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import ListedColormap


DEFAULT_MAP_SIZE = (100, 100)
DEFAULT_FRAMES = 240
DEFAULT_FPS = 24
DEFAULT_INTERVAL_MS = 40
DEFAULT_DPI = 90

REGROW_PROBABILITY = 0.05
SELF_IGNITION_PROBABILITY = 0.001
INITIAL_TREE_DENSITY = 0.5


class Neighborhood(Enum):
    VON_NEUMANN = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    MOORE = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


class Cell(Enum):
    EMPTY = 0
    TREE = 1
    BURNING = 2
    BURNT = 3


class ForestFire:
    def __init__(self, map_size: tuple[int, int], neighborhood: Neighborhood, title: str = "Forest Fire"):
        self.color_map = ListedColormap([
            "#522600",  # EMPTY
            "#009c00",  # TREE
            "#ff8000",  # BURNING
            "#000000",  # BURNT
        ])

        self.regrow_probability = REGROW_PROBABILITY
        self.self_ignition_probability = SELF_IGNITION_PROBABILITY
        self.initial_tree_density = INITIAL_TREE_DENSITY

        self.neighborhood = neighborhood
        self.neighbor_offsets = neighborhood.value
        self.grid = self._generate_initial_grid(map_size)

        self.fig, self.ax = plt.subplots()
        self.image = self.ax.imshow(self.grid, cmap=self.color_map, vmin=0, vmax=3)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title)

    def _generate_initial_grid(self, map_size: tuple[int, int]) -> np.ndarray:
        tree_mask = np.random.random(map_size) < self.initial_tree_density
        grid = np.full(map_size, Cell.EMPTY.value, dtype=np.uint8)
        grid[tree_mask] = Cell.TREE.value
        return grid

    def _burning_neighbor_mask(self, burning_mask: np.ndarray) -> np.ndarray:
        rows, cols = burning_mask.shape
        result = np.zeros_like(burning_mask, dtype=bool)

        for d_row, d_col in self.neighbor_offsets:
            dst_row_start = max(0, -d_row)
            dst_row_end = rows - max(0, d_row)
            dst_col_start = max(0, -d_col)
            dst_col_end = cols - max(0, d_col)

            src_row_start = dst_row_start + d_row
            src_row_end = dst_row_end + d_row
            src_col_start = dst_col_start + d_col
            src_col_end = dst_col_end + d_col

            result[dst_row_start:dst_row_end, dst_col_start:dst_col_end] |= burning_mask[
                src_row_start:src_row_end, src_col_start:src_col_end
            ]

        return result

    def _step(self) -> np.ndarray:
        current = self.grid
        next_grid = current.copy()

        empty_mask = current == Cell.EMPTY.value
        tree_mask = current == Cell.TREE.value
        burning_mask = current == Cell.BURNING.value
        burnt_mask = current == Cell.BURNT.value

        regrow_tree = np.random.random(current.shape) < self.regrow_probability
        self_ignite = np.random.random(current.shape) < self.self_ignition_probability
        neighbor_burning = self._burning_neighbor_mask(burning_mask)

        # 1) burnt cells disappear,
        # 2) burning cells become burnt,
        # 3) trees can catch fire,
        # 4) empty cells can regrow.
        next_grid[burnt_mask] = Cell.EMPTY.value
        next_grid[burning_mask] = Cell.BURNT.value
        next_grid[tree_mask & (neighbor_burning | self_ignite)] = Cell.BURNING.value
        next_grid[empty_mask & regrow_tree] = Cell.TREE.value

        return next_grid

    def run(
        self,
        interval: int = DEFAULT_INTERVAL_MS,
        frames: int = DEFAULT_FRAMES,
        fps: int = DEFAULT_FPS,
        save_video: bool = True,
        show_preview: bool = False,
        output_file: Optional[str] = None,
    ):
        def update_animation_frame(_):
            self.grid = self._step()
            self.image.set_data(self.grid)
            return (self.image,)

        animation = FuncAnimation(
            self.fig,
            update_animation_frame,
            frames=frames,
            interval=interval,
            blit=True,
            cache_frame_data=False,
        )

        if save_video:
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                extra_args=["-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p"],
            )
            file_name = output_file or f"{self.ax.get_title()}.mp4"
            animation.save(file_name, writer=writer, dpi=DEFAULT_DPI)

        if show_preview:
            plt.show()
        else:
            plt.close(self.fig)


def run_simulation(map_size: tuple[int, int], neighborhood: Neighborhood, title: str) -> None:
    sim = ForestFire(map_size, neighborhood, title)
    sim.run(frames=DEFAULT_FRAMES, fps=DEFAULT_FPS, save_video=True, show_preview=False)


if __name__ == "__main__":
    from multiprocessing import Process, set_start_method

    set_start_method("spawn", force=True)

    p1 = Process(target=run_simulation, args=(DEFAULT_MAP_SIZE, Neighborhood.MOORE, "Moore"))
    p2 = Process(target=run_simulation, args=(DEFAULT_MAP_SIZE, Neighborhood.VON_NEUMANN, "Von Neumann"))

    p1.start()
    p2.start()

    p1.join()
    p2.join()