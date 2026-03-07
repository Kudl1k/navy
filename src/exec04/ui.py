import tkinter as tk
from exec04 import QLearningMouseAndCheese
import time

CELL = 70


class App:
    def __init__(self, root):
        self.root = root
        self.env = QLearningMouseAndCheese()
        self.mode = "hole"

        self.root.configure(bg="#f3f4f6")

        self.create_controls()

        self.canvas = tk.Canvas(
            root,
            width=self.env.grid_size * CELL,
            height=self.env.grid_size * CELL,
            bg="white",
            highlightthickness=1,
            highlightbackground="#d1d5db"
        )
        self.canvas.pack(padx=12, pady=(0, 12))
        self.canvas.bind("<Button-1>", self.click)

        self.draw()

    def create_controls(self):
        container = tk.Frame(self.root, bg="#f3f4f6")
        container.pack(fill="x", padx=12, pady=12)

        mode_frame = tk.LabelFrame(
            container,
            text="Placement mode",
            bg="#f3f4f6",
            fg="#111827",
            padx=8,
            pady=8
        )
        mode_frame.pack(side="left", padx=(0, 10))

        self.mode_var = tk.StringVar(value=self.mode)
        tk.Radiobutton(
            mode_frame,
            text="Mouse",
            variable=self.mode_var,
            value="mouse",
            command=lambda: self.set_mode("mouse"),
            bg="#f3f4f6"
        ).pack(side="left", padx=4)

        tk.Radiobutton(
            mode_frame,
            text="Hole",
            variable=self.mode_var,
            value="hole",
            command=lambda: self.set_mode("hole"),
            bg="#f3f4f6"
        ).pack(side="left", padx=4)

        tk.Radiobutton(
            mode_frame,
            text="Cheese",
            variable=self.mode_var,
            value="cheese",
            command=lambda: self.set_mode("cheese"),
            bg="#f3f4f6"
        ).pack(side="left", padx=4)

        action_frame = tk.LabelFrame(
            container,
            text="Actions",
            bg="#f3f4f6",
            fg="#111827",
            padx=8,
            pady=8
        )
        action_frame.pack(side="left", padx=(0, 10))

        tk.Button(action_frame, text="Train", command=self.train, width=10).pack(side="left", padx=3)
        tk.Button(action_frame, text="Show Path", command=self.animate_path, width=10).pack(side="left", padx=3)
        tk.Button(action_frame, text="Policy", command=self.show_policy, width=10).pack(side="left", padx=3)
        tk.Button(action_frame, text="Reset", command=self.reset, width=10).pack(side="left", padx=3)

        grid_frame = tk.LabelFrame(
            container,
            text="Grid",
            bg="#f3f4f6",
            fg="#111827",
            padx=8,
            pady=8
        )
        grid_frame.pack(side="left")

        tk.Label(grid_frame, text="Size:", bg="#f3f4f6").pack(side="left", padx=(0, 4))
        self.grid_size_var = tk.StringVar(value=str(self.env.grid_size))
        tk.Entry(grid_frame, textvariable=self.grid_size_var, width=5).pack(side="left", padx=(0, 6))
        tk.Button(grid_frame, text="Apply", command=self.apply_grid_size).pack(side="left")

        self.status_var = tk.StringVar(value="Ready. Click cells to place objects.")
        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            bg="#f3f4f6",
            fg="#374151"
        )
        status.pack(fill="x", padx=12, pady=(0, 8))

    def set_mode(self, mode):
        self.mode = mode
        self.mode_var.set(mode)
        self.status_var.set(f"Mode: {mode}")

    def apply_grid_size(self):
        text = self.grid_size_var.get().strip()

        if not text.isdigit():
            self.status_var.set("Grid size must be a positive integer.")
            return

        size = int(text)
        if size < 2 or size > 20:
            self.status_var.set("Grid size must be between 2 and 20.")
            return

        self.env.set_grid_size(size)
        self.resize_canvas()
        self.draw()
        self.status_var.set(f"Grid resized to {size}x{size}.")

    def resize_canvas(self):
        self.canvas.config(width=self.env.grid_size * CELL, height=self.env.grid_size * CELL)

    def click(self, event):
        x = event.y // CELL
        y = event.x // CELL

        if x < 0 or y < 0 or x >= self.env.grid_size or y >= self.env.grid_size:
            return

        if self.mode == "mouse":
            self.env.place_mouse(x, y)
        elif self.mode == "hole":
            self.env.toggle_grid_item(x, y, self.env.HOLE)
        elif self.mode == "cheese":
            self.env.toggle_grid_item(x, y, self.env.CHEESE)

        self.draw()

    def draw(self):
        self.canvas.delete("all")

        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                color = "white"

                if self.env.grid[x, y] == self.env.HOLE:
                    color = "#111827"
                if self.env.grid[x, y] == self.env.CHEESE:
                    color = "#fde047"

                self.canvas.create_rectangle(
                    y * CELL,
                    x * CELL,
                    (y + 1) * CELL,
                    (x + 1) * CELL,
                    fill=color,
                    outline="#9ca3af"
                )

        mx, my = self.env.state
        self.canvas.create_oval(
            my * CELL + 15,
            mx * CELL + 15,
            (my + 1) * CELL - 15,
            (mx + 1) * CELL - 15,
            fill="#ef4444",
            outline=""
        )

    def train(self):
        self.env.find(episodes=2000)
        self.status_var.set("Training finished (2000 episodes).")

    def animate_path(self):
        path = self.env.best_path()
        self.status_var.set(f"Animating path with {len(path)} steps...")

        for (x, y) in path:
            self.draw()
            self.canvas.create_rectangle(
                y * CELL + 20,
                x * CELL + 20,
                (y + 1) * CELL - 20,
                (x + 1) * CELL - 20,
                fill="#3b82f6",
                outline=""
            )
            self.root.update()
            time.sleep(0.2)

        self.status_var.set("Path animation complete.")

    def show_policy(self):
        self.draw()
        policy = self.env.get_policy()

        arrows = {0: "^", 1: "v", 2: "<", 3: ">"}

        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                if self.env.grid[x, y] == self.env.EMPTY_SPACE:
                    arrow = arrows[int(policy[x, y])]
                    self.canvas.create_text(
                        y * CELL + CELL / 2,
                        x * CELL + CELL / 2,
                        text=arrow,
                        font=("Arial", 18, "bold"),
                        fill="#2563eb"
                    )

        self.status_var.set("Policy shown for empty cells.")

    def reset(self):
        self.env.set_grid_size(self.env.grid_size)
        self.draw()
        self.status_var.set("Grid reset.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Mouse and Cheese Q-Learning")
    App(root)
    root.mainloop()
