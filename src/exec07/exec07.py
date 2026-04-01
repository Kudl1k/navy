# PRI0192
import random
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np

class IFS:
    def __init__(self, model_mat: list[list[float]]):
        self.model = model_mat

    @staticmethod
    def __draw_graph(points, title: str):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        ax.scatter(xs, ys, zs, s=0.2)
        ax.set_title(title)
        plt.show()

    def simulate(self, num_iterations: int, *, show_plot: bool = True, title: str = "IFS"):
        position = np.array([0.0, 0.0, 0.0])
        position_history = [position]

        for _ in range(num_iterations):
            r = random.choice(self.model)
            m = np.array([
                [r[0], r[1], r[2]],
                [r[3], r[4], r[5]],
                [r[6], r[7], r[8]],
            ])

            v = np.array([r[9], r[10], r[11]])
            position = m @ position + v
            position_history.append(position)

        if show_plot:
            self.__draw_graph(position_history, title=title)

        return position_history

first_model = [
    [0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
    [0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00],
    [-0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00],
    [0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00]
]

second_model = [
    [0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
    [0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45, 0.00, 1.00, 0.00],
    [-0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22, -0.22, 0.45, 0.00, 1.25, 0.00],
    [0.49, -0.08, 0.08, 0.08, 0.49, 0.08, 0.08, -0.08, 0.49, 0.00, 2.00, 0.00]
]

MODEL_MAP = {
    "Model 1": first_model,
    "Model 2": second_model,
}


class IFSApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("IFS Generator")
        self.root.resizable(False, False)

        self.model_var = tk.StringVar(value="Model 1")
        self.iterations_var = tk.StringVar(value="10000")

        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Model").grid(row=0, column=0, sticky="w", pady=(0, 6))
        model_box = ttk.Combobox(
            frame,
            textvariable=self.model_var,
            values=list(MODEL_MAP.keys()),
            state="readonly",
            width=18,
        )
        model_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(frame, text="Iterations").grid(row=2, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(frame, textvariable=self.iterations_var, width=20).grid(
            row=3, column=0, sticky="ew", pady=(0, 12)
        )

        ttk.Button(frame, text="Generate", command=self.on_generate).grid(
            row=4, column=0, sticky="ew"
        )

    def on_generate(self):
        model_name = self.model_var.get()
        model = MODEL_MAP.get(model_name)
        if model is None:
            messagebox.showerror("Invalid model", "Please select a valid model.")
            return

        try:
            iterations = int(self.iterations_var.get().strip())
            if iterations <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid iterations", "Iterations must be a positive integer.")
            return

        ifs = IFS(model)
        ifs.simulate(iterations, title=f"{model_name} ({iterations} iterations)")

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    app = IFSApp()
    app.run()
