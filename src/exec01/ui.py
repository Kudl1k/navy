import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from exec01 import Perceptron
import re

from src.help import save_figure


class PerceptronGUI:
    def __init__(self, root):
        self.fig = None
        self.root = root
        self.root.title("Perceptron Classifier")
        self.root.geometry("1000x700")
        self.update_id = None

        # Control frame
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH)

        # Learning Rate
        ttk.Label(control_frame, text="Learning Rate").pack()
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Scale(control_frame, from_=0.01, to=1.0, variable=self.lr_var, command=self.debounce_update).pack(fill=tk.X)
        self.lr_label = ttk.Label(control_frame, text="0.1")
        self.lr_label.pack()

        # Iterations
        ttk.Label(control_frame, text="Iterations").pack(pady=(10, 0))
        self.iter_var = tk.IntVar(value=100)
        ttk.Scale(control_frame, from_=1, to=100, variable=self.iter_var, command=self.debounce_update).pack(fill=tk.X)
        self.iter_label = ttk.Label(control_frame, text="100")
        self.iter_label.pack()

        # Samples
        ttk.Label(control_frame, text="Training Samples").pack(pady=(10, 0))
        self.samples_var = tk.IntVar(value=100)
        ttk.Scale(control_frame, from_=20, to=200, variable=self.samples_var, command=self.debounce_update).pack(
            fill=tk.X)
        self.samples_label = ttk.Label(control_frame, text="100")
        self.samples_label.pack()

        # Linear Function Input
        ttk.Label(control_frame, text="Linear Function").pack(pady=(10, 0))
        self.func_entry = ttk.Entry(control_frame, width=20)
        self.func_entry.insert(0, "y = 3*x + 2")
        self.func_entry.pack(fill=tk.X)
        self.func_entry.bind("<Return>", lambda e: self.update_plot())
        
        #Save chart button
        save_button = ttk.Button(control_frame, text="Save Chart", command=lambda: self.save_chart())
        save_button.pack(pady=(20, 0))

        # Canvas for plot
        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.update_plot()
        
    def save_chart(self):
        if (self.fig is None): return
        save_figure("exec01", f"perceptron_chart_lr{self.lr_var.get():.2f}_iter{self.iter_var.get()}_samples{self.samples_var.get()}.png", self.fig)

    def debounce_update(self, *args):
        # Update labels immediately
        self.lr_label.config(text=f"{self.lr_var.get():.2f}")
        self.iter_label.config(text=str(self.iter_var.get()))
        self.samples_label.config(text=str(self.samples_var.get()))

        # Debounce plot update
        if self.update_id:
            self.root.after_cancel(self.update_id)
        self.update_id = self.root.after(300, self.update_plot)

    def parse_linear_function(self, func_str):
        try:
            func_str = func_str.replace(" ", "").replace("y=", "")
            matches = re.findall(r'[+-]?[^+-]+', func_str)
            slope, intercept = 0, 0
            for match in matches:
                if 'x' in match:
                    slope = float(match.replace('x', '').replace('*', '') or '1')
                else:
                    intercept = float(match)
            return slope, intercept
        except:
            return None, None

    def update_plot(self, *args):
        self.update_id = None

        # Update labels
        self.lr_label.config(text=f"{self.lr_var.get():.2f}")
        self.iter_label.config(text=str(self.iter_var.get()))
        self.samples_label.config(text=str(self.samples_var.get()))

        slope, intercept = self.parse_linear_function(self.func_entry.get())
        if slope is None or intercept is None:
            return

        np.random.seed(42)
        X = np.random.uniform(-10, 10, (self.samples_var.get(), 2))
        y = np.array([1 if p[1] > (slope * p[0] + intercept) else -1 for p in X])

        perceptron = Perceptron(X[0], l_r=self.lr_var.get(), n_iter=self.iter_var.get())

        # Track convergence
        initial_weights = perceptron.weights.copy()
        perceptron.train(X, y)
        final_weights = perceptron.weights.copy()

        # Print convergence info (optional)
        print(
            f"Iterations: {self.iter_var.get()}, Weight change: {np.linalg.norm(final_weights - initial_weights):.4f}")

        self.fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

        x_line = np.linspace(-10, 10, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'y--', linewidth=2, label=f'y = {slope}*x + {intercept}')

        w0, w1, w2 = perceptron.weights
        p_y_vals = -(w1 * x_line + w0) / w2
        ax.plot(x_line, p_y_vals, 'k-', linewidth=2, label='Perceptron Decision Boundary')

        ax.set_title(f"Perceptron Classification ({self.iter_var.get()} iterations | lr={self.lr_var.get():.2f})\n Needed {perceptron.iteration_count} iterations to converge")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        ax.legend()

        # Clear previous canvas
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(self.fig)




if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronGUI(root)
    root.mainloop()
