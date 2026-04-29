from typing import Any

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib import colors as mcolors
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

WIDTH, HEIGHT = 700, 700
MAX_ITER = 200
ZOOM_IN_FACTOR = 0.5
ZOOM_OUT_FACTOR = 2.0
MANDELBROT_BOUNDS = (-2.0, 1.0, -1.0, 1.0)
JULIA_BOUNDS = (-1.5, 1.5, -1.5, 1.5)
#https://en.wikipedia.org/wiki/Julia_set
JULIA_C_DEFAULT = complex(-0.7269, 0.1889)


def mandelbrot_set(x_min, x_max, y_min, y_max, width, height, max_iter):
    # Vytvorime mrizku komplexnich cisel c = x + i*y pro kazdy pixel obrazku.
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    c = x + y[:, None] * 1j

    # Pro Mandelbrot plati z0 = 0, proto z zacina vsude nulou.
    z = np.zeros_like(c)
    # Predvyplnime max_iter: body, ktere neutecou, zustanou na teto hodnote.
    result = np.full(c.shape, max_iter, dtype=np.int32)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        # Iterace zn+1 = zn^2 + c, ale jen pro body, ktere jeste neutekly.
        z[mask] = z[mask] * z[mask] + c[mask]
        # Utek nastane pri |z| > 2 (escape radius m = 2 podle zadani).
        escaped = np.abs(z) > 2.0
        newly_escaped = escaped & mask
        # Ulozime iteraci, ve ktere bod poprve utekl.
        result[newly_escaped] = i
        mask &= ~escaped

        # Kdyz utekly vsechny body, nema smysl pokracovat.
        if not mask.any():
            break

    return result


def julia_set(x_min, x_max, y_min, y_max, width, height, max_iter, julia_c):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    # U Julie je promenna pocatecni hodnota z0 = x + i*y.
    z = x + y[:, None] * 1j

    result = np.full(z.shape, max_iter, dtype=np.int32)
    mask = np.ones(z.shape, dtype=bool)

    for i in range(max_iter):
        # Iterace zn+1 = zn^2 + c, kde c je zde fixni konstanta julia_c.
        z[mask] = z[mask] * z[mask] + julia_c
        escaped = np.abs(z) > 2.0
        newly_escaped = escaped & mask
        result[newly_escaped] = i
        mask &= ~escaped

        if not mask.any():
            break

    return result


def compute_hsv_coloring(data: np.ndarray, max_iter: int) -> np.ndarray:
    normalized = data.astype(np.float32) / float(max_iter)
    hue = normalized
    saturation = np.where(data == max_iter, 0.0, 1.0)
    value = np.where(data == max_iter, 0.0, 1.0)
    hsv = np.stack((hue, saturation, value), axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def zoom_to_center(x_min, x_max, y_min, y_max, zoom_factor, center_x, center_y):
    width = x_max - x_min
    height = y_max - y_min
    new_width = width * zoom_factor
    new_height = height * zoom_factor
    return (
        center_x - new_width / 2,
        center_x + new_width / 2,
        center_y - new_height / 2,
        center_y + new_height / 2,
    )


class MandelbrotApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Fractal Explorer")

        self.width = WIDTH
        self.height = HEIGHT
        self.max_iter = MAX_ITER

        self.fractal_mode = tk.StringVar(value="Mandelbrot")
        self.julia_c = JULIA_C_DEFAULT

        self.x_min, self.x_max, self.y_min, self.y_max = self.current_bounds()
        self.zoom_total = 1.0
        self.image = None
        self.last_animation = None

        self._build_ui()
        self.render()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")

        ttk.Label(controls, text="Set:").pack(side="left", padx=(0, 4))
        mode_selector = ttk.Combobox(
            controls,
            textvariable=self.fractal_mode,
            values=("Mandelbrot", "Julia"),
            state="readonly",
            width=12,
        )
        mode_selector.pack(side="left", padx=(0, 12))
        mode_selector.bind("<<ComboboxSelected>>", self.on_mode_change)

        ttk.Button(controls, text="Zoom In", command=self.zoom_in_center).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Zoom Out", command=self.zoom_out_center).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Reset", command=self.reset_view).pack(side="left", padx=(0, 12))
        ttk.Button(controls, text="Export GIF", command=self.export_zoom_gif).pack(side="left", padx=(0, 12))

        self.status_var = tk.StringVar(value="Left click: zoom in | Right click: zoom out")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left")

        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def current_bounds(self) -> tuple[float, float, float, float]:
        if self.fractal_mode.get() == "Julia":
            return JULIA_BOUNDS
        return MANDELBROT_BOUNDS

    def render(self) -> None:
        if self.fractal_mode.get() == "Julia":
            data = julia_set(
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.width,
                self.height,
                self.max_iter,
                self.julia_c,
            )
        else:
            data = mandelbrot_set(
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.width,
                self.height,
                self.max_iter,
            )
        rgb_image = compute_hsv_coloring(data, self.max_iter)
        self.ax.clear()
        self.image = self.ax.imshow(
            rgb_image,
            extent=(self.x_min, self.x_max, self.y_min, self.y_max),
            origin="lower",
            interpolation="nearest",
        )
        if self.fractal_mode.get() == "Julia":
            self.ax.set_title(
                f"Julia c={self.julia_c.real:+.3f}{self.julia_c.imag:+.3f}i - CPU (zoom {self.zoom_total:.2e})"
            )
        else:
            self.ax.set_title(f"Mandelbrot - CPU (zoom {self.zoom_total:.2e})")
        self.canvas.draw_idle()

    def create_animation(self, target_x: float, target_y: float, total_frames: int, final_scale: float):
        base_xmin, base_xmax, base_ymin, base_ymax = self.current_bounds()
        base_x_range = base_xmax - base_xmin
        base_y_range = base_ymax - base_ymin

        # Geometric spacing gives a visually smooth zoom speed.
        scales = np.geomspace(1.0, final_scale, total_frames)
        is_julia = self.fractal_mode.get() == "Julia"

        def update(frame_idx: int):
            scale = scales[frame_idx]
            x_range = base_x_range * scale
            y_range = base_y_range * scale
            xmin = target_x - x_range / 2.0
            xmax = target_x + x_range / 2.0
            ymin = target_y - y_range / 2.0
            ymax = target_y + y_range / 2.0

            if is_julia:
                data = julia_set(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    width=self.width,
                    height=self.height,
                    max_iter=self.max_iter,
                    julia_c=self.julia_c,
                )
            else:
                data = mandelbrot_set(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    width=self.width,
                    height=self.height,
                    max_iter=self.max_iter,
                )

            rgb_image = compute_hsv_coloring(data, self.max_iter)
            self.image.set_data(rgb_image)
            self.image.set_extent((xmin, xmax, ymin, ymax))
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            return (self.image,)

        anim = animation.FuncAnimation(self.fig, update, frames=total_frames, interval=40, blit=False)
        return anim

    def export_zoom_gif(self) -> None:
        output_path = filedialog.asksaveasfilename(
            title="Save zoom animation as GIF",
            defaultextension=".gif",
            filetypes=[("GIF animation", "*.gif")],
            initialfile=f"{self.fractal_mode.get().lower()}_zoom.gif",
        )
        if not output_path:
            return

        center_x = (self.x_min + self.x_max) / 2.0
        center_y = (self.y_min + self.y_max) / 2.0
        total_frames = 140
        final_scale = 1e-4

        try:
            self.status_var.set("Rendering GIF... this can take a moment")
            self.root.update_idletasks()

            # Keep a reference so matplotlib does not garbage-collect the animation too early.
            self.last_animation = self.create_animation(center_x, center_y, total_frames, final_scale)
            self.last_animation.save(output_path, writer="pillow", fps=25)
            self.status_var.set(f"GIF saved: {output_path}")
            messagebox.showinfo("Export complete", f"GIF saved to:\n{output_path}")
        except Exception as exc:
            self.status_var.set("GIF export failed")
            messagebox.showerror("Export failed", str(exc))

    def apply_zoom(self, zoom_factor: float, center_x: float, center_y: float) -> None:
        self.x_min, self.x_max, self.y_min, self.y_max = zoom_to_center(
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            zoom_factor,
            center_x,
            center_y,
        )
        self.zoom_total *= zoom_factor
        self.render()

    def zoom_in_center(self) -> None:
        center_x = (self.x_min + self.x_max) / 2.0
        center_y = (self.y_min + self.y_max) / 2.0
        self.apply_zoom(ZOOM_IN_FACTOR, center_x, center_y)

    def zoom_out_center(self) -> None:
        center_x = (self.x_min + self.x_max) / 2.0
        center_y = (self.y_min + self.y_max) / 2.0
        self.apply_zoom(ZOOM_OUT_FACTOR, center_x, center_y)

    def reset_view(self) -> None:
        self.x_min, self.x_max, self.y_min, self.y_max = self.current_bounds()
        self.zoom_total = 1.0
        self.render()

    def on_mode_change(self, _event: Any) -> None:
        self.reset_view()

    def on_click(self, event: Any) -> None:
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            self.apply_zoom(ZOOM_IN_FACTOR, float(event.xdata), float(event.ydata))
        elif event.button in (2, 3):
            self.apply_zoom(ZOOM_OUT_FACTOR, float(event.xdata), float(event.ydata))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = MandelbrotApp()
    app.run()


if __name__ == "__main__":
    main()


