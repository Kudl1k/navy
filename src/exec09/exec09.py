from __future__ import annotations

import argparse
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import colorchooser, messagebox, ttk


# Dark, dashboard-style palette to make this UI visually different.
APP_BG = "#0F172A"
PANEL_BG = "#111827"
CANVAS_BG = "#E2E8F0"
TEXT_MAIN = "#E5E7EB"
TEXT_MUTED = "#9CA3AF"
ACCENT = "#22C55E"
ACCENT_ALT = "#3B82F6"
ERROR = "#EF4444"


@dataclass
class TerrainConfig:
	# Jednoducha konfigurace jedne vrstvy terenu.
	start_x: float
	start_y: float
	end_x: float
	end_y: float
	iterations: int
	offset: float
	color: str


DEFAULT_PRESETS: list[TerrainConfig] = [
	TerrainConfig(0, 450.5, 901, 450.5, 6, 10, "#0F172A"),
	TerrainConfig(0, 220, 901, 450.5, 10, 12, "#16A34A"),
	TerrainConfig(0, 580, 901, 580, 5, 10, "#1E293B"),
	TerrainConfig(0, 720, 901, 720, 4, 9, "#92400E"),
]


class TerrainGenerator:
	@staticmethod
	def generate_points(config: TerrainConfig, rng: random.Random | None = None) -> list[tuple[float, float]]:
		# Midpoint displacement: kazdy usek se rozdeli v polovine a stred se nahodne posune.
		rng = rng or random
		points = [(config.start_x, config.start_y), (config.end_x, config.end_y)]

		for _ in range(config.iterations):
			subdivided: list[tuple[float, float]] = []
			for i in range(len(points) - 1):
				p1 = points[i]
				p2 = points[i + 1]
				mid_x = (p1[0] + p2[0]) / 2.0
				displacement = rng.uniform(-config.offset, config.offset) * 1.2
				mid_y = (p1[1] + p2[1]) / 2.0 + displacement
				subdivided.append(p1)
				subdivided.append((mid_x, mid_y))

			subdivided.append(points[-1])
			# Dalsi iterace pracuje s jemnejsi krivkou.
			points = subdivided

		return points


class TerrainApp:
	def __init__(self, root: tk.Tk):
		# Inicializace okna a zakladniho stavu aplikace.
		self.root = root
		self.root.title("Terrain Studio")
		self.root.geometry("1320x960")
		self.root.configure(bg=APP_BG)
		self.root.minsize(1120, 760)

		self.canvas_width = 920
		self.canvas_height = 900
		self.selected_color = "#16A34A"

		self._build_style()
		self._build_layout()
		self._bind_shortcuts()
		self.draw_presets()

	def _build_style(self) -> None:
		# Definice vzhledu ttk widgetu.
		style = ttk.Style()
		style.theme_use("clam")

		style.configure("App.TFrame", background=APP_BG)
		style.configure("Panel.TFrame", background=PANEL_BG)
		style.configure("Panel.TLabel", background=PANEL_BG, foreground=TEXT_MAIN, font=("Arial", 10))
		style.configure("Header.TLabel", background=PANEL_BG, foreground=TEXT_MAIN, font=("Arial", 16, "bold"))
		style.configure("Muted.TLabel", background=PANEL_BG, foreground=TEXT_MUTED, font=("Arial", 9))
		style.configure("Action.TButton", font=("Arial", 10, "bold"), padding=(8, 8))

	def _build_layout(self) -> None:
		# Rozlozeni: vlevo panel s ovladanim, vpravo kreslici platno.
		root_frame = ttk.Frame(self.root, style="App.TFrame", padding=14)
		root_frame.pack(fill=tk.BOTH, expand=True)

		controls = ttk.Frame(root_frame, style="Panel.TFrame", padding=16)
		controls.pack(side=tk.LEFT, fill=tk.Y)

		canvas_wrap = ttk.Frame(root_frame, style="App.TFrame", padding=(14, 0, 0, 0))
		canvas_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		self._build_controls(controls)

		self.canvas = tk.Canvas(
			canvas_wrap,
			width=self.canvas_width,
			height=self.canvas_height,
			bg=CANVAS_BG,
			highlightthickness=1,
			highlightbackground="#475569",
		)
		self.canvas.pack(fill=tk.BOTH, expand=True)

	def _build_controls(self, parent: ttk.Frame) -> None:
		# Vstupy, slidery a akce pro tvorbu vrstev terenu.
		ttk.Label(parent, text="Terrain Studio", style="Header.TLabel").pack(anchor="w")
		ttk.Label(parent, text="Midpoint displacement fractal", style="Muted.TLabel").pack(anchor="w", pady=(2, 14))

		self.vars = {
			"start_x": tk.StringVar(value="0"),
			"start_y": tk.StringVar(value="260"),
			"end_x": tk.StringVar(value=str(self.canvas_width)),
			"end_y": tk.StringVar(value="480"),
			"iterations": tk.IntVar(value=9),
			"offset": tk.DoubleVar(value=12.0),
		}

		self._entry_row(parent, "Start X", self.vars["start_x"])
		self._entry_row(parent, "Start Y", self.vars["start_y"])
		self._entry_row(parent, "End X", self.vars["end_x"])
		self._entry_row(parent, "End Y", self.vars["end_y"])

		ttk.Label(parent, text="Iterations", style="Panel.TLabel").pack(anchor="w", pady=(10, 4))
		ttk.Scale(parent, from_=1, to=12, variable=self.vars["iterations"], orient=tk.HORIZONTAL).pack(fill=tk.X)
		ttk.Label(parent, textvariable=self.vars["iterations"], style="Muted.TLabel").pack(anchor="w", pady=(2, 8))

		ttk.Label(parent, text="Offset Strength", style="Panel.TLabel").pack(anchor="w", pady=(8, 4))
		ttk.Scale(parent, from_=1.0, to=80.0, variable=self.vars["offset"], orient=tk.HORIZONTAL).pack(fill=tk.X)
		self.offset_value = ttk.Label(parent, style="Muted.TLabel")
		self.offset_value.pack(anchor="w", pady=(2, 10))
		self._refresh_offset_label()
		self.vars["offset"].trace_add("write", lambda *_: self._refresh_offset_label())

		color_row = ttk.Frame(parent, style="Panel.TFrame")
		color_row.pack(fill=tk.X, pady=(4, 10))
		self.color_preview = tk.Frame(color_row, bg=self.selected_color, width=44, height=24, highlightthickness=1, highlightbackground="#64748B")
		self.color_preview.pack(side=tk.LEFT, padx=(0, 8))
		self.color_preview.pack_propagate(False)
		ttk.Button(color_row, text="Pick Color", style="Action.TButton", command=self.pick_color).pack(side=tk.LEFT, fill=tk.X, expand=True)

		ttk.Button(parent, text="Draw Terrain", style="Action.TButton", command=self.on_draw).pack(fill=tk.X, pady=(6, 4))
		ttk.Button(parent, text="Draw Presets", style="Action.TButton", command=self.draw_presets).pack(fill=tk.X, pady=4)
		ttk.Button(parent, text="Random Layer", style="Action.TButton", command=self.draw_random_layer).pack(fill=tk.X, pady=4)
		ttk.Button(parent, text="Clear Canvas", style="Action.TButton", command=self.clear_canvas).pack(fill=tk.X, pady=4)

		self.status_var = tk.StringVar(value="Ready")
		status_label = ttk.Label(parent, textvariable=self.status_var, style="Muted.TLabel", wraplength=280)
		status_label.pack(anchor="w", pady=(14, 0))

	def _entry_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
		ttk.Label(parent, text=label, style="Panel.TLabel").pack(anchor="w", pady=(4, 2))
		entry = ttk.Entry(parent, textvariable=variable)
		entry.pack(fill=tk.X)

	def _bind_shortcuts(self) -> None:
		# Klavesove zkratky pro rychle kresleni/cisteni.
		self.root.bind("<Control-Return>", lambda _event: self.on_draw())
		self.root.bind("<Escape>", lambda _event: self.clear_canvas())

	def _refresh_offset_label(self) -> None:
		self.offset_value.config(text=f"{self.vars['offset'].get():.1f}")

	def pick_color(self) -> None:
		picked = colorchooser.askcolor(title="Select terrain color", color=self.selected_color)[1]
		if picked:
			self.selected_color = picked
			self.color_preview.config(bg=picked)
			self.status_var.set(f"Selected color: {picked}")

	def clear_canvas(self) -> None:
		self.canvas.delete("all")
		self.status_var.set("Canvas cleared")

	def _config_from_form(self) -> TerrainConfig:
		# Nacteni hodnot z formulare a prevod na TerrainConfig.
		return TerrainConfig(
			start_x=float(self.vars["start_x"].get()),
			start_y=float(self.vars["start_y"].get()),
			end_x=float(self.vars["end_x"].get()),
			end_y=float(self.vars["end_y"].get()),
			iterations=int(self.vars["iterations"].get()),
			offset=float(self.vars["offset"].get()),
			color=self.selected_color,
		)

	def _draw_with_config(self, config: TerrainConfig) -> None:
		# Body prevedeme na polygon a uzavreme jej smerem ke spodku canvasu.
		points = TerrainGenerator.generate_points(config)
		polygon: list[float] = []
		for px, py in points:
			polygon.extend([px, py])

		polygon.extend([config.end_x, self.canvas_height, config.start_x, self.canvas_height])
		self.canvas.create_polygon(polygon, fill=config.color, outline="")

	def on_draw(self) -> None:
		# Vykresleni jedne vrstvy podle aktualniho formulare.
		try:
			config = self._config_from_form()
			self._draw_with_config(config)
			self.status_var.set("Terrain rendered")
		except ValueError:
			self.status_var.set("Invalid input values")
			messagebox.showerror("Input error", "Please enter valid numeric values.")

	def draw_presets(self) -> None:
		# Preddefinovane vrstvy vytvori rychly "hotovy" teren.
		for config in DEFAULT_PRESETS:
			self._draw_with_config(config)
		self.status_var.set("Preset terrain layers rendered")

	def draw_random_layer(self) -> None:
		# Nahodna variace z aktualnich vstupu pro snadne experimentovani.
		try:
			base = self._config_from_form()
		except ValueError:
			self.status_var.set("Invalid input values")
			return

		random_config = TerrainConfig(
			start_x=base.start_x,
			start_y=base.start_y + random.uniform(-120, 120),
			end_x=base.end_x,
			end_y=base.end_y + random.uniform(-120, 120),
			iterations=max(1, base.iterations + random.randint(-2, 2)),
			offset=max(1.0, base.offset + random.uniform(-5, 10)),
			color=self.selected_color,
		)
		self._draw_with_config(random_config)
		self.status_var.set("Random terrain layer rendered")


def run_generation_smoke_test() -> None:
	# Minimalni test: overi ocekavany pocet bodu po iteracich.
	sample = TerrainConfig(0, 260, 920, 480, 8, 12.0, "#16A34A")
	rng = random.Random(123)
	points = TerrainGenerator.generate_points(sample, rng)
	assert len(points) == 2**sample.iterations + 1
	print("Smoke test passed")


def main() -> None:
	parser = argparse.ArgumentParser(description="Terrain Studio")
	parser.add_argument("--smoke-test", action="store_true", help="Run point-generation smoke test and exit")
	args = parser.parse_args()

	if args.smoke_test:
		run_generation_smoke_test()
		return

	root = tk.Tk()
	TerrainApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()

