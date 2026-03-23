import math
import tkinter as tk

try:
    from exec06 import LSystem
except ImportError:
    from src.exec06.exec06 import LSystem


class LSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("L-systems drawing by Michael Machu")
        self.root.geometry("1100x720")

        self.system = LSystem(iterations=3)

        self.canvas_width = 850
        self.canvas_height = 700

        self.frame_left = tk.Frame(root, padx=10, pady=10)
        self.frame_left.pack(side="left", fill="both", expand=True)

        self.frame_right = tk.Frame(root, padx=10, pady=10)
        self.frame_right.pack(side="right", fill="y")

        self.canvas = tk.Canvas(
            self.frame_left,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#f8f8f8",
            highlightthickness=1,
            highlightbackground="#d0d0d0",
        )
        self.canvas.pack(fill="both", expand=True)

        self.start_x_var = tk.StringVar(value="200")
        self.start_y_var = tk.StringVar(value="200")
        self.start_angle_deg_var = tk.StringVar(value="0")
        self.start_angle_rad_var = tk.StringVar(value="0")
        self.nesting_var = tk.StringVar(value="3")
        self.line_size_var = tk.StringVar(value="5")

        self.custom_axiom_var = tk.StringVar(value="F+F+F+F")
        self.custom_rule_var = tk.StringVar(value="F -> F+F-F-FF+F+F-F")
        self.custom_angle_deg_var = tk.StringVar(value="90")
        self.custom_angle_rad_var = tk.StringVar(value=str(round(math.pi / 2, 5)))

        self.status_var = tk.StringVar(value="Ready")

        self.presets = [
            {
                "axiom": "F+F+F+F",
                "rules": {"F": "F+F-F-FF+F+F-F"},
                "angle_deg": 90,
            },
            {
                "axiom": "F--F--F",
                "rules": {"F": "F+F--F+F"},
                "angle_deg": 60,
            },
            {
                "axiom": "F",
                "rules": {"F": "F[+F]F[-F]F"},
                "angle_deg": 25,
            },
        ]

        self._build_controls()

    def _build_controls(self):
        self._add_label_entry("Starting X position (int)", self.start_x_var)
        self._add_label_entry("Starting Y position (int)", self.start_y_var)
        self._add_label_entry("Starting angle (degree)", self.start_angle_deg_var)
        self._add_label_entry("Starting angle (radians)", self.start_angle_rad_var)
        self._add_label_entry("The number of nesting (int)", self.nesting_var)
        self._add_label_entry("Size of the line (int)", self.line_size_var)

        tk.Button(self.frame_right, text="Draw first", bg="#a9dfbf", command=lambda: self.draw_preset(0)).pack(
            fill="x", pady=(8, 4)
        )
        tk.Button(self.frame_right, text="Draw second", bg="#a9dfbf", command=lambda: self.draw_preset(1)).pack(
            fill="x", pady=4
        )
        tk.Button(self.frame_right, text="Draw third", bg="#a9dfbf", command=lambda: self.draw_preset(2)).pack(
            fill="x", pady=4
        )

        tk.Label(self.frame_right, text="Custom", pady=8).pack()

        tk.Entry(self.frame_right, textvariable=self.custom_axiom_var).pack(fill="x", pady=4)
        tk.Entry(self.frame_right, textvariable=self.custom_rule_var).pack(fill="x", pady=4)
        tk.Entry(self.frame_right, textvariable=self.custom_angle_deg_var).pack(fill="x", pady=4)
        tk.Entry(self.frame_right, textvariable=self.custom_angle_rad_var).pack(fill="x", pady=4)

        tk.Button(self.frame_right, text="Draw custom", bg="#a9dfbf", command=self.draw_custom).pack(
            fill="x", pady=(8, 4)
        )
        tk.Button(self.frame_right, text="Clear canvas", bg="#f5b7b1", command=self.clear_canvas).pack(
            fill="x", pady=(4, 8)
        )

        tk.Label(self.frame_right, textvariable=self.status_var, wraplength=190, justify="left").pack(
            fill="x", pady=(8, 0)
        )

    def _add_label_entry(self, label_text, variable):
        tk.Label(self.frame_right, text=label_text, anchor="w", justify="left").pack(fill="x", pady=(2, 0))
        tk.Entry(self.frame_right, textvariable=variable).pack(fill="x", pady=(0, 4))

    def _read_float(self, text, fallback):
        try:
            return float(text)
        except (TypeError, ValueError):
            return fallback

    def _read_int(self, text, fallback):
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return fallback

    def _angle_to_radians(self, degree_text, rad_text, default_deg):
        degree_value = self._read_float(degree_text, default_deg)
        rad_value = self._read_float(rad_text, math.radians(degree_value))

        # Keep both fields consistent with the value used for drawing.
        self.start_angle_deg_var.set(str(round(degree_value, 5)))
        self.start_angle_rad_var.set(str(round(rad_value, 5)))

        return rad_value

    def _parse_rules(self, raw_rules):
        text = raw_rules.strip()

        if ";" in text:
            mapping = {}
            for chunk in text.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                separator = "->" if "->" in chunk else "="
                if separator not in chunk:
                    raise ValueError("Use ';' separated rules like 'F->FF;X->X+F'.")
                left, right = chunk.split(separator, 1)
                left = left.strip()
                if not left:
                    raise ValueError("Rule left side cannot be empty.")
                mapping[left] = right.strip()
            if not mapping:
                raise ValueError("No valid rules provided.")
            return mapping

        return self.system._normalize_rules(text)

    def draw_preset(self, index):
        preset = self.presets[index]
        self.custom_axiom_var.set(preset["axiom"])
        if len(preset["rules"]) == 1:
            key, value = next(iter(preset["rules"].items()))
            self.custom_rule_var.set(f"{key} -> {value}")
        else:
            self.custom_rule_var.set("; ".join(f"{k} -> {v}" for k, v in preset["rules"].items()))

        angle_deg = preset["angle_deg"]
        self.custom_angle_deg_var.set(str(angle_deg))
        self.custom_angle_rad_var.set(str(round(math.radians(angle_deg), 5)))

        self._draw_lsystem(preset["axiom"], preset["rules"], math.radians(angle_deg))

    def draw_custom(self):
        try:
            axiom = self.custom_axiom_var.get().strip()
            if not axiom:
                raise ValueError("Axiom cannot be empty.")

            rules = self._parse_rules(self.custom_rule_var.get())

            angle_deg = self._read_float(self.custom_angle_deg_var.get(), 90.0)
            angle_rad = self._read_float(self.custom_angle_rad_var.get(), math.radians(angle_deg))

            self.custom_angle_deg_var.set(str(round(angle_deg, 5)))
            self.custom_angle_rad_var.set(str(round(angle_rad, 5)))

            self._draw_lsystem(axiom, rules, angle_rad)
        except Exception as exc:
            self.status_var.set(f"Input error: {exc}")

    def _draw_lsystem(self, axiom, rules, turn_angle_rad):
        self.clear_canvas(update_status=False)

        start_x = self._read_float(self.start_x_var.get(), 200.0)
        start_y = self._read_float(self.start_y_var.get(), 200.0)
        base_deg = self._read_float(self.start_angle_deg_var.get(), 0.0)
        current_angle = self._angle_to_radians(self.start_angle_deg_var.get(), self.start_angle_rad_var.get(), base_deg)

        iterations = max(0, self._read_int(self.nesting_var.get(), 3))
        line_size = max(1.0, self._read_float(self.line_size_var.get(), 5.0))

        self.system.iterations = iterations
        commands = self.system.generate(rules, axiom)

        x = start_x
        y = start_y
        stack = []

        for symbol in commands:
            if symbol in {"F", "G", "I", "D"}:
                nx = x + line_size * math.cos(current_angle)
                ny = y - line_size * math.sin(current_angle)
                self.canvas.create_line(x, y, nx, ny, fill="#666666", width=1)
                x, y = nx, ny
            elif symbol in {"f", "g"}:
                x += line_size * math.cos(current_angle)
                y -= line_size * math.sin(current_angle)
            elif symbol == "+":
                current_angle += turn_angle_rad
            elif symbol == "-":
                current_angle -= turn_angle_rad
            elif symbol == "[":
                stack.append((x, y, current_angle))
            elif symbol == "]" and stack:
                x, y, current_angle = stack.pop()

        self.status_var.set(
            f"Drawn: {len(commands)} commands, nesting={iterations}, line={line_size}, angle={round(turn_angle_rad, 5)} rad"
        )

    def clear_canvas(self, update_status=True):
        self.canvas.delete("all")
        if update_status:
            self.status_var.set("Canvas cleared")


if __name__ == "__main__":
    root = tk.Tk()
    app = LSystemApp(root)
    root.mainloop()

