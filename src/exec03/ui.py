import tkinter as tk
import numpy as np
import time
from exec03 import HopfieldNetwork


# Sem vlož svou třídu HopfieldNetwork, kterou jsme napsali výše

class HopfieldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hopfield Network")

        # Nastavení mřížky
        self.grid_size = 10
        self.cell_size = 50

        # Reprezentace aktuálního stavu mřížky (na začátku vše bílé, tedy -1)
        self.grid_state = np.full((self.grid_size, self.grid_size), -1)

        # Instance tvé sítě a paměť na uložené vzory
        self.network = HopfieldNetwork()
        self.saved_patterns = []
        self.load_patterns_from_file()

        # -- ROZVRŽENÍ OKNA --
        self.frame_left = tk.Frame(root, padx=10, pady=10)
        self.frame_left.pack(side=tk.LEFT)

        self.frame_right = tk.Frame(root, padx=10, pady=10)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.Y)

        # -- KRESLÍCÍ PLÁTNO --
        self.canvas = tk.Canvas(
            self.frame_left,
            width=self.grid_size * self.cell_size,
            height=self.grid_size * self.cell_size,
            bg="white"
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.draw_grid()

        # -- TLAČÍTKA --
        # Využíváme barvy ze zadání
        tk.Button(self.frame_right, text="Save pattern", bg="#a9dfbf", command=self.save_pattern).pack(fill=tk.X,
                                                                                                       pady=5)
        tk.Button(self.frame_right, text="Repair pattern Sync", bg="#f9e79f", command=self.repair_sync).pack(fill=tk.X,
                                                                                                             pady=5)
        tk.Button(self.frame_right, text="Repair pattern Async", bg="#f9e79f", command=self.repair_async).pack(
            fill=tk.X, pady=5)
        tk.Button(self.frame_right, text="Show saved patterns", bg="#AED6F1", command=self.show_saved_patterns).pack(
            fill=tk.X, pady=5)
        tk.Button(self.frame_right, text="Clear grid", bg="#F5B7B1", command=self.clear_grid).pack(fill=tk.X, pady=5)

        tk.Label(self.frame_right, text="\nMax recommended amount\nof saved patterns is 5", justify=tk.CENTER).pack(
            pady=10)

    # -- METODY PRO VYKRESLOVÁNÍ --
    def draw_grid(self):
        self.canvas.delete("all")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                # Zjištění barvy podle matice (-1 = bílá, 1 = černá)
                color = "black" if self.grid_state[row, col] == 1 else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

    def on_canvas_click(self, event):
        # Zjištění souřadnic kliknutí
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            # Přepnutí barvy (z 1 na -1 nebo naopak)
            self.grid_state[row, col] *= -1
            self.draw_grid()

    def clear_grid(self):
        # Reset matice na -1 a překreslení
        self.grid_state = np.full((self.grid_size, self.grid_size), -1)
        self.draw_grid()

    # -- PROPOJENÍ S TVOU HOPFIELDOVOU SÍTÍ --
    def save_pattern(self):
        if len(self.saved_patterns) >= 5:
            print("Dosažen maximální počet 5 vzorů!")

        # Převedeme 2D matici na 1D vektor a uložíme
        vector = self.network.prepare_vector(self.grid_state)
        self.saved_patterns.append(vector)

        # Provedeme učící fázi nad všemi uloženými vzory
        self.network.phase_learning(self.saved_patterns)
        print(f"Vzor uložen. Celkem uložených vzorů: {len(self.saved_patterns)}")
        self.clear_grid()

    def repair_sync(self):
        if self.network.W is None:
            print("Nejprve ulož nějaký vzor!")
            return

        vector = self.network.prepare_vector(self.grid_state)
        recovered_vector = self.network.sync_recovery(vector)

        # Převod 1D vektoru zpět na 2D matici a překreslení
        self.grid_state = recovered_vector.reshape((self.grid_size, self.grid_size))
        self.draw_grid()

    def animate_step(self, current_vector):
        """Tato funkce se zavolá pokaždé, když síť opraví jeden pixel."""
        self.grid_state = current_vector.reshape((self.grid_size, self.grid_size))
        self.draw_grid()
        self.root.update()  # Donutí Tkinter OKAMŽITĚ vykreslit změnu
        time.sleep(0.05)  # Zpomalí program o 50 milisekund (můžeš si upravit)

    def repair_async(self):
        if self.network.W is None:
            print("Nejprve ulož nějaký vzor!")
            return

        vector = self.network.prepare_vector(self.grid_state)

        # ZDE JE ZMĚNA: Předáváme animate_step jako parametr callback
        recovered_vector = self.network.async_recovery(vector, callback=self.animate_step)

        self.grid_state = recovered_vector.reshape((self.grid_size, self.grid_size))
        self.draw_grid()

    def show_saved_patterns(self):
        if not self.saved_patterns:
            print("Zatím nejsou uloženy žádné vzory.")
            return

        # Projdeme všechny uložené vzory a pro každý vytvoříme nové okno
        for i, vector in enumerate(self.saved_patterns):
            self.create_pattern_window(i, vector)

    def create_pattern_window(self, index, vector):
        # Vytvoření nového vyskakovacího okna
        window = tk.Toplevel(self.root)
        window.title(f"{index + 1}. matrix")  # Název okna podle zadání

        frame_left = tk.Frame(window, padx=10, pady=10)
        frame_left.pack(side=tk.LEFT)

        frame_right = tk.Frame(window, padx=10, pady=10)
        frame_right.pack(side=tk.RIGHT, fill=tk.Y)

        # Kreslící plátno (pouze pro zobrazení, neklikatelné)
        canvas = tk.Canvas(frame_left, width=self.grid_size * self.cell_size, height=self.grid_size * self.cell_size,
                           bg="white")
        canvas.pack()

        # Převedení 1D vektoru zpět na 2D matici a vykreslení
        matrix = vector.reshape((self.grid_size, self.grid_size))
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = "black" if matrix[row, col] == 1 else "white"
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

        # -- Tlačítka pro zobrazení metrik --
        # "Show matrix" -> Vytiskne matici s 1 a -1
        tk.Button(frame_right, text="Show matrix", bg="#AED6F1",
                  command=lambda: print(f"\n--- Matrix {index + 1} ---\n{matrix}")).pack(fill=tk.X, pady=5)

        # "Show matrix without zeros" -> Síť má -1, ale originál zadání měl 0. Převedeme -1 zpět na 0 pro tisk.
        matrix_zeros = np.where(matrix == -1, 0, 1)
        tk.Button(frame_right, text="Show matrix without zeros", bg="#AED6F1",
                  command=lambda: print(f"\n--- Matrix (1/0) {index + 1} ---\n{matrix_zeros}")).pack(fill=tk.X, pady=5)

        # "Show vector" -> Vytiskne zploštělý 1D vektor
        tk.Button(frame_right, text="Show vector", bg="#AED6F1",
                  command=lambda: print(f"\n--- Vector {index + 1} ---\n{vector}")).pack(fill=tk.X, pady=5)

        # "Show weighted matrix" -> Spočítá a vytiskne váhovou matici jen pro TENTO jeden vzor
        w_matrix = np.outer(vector, vector)
        np.fill_diagonal(w_matrix, 0)
        tk.Button(frame_right, text="Show weighted matrix", bg="#AED6F1",
                  command=lambda: print(f"\n--- Weighted matrix {index + 1} ---\n{w_matrix}")).pack(fill=tk.X, pady=5)

        # "Show full pattern" -> Tohle tlačítko je v zadání, vizuálně odpovídá otevřenému oknu
        tk.Button(frame_right, text="Show full pattern", bg="#AED6F1",
                  command=lambda: print(f"Zobrazeno v okně.")).pack(fill=tk.X, pady=5)

        # "Forget the pattern" -> Červené tlačítko pro smazání
        tk.Button(frame_right, text="Forget the pattern", bg="#F5B7B1",
                  command=lambda: self.forget_pattern(index, window)).pack(fill=tk.X, pady=5)

    def forget_pattern(self, index, window):
        # 1. Odstraníme vzor ze seznamu uložených vzorů
        self.saved_patterns.pop(index)

        # 2. Musíme přeučit síť (vytvořit novou matici W bez smazaného vzoru)
        if self.saved_patterns:
            self.network.phase_learning(self.saved_patterns)
        else:
            self.network.W = None  # Pokud už nezbyl žádný vzor

        # 3. Zavřeme vyskakovací okno
        window.destroy()
        print(f"\nVzor byl zapomenut. Zbylo vzorů: {len(self.saved_patterns)}")

    def load_patterns_from_file(self, filename="patterns.txt"):
        import os
        if not os.path.exists(filename):
            print(f"Soubor {filename} nebyl nalezen. Začínáme s prázdnou pamětí.")
            return

        with open(filename, 'r') as f:
            # Rozdělí text na bloky podle prázdných řádků
            content = f.read().strip().split('\n\n')

        for block in content:
            # Převod textového bloku na 2D numpy matici
            lines = block.strip().split('\n')
            matrix = []
            for line in lines:
                # Převede znaky '0' a '1' na čísla
                row = [int(char) for char in line.strip()]
                matrix.append(row)

            matrix = np.array(matrix)

            # Převedeme do formátu pro síť (z 0 a 1 na -1 a 1) a zploštíme
            vector = self.network.prepare_vector(matrix)
            self.saved_patterns.append(vector)

        # Pokud se načetly nějaké vzory, rovnou síť naučíme
        if self.saved_patterns:
            self.network.phase_learning(self.saved_patterns)
            print(f"Úspěšně načteno a naučeno {len(self.saved_patterns)} vzorů ze souboru {filename}.")


# Spuštění aplikace
if __name__ == "__main__":
    root = tk.Tk()
    app = HopfieldApp(root)
    root.mainloop()