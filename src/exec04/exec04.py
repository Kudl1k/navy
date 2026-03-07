import random

import numpy as np


class QLearningMouseAndCheese:

    EMPTY_SPACE = 0
    HOLE = -1
    CHEESE = 1


    def __init__(self):
        self.grid = None
        self.grid_size = None
        self.Q = None
        self.state = None
        self.set_grid_size(5)

    def set_grid_size(self, size):
        self.grid_size = size                                       # Velikost gridu
        self.grid = np.zeros((self.grid_size, self.grid_size))      # Inicializace gridu
        self.Q = np.zeros((self.grid_size, self.grid_size, 4))      # Inicializace Q-tabulky - vyplnit s nuly + 4 akcemi
        self.state = (0,0)

    def place_mouse(self, x, y):
        self.state = (x,y)

    def toggle_grid_item(self, x, y, item):
        if self.grid[x,y] == item:                                  # Pokud je na pozici již umístěn daný prvek, tak ho odstraníme
            self.grid[x,y] = self.EMPTY_SPACE
        else:
            self.grid[x,y] = item                                    # Jinak ho umístíme

    def get_policy(self):                                           # ziskani nejlepsi cesty
        return np.argmax(self.Q, axis=2)

    def best_path(self, max_steps=200):
        state = self.state
        path = [state]

        for _ in range(max_steps):
            x,y = state
            action = np.argmax(self.Q[x,y])

            new_state = self.move(state, action)
            path.append(new_state)

            state = new_state

            if self.grid[state] in (self.HOLE, self.CHEESE):
                break

        return path

    def move(self, state, action):
        x,y = state

        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size-1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size-1:
            y += 1

        return x,y

    def reward(self, state):
        x,y = state

        if self.grid[x,y] == self.CHEESE:                           # 100 za syr
            return 100
        elif self.grid[x,y] == self.HOLE:                           # -100 za trapu
            return -100

        return -1                                                   # -1 za normalni krok

    def find(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995,
             epsilon_min=0.01, max_steps=200, episodes=1000):
        for _ in range(episodes):
            episode_state = self.state
            for _ in range(max_steps):
                x,y = episode_state                                  # aktualni pozice

                if random.uniform(0,1) < epsilon:               # vygenerovani akce podle epsilonu
                    action = random.randint(0,3)
                else:
                    action = np.argmax(self.Q[x,y])                 # vyber akce podle Q-tabulky

                new_state = self.move(episode_state, action)           # posunuti do noveho stavu

                r = self.reward(new_state)                          # ziskani odmeny za posun

                nx, ny = new_state

                self.Q[x,y,action] += alpha * (r + gamma * np.max(self.Q[nx,ny]) - self.Q[x,y,action])  # vypocteni aktualniho Q-hodnoty

                episode_state = new_state                              # aktualizace stavu

                if self.grid[nx,ny] in (self.HOLE, self.CHEESE): # pokud se dostaneme do trapy nebo syr, tak konec
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)     # postupne snizovani epsilonu pro mene nahodne akce
