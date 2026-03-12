import math

import gymnasium as gym
import numpy as np


class QLearningPoleBalancing:

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.x_limit = 2.4                              # limit 2.4
        self.theta_limit = math.radians(15)             # limit 15 stupnu
        self.v_limit = 10
        self.w_limit = 4.0
        self.bins_x = None
        self.bins_theta = None
        self.bins_v = None
        self.bins_w = None
        self.bins_size = 10
        self.initialize_bins()
        self.state = None
        self.Q = np.zeros((self.bins_size, self.bins_size, self.bins_size, self.bins_size, 2))

    def initialize_bins(self):
        self.bins_x = np.linspace(-self.x_limit, self.x_limit, self.bins_size - 1)
        self.bins_theta = np.linspace(-self.theta_limit, self.theta_limit, self.bins_size - 1)
        self.bins_v = np.linspace(-self.v_limit, self.v_limit, self.bins_size - 1)
        self.bins_w = np.linspace(-self.w_limit, self.w_limit, self.bins_size - 1)

    def discretize_state(self, state):
        x, theta, v, w = state
        return (
            int(np.digitize(x, self.bins_x)),
            int(np.digitize(theta, self.bins_theta)),
            int(np.digitize(v, self.bins_v)),
            int(np.digitize(w, self.bins_w))
        )

    def balance(self, gamma=0.99, lr=0.05, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, episodes=2000):
        for episode in range(episodes):
            state, _ = self.env.reset()
            discrete_state = self.discretize_state(state)
            done = False
            total_reward = 0

            while not done:
                if np.random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[discrete_state])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_discrete_state = self.discretize_state(next_state)

                # Index do Q tabulky (x, theta, v, w, akce)
                state_action_idx = discrete_state + (action,)

                # Výpočet odměny a cílové Q hodnoty
                if not done:
                    best_next_action_q = np.max(self.Q[next_discrete_state])
                    target_q = float(reward) + gamma * best_next_action_q
                else:
                    # Pokud spadl, přidáme silnou penalizaci
                    target_q = -100

                    # Aktualizace Q-tabulky
                current_q = self.Q[state_action_idx]
                self.Q[state_action_idx] = current_q + lr * (target_q - current_q)

                discrete_state = next_discrete_state
                total_reward += reward

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if (episode + 1) % 100 == 0:
                print(f"Epizoda: {episode + 1}, Odměna: {total_reward}, Epsilon: {epsilon:.2f}")

    def visualize(self):
        print("Spouštím vizualizaci nejlepšího pokusu...")
        env_render = gym.make("CartPole-v1", render_mode="human")
        state, _ = env_render.reset()
        done = False
        total_reward = 0

        while not done:
            discrete_state = self.discretize_state(state)
            action = np.argmax(self.Q[discrete_state])
            state, reward, terminated, truncated, _ = env_render.step(action)
            done = terminated or truncated
            total_reward += reward
            env_render.render()

        env_render.close()
        print(f"Vizualizace skončila. Agent udržel tyč balancovat po dobu {total_reward} kroků.")

if __name__ == "__main__":
    agent = QLearningPoleBalancing()
    agent.balance(episodes=4000)
    agent.visualize()
