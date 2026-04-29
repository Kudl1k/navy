# PRI0192
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class DoublePendulum:
    def __init__(self,
             mass_1: float = 1,
             mass_2: float = 1,
             length_1: float = 1,
             length_2: float = 1,
             theta_1 = 2 * np.pi / 6,
             theta_2 = 5 * np.pi / 8
        ):
        self.g = 9.81
        self.l1 = length_1
        self.l2 = length_2
        self.m1 = mass_1
        self.m2 = mass_2

        self.theta_1 = theta_1
        self.theta_2 = theta_2
        # Velocity
        self.vel_theta_1 = 0
        self.vel_theta_2 = 0

        # Initial state
        self.y0 = np.array([self.theta_1, self.vel_theta_1, self.theta_2, self.vel_theta_2])

    # Calculates the second derivations for odeint
    def __get_derivative(self, y, _, l1, l2, m1, m2):
        theta_1, vel_theta_1, theta_2, vel_theta_2 = y
        delta = theta_1 - theta_2

        # Helper variables
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        sin_theta_1 = np.sin(theta_1)
        sin_theta_2 = np.sin(theta_2)
        denominator_common = m1 + m2 * sin_delta ** 2

        # Calculate the acceleration of the pendulums
        accel_theta_1 = (
            m2 * self.g * sin_theta_2 * cos_delta - m2 * sin_delta *
            (l1 * vel_theta_1 ** 2 * cos_delta + l2 * vel_theta_2 ** 2) -
            (m1 + m2) * self.g * sin_theta_1
        ) / (l1 * denominator_common)

        accel_theta_2 = (
            (m1 + m2) * (l1 * vel_theta_1 ** 2 * sin_delta - self.g * sin_theta_2 + self.g * sin_theta_1 * cos_delta)
            + m2 * l2 * vel_theta_2 ** 2 * sin_delta * cos_delta
        ) / (l2 * denominator_common)

        return vel_theta_1, accel_theta_1, vel_theta_2, accel_theta_2

    # Calculates the cartesian coordinates of the two pendulums
    def __calculate_cartesian_coordinates(self, theta_1_vals, theta_2_vals):
        x1 = self.l1 * np.sin(theta_1_vals)
        y1 = -self.l1 * np.cos(theta_1_vals)
        x2 = x1 + self.l2 * np.sin(theta_2_vals)
        y2 = y1 - self.l2 * np.cos(theta_2_vals)
        return x1, y1, x2, y2

    # Animation stuff
    def run(self, fps=30, duration=30, trail_length=100):
        t = np.linspace(0, duration, duration * fps)
        # Integrate the ODEs to get the angles over time
        solution = odeint(self.__get_derivative, self.y0, t, args=(self.l1, self.l2, self.m1, self.m2))
        theta_1_vals = solution[:, 0]
        theta_2_vals = solution[:, 2]

        x1_vals, y1_vals, x2_vals, y2_vals = self.__calculate_cartesian_coordinates(theta_1_vals, theta_2_vals)

        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'o-', lw=2)
        trail, = ax.plot([], [], 'r-', alpha=0.6)

        ax.set_xlim(- (self.l1 + self.l2) * 1.1, (self.l1 + self.l2) * 1.1)
        ax.set_ylim(- (self.l1 + self.l2) * 1.1, (self.l1 + self.l2) * 1.1)
        ax.set_aspect('equal')
        ax.grid()

        trail_x, trail_y = [], []

        def init():
            line.set_data([], [])
            trail.set_data([], [])
            return line, trail

        def update(i):
            this_x = [0, x1_vals[i], x2_vals[i]]
            this_y = [0, y1_vals[i], y2_vals[i]]
            line.set_data(this_x, this_y)

            trail_x.append(x2_vals[i])
            trail_y.append(y2_vals[i])

            if len(trail_x) > trail_length:
                trail_x.pop(0)
                trail_y.pop(0)

            trail.set_data(trail_x, trail_y)
            return line, trail

        animation = FuncAnimation(
            fig, update, frames=len(t), init_func=init,
            blit=True, interval=1000 / fps
        )

        # Save animation
        animation.save("exec11.mp4", writer="ffmpeg", fps=fps)

        plt.show()

if __name__ == '__main__':
    pendulum = DoublePendulum(
        mass_1=1,
        mass_2=1,
        length_1=1,
        length_2=1
    )
    pendulum.run(
        fps=60,
        duration=30,
        trail_length=200
    )