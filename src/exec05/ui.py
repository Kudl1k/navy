import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # random action
        observation, reward, terminated, truncated, info = env.step(action)

        print(env.observation_space)
        print(env.action_space)

        # U

    env.close()
