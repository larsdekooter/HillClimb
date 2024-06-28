import gymnasium as gym
from network import Network

env = gym.make("MountainCar-v0", render_mode="human")
observation, info = env.reset()

network = Network()

# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(observation)
#     if terminated or truncated:
#         observation, info = env.reset()
ngames = 0
while True:
    move = network.getMove(observation)
    oldstate = observation
    observation, reward, terminated, truncated, info = env.step(move)
    network.remember(oldstate, move, reward, observation, terminated)
    network.trainShort(oldstate, move, reward, observation, terminated)

    if terminated or truncated:
        ngames += 1
        observation, info = env.reset()
        network.trainLong()
        net = network.net
        rand = network.rand
        network.net = 0
        network.rand = 0

        try:
            percentage = round((net / (net + rand)) * 100.0, 2)
        except:
            percentage = 0

        print("Game", ngames, "%", percentage)

env.close()
