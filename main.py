# This is a sample Python script.
import collections
import random

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import gymnasium
import numpy
import numpy as np
import random

from gymnasium.wrappers import TransformObservation


# def reward_wrapper(reward):
#     global action
#     if reward == 1.0:
#         return reward * 10
#     elif action == 1
#     return reward

def get_chicken_loc(observation, chicken):
    for i in range(15, 195):
        if numpy.array_equal(observation[i][48], chicken):
            return i
    return 0


def observation_wrapper(observation):
    # pass
    asphalt = [142, 142, 142]
    chicken = [252, 252, 84]
    cars_location = []
    chicken_location = -1
    for i in range(32, 180, 16):  # TODO
        # In case the car is out of picture
        in_frame = False
        for j in range(8, 160):
            if not numpy.array_equal(observation[i][j], chicken) and not numpy.array_equal(observation[i][j], asphalt):
                cars_location.append(j)
                in_frame = True
                break
        if not in_frame:
            if i < 100:
                cars_location.append(0)
            else:
                cars_location.append(160)

    for i in range(len(cars_location)):
        cars_location[i] /= 160

    chicken_location = get_chicken_loc(observation, chicken)
    cars_location.append(chicken_location / 195)

    return cars_location


def get_modified_reward(observation, next_observation, reward):
    if reward == 1.0:
        return reward * 100
    if next_observation[-1] < observation[-1]:
        return 10
    if next_observation[-1] == observation[-1]:
        return 0
    else:
        return -10


def Q(observation, weights, action_space=None, argmax=False):
    if argmax:
        max_value = -1
        for _ in action_space:
            value = numpy.dot(weights, observation)
            if max_value < value:
                max_value = value
        return max_value

    else:
        return numpy.dot(weights, observation)


def get_best_action(actions, observation, weights):
    max_value = -1
    best_action = -1
    for a in actions:
        next_observation = observation[:len(observation) - 1]
        if a == 0:
            next_observation.append(observation[-1])
        elif a == 1:
            next_observation.append(observation[-1] + 1)
        else:
            next_observation.append(observation[-1] - 1)
        value = Q(next_observation, weights)
        if value > max_value:
            max_value = value
            best_action = a
    return best_action


def main():
    # global action
    env = gymnasium.make("ALE/Freeway-v5",render_mode="human", mode=0)
    # env = TransformReward(env, reward_wrapper)
    env = TransformObservation(env, observation_wrapper)
    # env = gymnasium.make("ALE/Freeway-v5", mode=0)
    observation, info = env.reset()

    epsilon = 0.7
    lr = 0.1
    discount = 1

    # TODO insert source
    # Q = collections.defaultdict(lambda: np.zeros(env.action_space.n))
    weights = numpy.ones(11)
    # Q[str(observation)][2] = 1
    # print(int(numpy.argmax(Q[str(observation)])))

    for i in range(1000):
        print(f"Ep {i}")
        reward = 0
        epsilon += (1 - epsilon) / 1000
        # if i % 20 == 0:
        #     env.reset()
        while reward < 50:

            if random.random() < epsilon:
                action = get_best_action([0, 1, 2], observation, weights)
                if action != 1:
                    print(action)
            else:
                action = env.action_space.sample()

            next_observation, reward, terminated, truncated, info = env.step(action)

            reward = get_modified_reward(observation, next_observation, reward)

            # Got hit
            if action != 2 and reward < 0:
                for _ in range(12):
                    next_observation, _, terminated, truncated, info = env.step(action)

            diff = reward + discount * Q(next_observation, weights, [0, 1, 2], argmax=True) - Q(observation,
                                                                                                weights, action)
            for i in range(len(weights)):
                weights[i] += lr * diff * observation[i]

            observation = next_observation
            if terminated or truncated:
                print("Game finished")
                break

        # print(Q[str(observation)][action])

        # print(observation)
        print("Epsilon = " + str(epsilon))
        print("///////////////////////////////////")

        # print(f"reward = {reward} for action {action} ")
        if terminated or truncated:
            observation, info = env.reset()
            # terminated = False
            # truncated = False
    env.close()


main()
