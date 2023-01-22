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

asphalt_color = [142, 142, 142]
chicken_color = [252, 252, 84]
car_speeds = numpy.zeros(10)


def get_chicken_loc(observation):
    for i in range(15, 195):
        if numpy.array_equal(observation[i][48], chicken_color):
            return i
    return 0


def observation_wrapper(observation):
    # pass
    cars_location = get_car_locations(observation)

    for i in range(len(cars_location)):
        # distance = 0
        if i < 5:
            distance = cars_location[i] - 48
            if distance < 0:
                distance += 180
        else:
            distance = 48 - cars_location[i]
            if distance < 0:
                distance += 180
        cars_location[i] = distance
    for i in range(len(cars_location)):
        cars_location[i] /= 160

    chicken_location = get_chicken_loc(observation)
    chicken_lane = (chicken_location - 20) // 16

    # chicken_lanes = get_chicken_lanes(observation, chicken_location)
    # lower_lane = max(chicken_lane - 1, 0)
    # upper_lane = min(chicken_lane + 2, 9)

    # cars_location.insert(0, 180)
    # cars_location.insert(0, 180)
    # cars_location.insert(len(cars_location), 180)
    # cars_location.insert(len(cars_location), 180)
    # cars_location = cars_location[chicken_lane + 2: chicken_lane + 3 + 2]

    # features = numpy.concatenate([cars_location, car_speeds[chicken_lane + 2: chicken_lane + 5], [chicken_location]])
    features = numpy.concatenate([cars_location, [chicken_lane]])
    return features
    # return cars_location


def get_car_locations(observation):
    asphalt = [142, 142, 142]
    chicken = [252, 252, 84]
    cars_location = []
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
    return cars_location


def get_modified_reward(observation, next_observation, reward, action):
    if reward == 1.0:
        return reward * 100
    if action == 1:
        return 10
    if action == 0:
        return 0
    else:
        return -2


def Q(observation, weights, action_space=None, argmax=False):
    features = get_features(observation)

    if argmax:
        max_value = -1
        for _ in action_space:
            value = numpy.dot(weights, features)
            if max_value < value:
                max_value = value
        return max_value

    else:
        return numpy.dot(weights, features)


def get_features(observation):
    features = []
    cars_distances = observation[:-1]
    chicken_lane = observation[-1]
    for i in range(3):
        if 0 <= chicken_lane + i - 1 <= 9:
            features.append(cars_distances[int(chicken_lane) + i - 1])
            features.append(car_speeds[int(chicken_lane) + i - 1])
        else:
            features.append(1)
            features.append(0)
    features.append(chicken_lane / 11)
    return features


def get_best_action(actions, observation, weights):
    max_value = -1
    best_action = -1
    for a in actions:
        next_observation = observation[:len(observation) - 1]
        if a == 0:
            next_observation = numpy.append(next_observation, observation[-1])
        elif a == 1:
            next_observation = numpy.append(next_observation, observation[-1] - 1)
        else:
            next_observation = numpy.append(next_observation, observation[-1] + 1)
        value = Q(next_observation, weights)
        if value > max_value:
            max_value = value
            best_action = a
    return best_action


def calc_speeds(env, observation):
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    next_observation, reward, terminated, truncated, info = env.step(0)
    initial_car_locations = get_car_locations(observation)
    next_car_locations = get_car_locations(next_observation)

    speeds = numpy.divide(numpy.abs(numpy.subtract(initial_car_locations[:5], next_car_locations[:5])), 10)
    speeds = numpy.concatenate([speeds, numpy.flip(speeds)]).tolist()
    max_speed = max(speeds)
    for i in range(len(speeds)):
        speeds[i] /= max_speed
    # speeds.insert(0, 0)
    # speeds.insert(0, 0)
    # speeds.insert(len(speeds), 0)
    # speeds.insert(len(speeds), 0)

    return speeds


def calc_car_lengths(env):
    pass


def main():
    global car_speeds
    env = gymnasium.make("ALE/Freeway-v5", render_mode="human")
    # env = TransformReward(env, reward_wrapper)
    # env = gymnasium.make("ALE/Freeway-v5", mode=0)
    observation, info = env.reset()

    epsilon = 0.3
    lr = 0.1
    discount = 1

    # TODO insert source
    weights = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])

    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    observation, reward, terminated, truncated, info = env.step(0)
    car_speeds = calc_speeds(env, observation)
    # car_lengths = calc_car_lengths(env, observation)

    env = TransformObservation(env, observation_wrapper)

    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)
    observation, reward, terminated, truncated, info = env.step(0)
    # env.step(0)

    for i in range(1000):
        print(f"Ep {i}")
        reward = 0
        epsilon += (1 - epsilon) / 10
        # if i % 20 == 0:
        #     env.reset()
        while reward < 50:

            if random.random() < epsilon:
                action = get_best_action([0, 1, 2], observation, weights)
                # if action != 1:
                #     print(action)
            else:
                action = env.action_space.sample()

            next_observation, reward, terminated, truncated, info = env.step(action)

            reward = get_modified_reward(observation, next_observation, reward, action)

            # Got hit
            if action != 2 and reward < 0:
                for _ in range(12):
                    next_observation, _, terminated, truncated, info = env.step(action)

            diff = reward + discount * Q(next_observation, weights, [0, 1, 2], argmax=True) - Q(observation,
                                                                                                weights, action)
            features = get_features(observation)
            for i in range(len(weights)):
                weights[i] += lr * diff * features[i]

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
