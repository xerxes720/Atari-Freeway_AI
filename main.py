import math
from copy import deepcopy

import gymnasium
import numpy
import random

from gymnasium.wrappers import TransformObservation

asphalt_color = [142, 142, 142]
chicken_color = [252, 252, 84]
car_speeds = numpy.zeros(10)
weights = numpy.array([1.0, 1.0])


def get_chicken_loc(observation):
    for i in range(15, 195):
        if numpy.array_equal(observation[i][48], chicken_color):
            return i
    return 0


def observation_wrapper(observation):
    cars_location = get_car_locations(observation)
    chicken_location = get_chicken_loc(observation)
    return numpy.concatenate([cars_location, [chicken_location]])


def get_car_locations(observation):
    asphalt = [142, 142, 142]
    chicken = [252, 252, 84]
    cars_location = []
    for i in range(32, 100, 16):
        # In case the car is out of picture
        in_frame = False
        for j in range(8, 160):
            if not numpy.array_equal(observation[i][j], chicken) and not numpy.array_equal(observation[i][j], asphalt):
                cars_location.append(j)
                in_frame = True
                break
        if not in_frame:
            cars_location.append(160)

    for i in range(112, 180, 16):
        # In case the car is out of picture
        in_frame = False
        for j in range(159, 7, -1):
            if not numpy.array_equal(observation[i][j], chicken) and not numpy.array_equal(observation[i][j], asphalt):
                cars_location.append(j)
                in_frame = True
                break
        if not in_frame:
            cars_location.append(0)
    return cars_location


def get_modified_reward(observation, next_observation, reward, action):
    features = get_features(observation)
    next_features = get_features(next_observation)
    if reward == 1.0:
        return reward
    if next_features[-1] > features[-1]:
        return 0.1
    if next_features[-1] == features[-1]:
        return -0.05
    else:
        return -0.1


def Q(observation, weights, action_space=None, argmax=False):
    features = get_features(observation)

    if argmax:
        max_value = -1.0
        for _ in action_space:
            value = numpy.dot(weights, features)
            if max_value < value:
                max_value = value
        return max_value

    else:
        return numpy.dot(weights, features)


def get_chicken_lane(chicken_location):
    if chicken_location > 100:
        return (chicken_location - 21) // 16
    else:
        return (chicken_location - 21) // 16


def get_features(observation):
    features = []
    cars_distances = deepcopy(observation[:-1])

    for i in range(len(cars_distances)):
        # distance = 0
        if i < 5:
            distance = cars_distances[i] - 49
            if distance < -14:
                distance += 180
            elif -14 <= distance < 0:
                distance = 0
        else:
            distance = 44 - cars_distances[i]
            if distance < -14:
                distance += 180
            elif -14 <= distance < 0:
                distance = 0
        cars_distances[i] = distance

    chicken_location = observation[-1]
    chicken_lane = get_chicken_lane(chicken_location)
    if 0 <= chicken_lane <= 9:
        dist_c = cars_distances[int(chicken_lane)]
        if dist_c == 0:
            dist_c = 0.9
        features.append(1 / (dist_c))
    else:
        features.append(0)
    features.append((186 - chicken_location) / 186)
    return features


def get_best_action(actions, observation, weights):
    max_value = -math.inf
    best_action = -1
    next_observation = deepcopy(observation[:-1])
    for i in range(len(next_observation)):
        next_observation[i] += car_speeds[i]

    for a in actions:

        if a == 0:
            next_observation = numpy.append(next_observation, observation[-1])
        elif a == 1:
            next_observation = numpy.append(next_observation, observation[-1] - 10)
        else:
            next_observation = numpy.append(next_observation, observation[-1] + 10)
        value = Q(next_observation, weights)
        next_observation = next_observation[:-1]
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
    speeds = numpy.concatenate([numpy.negative(speeds), numpy.flip(speeds)]).tolist()

    return speeds


def calc_car_lengths(env):
    pass


def main():
    global car_speeds
    env = gymnasium.make("ALE/Freeway-v5", render_mode="human")
    observation, info = env.reset()

    epsilon = 0.5
    lr = 0.3
    discount = 1

    # TODO insert source

    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    observation, reward, terminated, truncated, info = env.step(0)
    car_speeds = calc_speeds(env, observation)

    env = TransformObservation(env, observation_wrapper)

    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)
    observation, reward, terminated, truncated, info = env.step(0)

    for i in range(1000):
        print(f"Ep {i}")
        reward = 0
        epsilon += (1 - epsilon) / 10
        lr -= lr * 0.1
        while reward != 1.0:
            if random.random() < epsilon:
                action = get_best_action([0, 1, 2], observation, weights)
            else:
                action = env.action_space.sample()

            next_observation, reward, terminated, truncated, info = env.step(action)

            reward = get_modified_reward(observation, next_observation, reward, action)
            if reward == 1.0:
                break

            hit_loc = observation[-1]
            # Got hit
            while next_observation[-1] > hit_loc and action != 2:
                hit_loc = next_observation[-1]
                next_observation, _, terminated, truncated, info = env.step(action)
                reward -= 0.2

            diff = reward + discount * Q(next_observation, weights, [0, 1, 2], argmax=True) - Q(observation,
                                                                                                weights, action)
            features = get_features(observation)
            for i in range(len(weights)):
                weights[i] += lr * diff * features[i]

            observation = next_observation
            if terminated or truncated:
                print("Game finished")
                break

        print("Epsilon = " + str(epsilon))
        print("///////////////////////////////////")

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


main()
