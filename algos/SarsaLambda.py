import random

import numpy as np

from FeatureExtractor import FeatureExtractor


class SarsaLambda:

    def __init__(self, num_actions: int, epsilon: float, alpha: float, gamma: float, s_lambda: float):
        self.__num_actions = num_actions
        self.__epsilon = epsilon
        self.__alpha = alpha
        self.__gamma = gamma
        self.__s_lambda = s_lambda
        self.__feature_extractor = FeatureExtractor()
        self.__num_features = self.__feature_extractor.num_of_features
        self.__weights = np.zeros(self.__num_features)
        self.__z = np.zeros(self.__num_features)

    def sample_action(self, observation):
        if random.random() < self.__epsilon:
            return random.randrange(self.__num_actions)
        return self.argmax(observation)

    def argmax(self, observation):
        a_max = None
        q_max = float('-inf')
        for action in range(self.__num_actions):
            q_value = self.__get_q_estimate(observation, action)
            if q_value > q_max:
                q_max = q_value
                a_max = [action]
            elif q_value == q_max:
                a_max.append(action)
        return random.choice(a_max)

    def __get_q_estimate(self, observation, action):
        x = self.__feature_extractor.get_features(observation, action)
        return np.dot(self.__weights, x)

    def learn(self, observation, action, reward, next_observation, next_action, done):
        x = self.__feature_extractor.get_features(observation, action)
        self.__z = self.__gamma * self.__s_lambda * self.__z + x
        self.__z = np.clip(self.__z, 0.0, 1.0)
        delta = reward + self.__gamma * self.__get_q_estimate(next_observation, next_action) - self.__get_q_estimate(
            observation, action)
        self.__weights += self.__alpha * delta * self.__z