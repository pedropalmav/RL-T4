## actor critico con aproximación lineal
import numpy as np

from FeatureExtractor import FeatureExtractor


class ActorCritic:

    def __init__(self, gamma: float, alpha_v: float, alpha_pi: float):
        self.__alpha_v = alpha_v
        self.__alpha_pi = alpha_pi
        self.__gamma = gamma
        #########
        self.__feature_extractor = FeatureExtractor()
        self.__num_features = self.__feature_extractor.num_of_features
        #########
        self.__weights_v = np.zeros(self.__num_features)
        self.__theta_mu = np.zeros(self.__num_features)
        self.__theta_sigma = np.zeros(self.__num_features)
        self.__I = None

    def reset_episode_values(self):
        self.__I = 1.0

    def sample_action(self, observation):
        x = self.__feature_extractor.get_features(observation)
        mu = np.dot(self.__theta_mu, x)
        sigma = np.exp(np.dot(self.__theta_sigma, x))
        return np.random.normal(loc=mu, scale=sigma, size=None)
    
    def dot(self, x,y):
        return np.dot(x, self.__feature_extractor.get_features(y))

    def learn(self, observation, action, reward, next_observation, done):
        # TODO: Implementar la regla de aprendizaje de actor-critic
        if done:
            delta = reward - self.dot(self.__weights_v,observation)
        else:
            delta = reward + self.__gamma * self.dot(self.__weights_v, next_observation) - self.dot(self.__weights_v, observation)
        self.__weights_v += self.__alpha_v * delta * self.__feature_extractor.get_features(observation)
        mu = self.dot(self.__theta_mu, observation)
        sigma = np.exp(self.dot(self.__theta_sigma, observation))
        
        grad_theta_mu = (1/(sigma**2)) * (action - mu) * self.__feature_extractor.get_features(observation)
        grad_theta_sigma = ((((action-mu)**2)/(sigma**2))-1) * self.__feature_extractor.get_features(observation)
        self.__theta_mu += self.__alpha_pi * self.__I * delta  * grad_theta_mu
        self.__theta_sigma += self.__alpha_pi * self.__I * delta *  grad_theta_sigma
        self.__I *= self.__gamma