from algos.ActorCritic import ActorCritic
from algos.QLearning import QLearning
from algos.Sarsa import Sarsa
from algos.SarsaLambda import SarsaLambda
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

def run_actor_critic():
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env, filename=None)
    actor_critic = ActorCritic(gamma=1.0, alpha_v=0.001, alpha_pi=0.0001)
    for episode in range(1000):
        observation, info = env.reset()
        actor_critic.reset_episode_values()
        terminated = truncated = False
        while not terminated and not truncated:
            action = actor_critic.sample_action(observation)
            next_observation, reward, terminated, truncated, info = env.step([action])
            actor_critic.learn(observation, action, reward, next_observation, terminated)
            observation = next_observation
    ep_len = env.get_episode_lengths()
    env.close()
    return ep_len

def run_SAC():
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env, filename=None)
    model = SAC("MlpPolicy", env, verbose=0, gamma=1.0, use_sde=True, train_freq=32)
    model.learn(total_timesteps=300000)
    ep_len = env.get_episode_lengths()
    env.close()
    return ep_len
    
def init_parameters_SAC(selection:str):
    default_kwargs = {
        "policy":"MlpPolicy",
        "learning_rate":0.0003,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":256,
        "tau":0.005,
        "gamma":0.99,
        "train_freq":1,
        "gradient_steps":1,
        "ent_coef":"auto",
        "target_update_interval":1,
        "use_sde":False,
        "verbose":0}

    original_kwargs = {
        "policy": "MlpPolicy",
        "learning_rate":0.0003,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":256,
        "tau":0.005,
        "gamma":1.0,##
        "train_freq":32,##
        "gradient_steps":1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "use_sde": True,##
        "verbose": 0}

    kwargs_1 = {"policy": "MlpPolicy", #subiendo el learning_rate, sin sde
        "learning_rate":0.0007,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":256,
        "tau":0.005,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    kwargs_2 = {"policy": "MlpPolicy", #subiendo el learning_rate, sin sde
        "learning_rate":0.001,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":256,
        "tau":0.005,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    kwargs_3 = {"policy": "MlpPolicy", #subiendo el learning_rate, sin sde
        "learning_rate":0.0005,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":256,
        "tau":0.005,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    kwargs_4 = {"policy": "MlpPolicy", #bajando el learning_rate, sin sde
        "learning_rate":0.0001,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":256,
        "tau":0.005,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    
    ###################################################################################
    kwargs_5 = {"policy": "MlpPolicy", #subiendo learning_starts, sin sde
        "learning_rate":0.0003,
        "buffer_size":1000000,
        "learning_starts":200,
        "batch_size":256,
        "tau":0.005,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    kwargs_6 = {"policy": "MlpPolicy", #subiendo learning_starts, sin sde
        "learning_rate":0.0003,
        "buffer_size":1000000,
        "learning_starts":500,
        "batch_size":256,
        "tau":0.005,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    ###################################################################################
    kwargs_7 = {"policy": "MlpPolicy", #subiendo batch_size, sin sde
        "learning_rate":0.0003,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":512,
        "tau":0.005,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    kwargs_8 = {"policy": "MlpPolicy", #bajando batch_size, sin sde
        "learning_rate":0.0003,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":128,
        "tau":0.005,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    ###################################################################
    kwargs_9 = {"policy": "MlpPolicy", #subiendo tau, sin sde
        "learning_rate":0.0003,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":256,
        "tau":0.008,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    kwargs_10 = {"policy": "MlpPolicy", #bajando tau, sin sde
        "learning_rate":0.0003,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":256,
        "tau":0.002,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    ###################################################################
    kwargs_best = {"policy": "MlpPolicy", #bajando tau, sin sde
        "learning_rate":0.0005,
        "buffer_size":1000000,
        "learning_starts":100,
        "batch_size":256,
        "tau":0.008,
        "gamma": 1.0,##
        "train_freq": 32,##
        "gradient_steps" : 1,
        "ent_coef" :"auto",
        "target_update_interval": 1,
        "use_sde" : True,##
        "verbose" : 0}
    sel_dict = {'Default':default_kwargs,
                'Original':original_kwargs,
                '1':kwargs_1,
                '2':kwargs_2, 
                '3':kwargs_3,
                '4':kwargs_4, 
                '5':kwargs_5, 
                '6':kwargs_6, 
                '7':kwargs_7, 
                '8':kwargs_8, 
                '9':kwargs_9, 
                '10':kwargs_10,
                'Best':kwargs_best}
    return sel_dict[selection]

def explore_SAC(params):
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env, filename=None)
    model = SAC(env= env, **params)
    model.learn(total_timesteps=300000)
    ep_len = env.get_episode_lengths() 
    env.close()
    return ep_len 

def run_qlearning():

    env = gym.make("MountainCar-v0")
    n_actions = env.action_space.n
    env = Monitor(env, filename=None)
    qlearning = QLearning(n_actions, epsilon=0.0, alpha=0.5/8, gamma=1.0)
    for episode in range(1000):
        observation, info = env.reset()
        action = qlearning.sample_action(observation)
        terminated = truncated = False

        while not terminated and not truncated:
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_action = qlearning.sample_action(next_observation)

            qlearning.learn(observation, action, reward, next_observation, terminated)

            observation, action = next_observation, next_action
    ep_len = env.get_episode_lengths()
    env.close()
    return ep_len

def run_sarsa():
    env = gym.make("MountainCar-v0")
    n_actions = env.action_space.n
    env = Monitor(env, filename=None)
    sarsa = Sarsa(n_actions, epsilon=0.0, alpha=0.5/8, gamma=1.0)
    for episode in range(1000):
        observation, info = env.reset()
        action = sarsa.sample_action(observation)
        terminated = truncated = False

        while not terminated and not truncated:
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_action = sarsa.sample_action(next_observation)


            sarsa.learn(observation, action, reward, next_observation, next_action, terminated)

            observation, action = next_observation, next_action

    ep_len = env.get_episode_lengths()
    env.close()
    return ep_len

def run_sarsa_lambda():

    env = gym.make("MountainCar-v0")
    n_actions = env.action_space.n
    env = Monitor(env, filename=None)

    sarsa = SarsaLambda(n_actions, epsilon=0.0, alpha=0.5/8, gamma=1.0, s_lambda=0.5)
    for episode in range(1000):
        observation, info = env.reset()
        action = sarsa.sample_action(observation)
        terminated = truncated = False

        while not terminated and not truncated:
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_action = sarsa.sample_action(next_observation)

            sarsa.learn(observation, action, reward, next_observation, next_action, terminated)

            observation, action = next_observation, next_action

    ep_len = env.get_episode_lengths()  
    env.close()
    return ep_len
