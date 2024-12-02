import numpy as np
from matplotlib import pyplot as plt


def avg_over_10(eps):
    avg = []
    for i, ep in enumerate(eps):
        if i%10 == 0:
            avg.append(ep)
    return avg

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_b():
    avg_len_ActorCritic = np.load("results/average_episode_length_ActorCritic.npy")
    avg_len_SAC = np.load("results/average_episode_length_SAC.npy")
    plt.figure()
    avg_len_ActorCritic = avg_over_10(avg_len_ActorCritic)  
    avg_len_SAC = avg_over_10(avg_len_SAC)
    x = np.linspace(0, 1000, 100)
    plt.title("Actor-Critic")
    plt.xlabel("Episode")
    plt.ylabel("Average Length")
    plt.plot(x, avg_len_ActorCritic, 'r',x, avg_len_SAC, 'b')
    plt.legend(["Actor-Critic", "SAC"])
    plt.savefig("img/Actor-Critic.png")

w = 20
def plot_c_lr(w=w):
    plt.figure()

    ep_len_SAC_Original = np.load("results/episode_length_SAC_Original.npy")
    ep_len_SAC_Original = ep_len_SAC_Original[:1000]
    ep_len_SAC_Original = moving_average(ep_len_SAC_Original,w)
    xo = np.linspace(0, len(ep_len_SAC_Original), len(ep_len_SAC_Original))

    ep_len_SAC_1 = np.load("results/episode_length_SAC_1.npy")
    ep_len_SAC_1 = ep_len_SAC_1[:1000]
    ep_len_SAC_1 = moving_average(ep_len_SAC_1,w)
    x1 = np.linspace(0, len(ep_len_SAC_1), len(ep_len_SAC_1))

    ep_len_SAC_2 = np.load("results/episode_length_SAC_2.npy")
    ep_len_SAC_2 = ep_len_SAC_2[:1000]
    ep_len_SAC_2 = moving_average(ep_len_SAC_2,w)
    x2 = np.linspace(0, len(ep_len_SAC_2), len(ep_len_SAC_2))

    ep_len_SAC_3 = np.load("results/episode_length_SAC_3.npy")
    ep_len_SAC_3 = ep_len_SAC_3[:1000]
    ep_len_SAC_3 = moving_average(ep_len_SAC_3,w)
    x3 = np.linspace(0, len(ep_len_SAC_3), len(ep_len_SAC_3))

    ep_len_SAC_4 = np.load("results/episode_length_SAC_4.npy")
    ep_len_SAC_4 = ep_len_SAC_4[:1000]
    ep_len_SAC_4 = moving_average(ep_len_SAC_4,w)
    x4 = np.linspace(0, len(ep_len_SAC_4), len(ep_len_SAC_4))

    plt.title("Rendimiento de SAC, variando learning_rate")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.plot(xo,ep_len_SAC_Original,'r', x1, ep_len_SAC_1, 'b', x2,ep_len_SAC_2, 'g', x3,ep_len_SAC_3, 'y', x4,ep_len_SAC_4, 'c')
    plt.legend(["0.0003","0.0007", "0.001", "0.0005", "0.0001"])
    plt.savefig("img/learning_rate.png")

def plot_c_ls(w=w):
    plt.figure()
    ep_len_SAC_Original = np.load("results/episode_length_SAC_Original.npy")
    ep_len_SAC_Original = ep_len_SAC_Original[:1000]
    ep_len_SAC_Original = moving_average(ep_len_SAC_Original,w)
    xo = np.linspace(0, len(ep_len_SAC_Original), len(ep_len_SAC_Original))

    ep_len_SAC_5 = np.load("results/episode_length_SAC_5.npy")
    ep_len_SAC_5 = ep_len_SAC_5[:1000]
    ep_len_SAC_5 = moving_average(ep_len_SAC_5,w)
    x5 = np.linspace(0, len(ep_len_SAC_5), len(ep_len_SAC_5))

    ep_len_SAC_6 = np.load("results/episode_length_SAC_6.npy")
    ep_len_SAC_6 = ep_len_SAC_6[:1000]
    ep_len_SAC_6 = moving_average(ep_len_SAC_6,w)
    x6 = np.linspace(0, len(ep_len_SAC_6), len(ep_len_SAC_6))

    plt.title("Rendimiento de SAC, variando learning_starts")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.plot(xo,ep_len_SAC_Original,'r', x5, ep_len_SAC_5, 'b', x6,ep_len_SAC_6, 'g')
    plt.legend(["100","200", "500"])
    plt.savefig("img/learning_starts.png")

def plot_c_bs(w=w):
    plt.figure()
    ep_len_SAC_Original = np.load("results/episode_length_SAC_Original.npy")
    ep_len_SAC_Original = ep_len_SAC_Original[:1000]
    ep_len_SAC_Original = moving_average(ep_len_SAC_Original,w)
    xo = np.linspace(0, len(ep_len_SAC_Original), len(ep_len_SAC_Original))

    ep_len_SAC_7 = np.load("results/episode_length_SAC_7.npy")
    ep_len_SAC_7 = ep_len_SAC_7[:1000]
    ep_len_SAC_7 = moving_average(ep_len_SAC_7,w)
    x7 = np.linspace(0, len(ep_len_SAC_7), len(ep_len_SAC_7))

    ep_len_SAC_8 = np.load("results/episode_length_SAC_8.npy")
    ep_len_SAC_8 = ep_len_SAC_8[:1000]
    ep_len_SAC_8 = moving_average(ep_len_SAC_8,w)
    x8 = np.linspace(0, len(ep_len_SAC_8), len(ep_len_SAC_8))

    plt.title("Rendimiento de SAC, variando batch_size")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.plot(xo,ep_len_SAC_Original,'r', x7, ep_len_SAC_7, 'b', x8,ep_len_SAC_8, 'g')
    plt.legend(["256","512", "128"])
    plt.savefig("img/batch_size.png")

def plot_c_tau(w=w):
    plt.figure()
    ep_len_SAC_Original = np.load("results/episode_length_SAC_Original.npy")
    ep_len_SAC_Original = ep_len_SAC_Original[:1000]
    ep_len_SAC_Original = moving_average(ep_len_SAC_Original,w)
    xo = np.linspace(0, len(ep_len_SAC_Original), len(ep_len_SAC_Original))

    ep_len_SAC_9 = np.load("results/episode_length_SAC_9.npy")
    ep_len_SAC_9 = ep_len_SAC_9[:1000]
    ep_len_SAC_9 = moving_average(ep_len_SAC_9,w)
    x9 = np.linspace(0, len(ep_len_SAC_9), len(ep_len_SAC_9))

    ep_len_SAC_10 = np.load("results/episode_length_SAC_10.npy")
    ep_len_SAC_10 = ep_len_SAC_10[:1000]
    ep_len_SAC_10 = moving_average(ep_len_SAC_10,w)
    x10 = np.linspace(0, len(ep_len_SAC_10), len(ep_len_SAC_10))

    plt.title("Rendimiento de SAC, variando tau")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.plot(xo,ep_len_SAC_Original,'r', x9, ep_len_SAC_9, 'b', x10,ep_len_SAC_10, 'g')
    plt.legend(["0.005","0.008", "0.002"])
    plt.savefig("img/tau.png")

def plot_c_best(w=w):
    plt.figure()
    ep_len_SAC_Original = np.load("results/episode_length_SAC_Original.npy")
    ep_len_SAC_Original = ep_len_SAC_Original[:1000]
    ep_len_SAC_Original = moving_average(ep_len_SAC_Original,w)
    xo = np.linspace(0, len(ep_len_SAC_Original), len(ep_len_SAC_Original))

    ep_len_SAC_Best = np.load("results/episode_length_SAC_Best.npy")
    ep_len_SAC_Best = ep_len_SAC_Best[:1000]
    ep_len_SAC_Best = moving_average(ep_len_SAC_Best,w)
    x_best = np.linspace(0, len(ep_len_SAC_Best), len(ep_len_SAC_Best))

    plt.title("Rendimiento de SAC, mejor set de parámetros")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.plot(xo,ep_len_SAC_Original,'r', x_best, ep_len_SAC_Best, 'b')
    plt.legend(["Original","Best"])
    plt.savefig("img/best.png")
    
def plot_c_all(w=w):
    plot_c_lr(w=w)
    plot_c_ls(w=w)
    plot_c_bs(w=w)
    plot_c_tau(w=w)
    plot_c_best(w=w)
    return None

def plot_a():
    avg_len_Qlearning = np.load("results/average_episode_length_Qlearning.npy")
    avg_len_Sarsa = np.load("results/average_episode_length_Sarsa.npy")
    avg_len_SarsaLambda = np.load("results/average_episode_length_SarsaLambda.npy")
    plt.figure()
    avg_len_Qlearning = avg_over_10(avg_len_Qlearning)
    avg_len_Sarsa = avg_over_10(avg_len_Sarsa)
    avg_len_SarsaLambda = avg_over_10(avg_len_SarsaLambda)
    x = np.linspace(0, 1000, 100)
    plt.title("Rendimiento en MountainCar-v0")
    plt.xlabel("Episode")
    plt.ylabel("Average Length")
    plt.plot(x, avg_len_Qlearning, 'r',x, avg_len_Sarsa, 'b', x, avg_len_SarsaLambda, 'g')
    plt.legend(["Q-Learning", "Sarsa", "Sarsa Lambda"])
    plt.savefig("img/discrete.png")

if __name__ == "__main__":
    plot_a()
    #plot_b()
    #plot_c_all()