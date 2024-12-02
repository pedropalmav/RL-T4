from running_loops import run_actor_critic, run_SAC, explore_SAC, init_parameters_SAC, run_qlearning, run_sarsa, run_sarsa_lambda
import numpy as np
from tqdm.auto import tqdm

def average(avg, new_data, k):
    avg += (np.array(new_data) - avg) / (k + 1)
    return avg
if __name__ == "__main__":
    run = True
    while run:
        selection = input("Seleccione el algoritmo a ejecutar:\na1. Q-Learning\na2. Sarsa\na3. Sarsa Lambda\nb1. Actor-Critic\nb2. SAC\nc. Explorar parámetros SAC\n")
        if selection == "b1":
            average_episode_length = np.zeros(1000)
            for k in tqdm(range(30)):
                episode_length = run_actor_critic()
                average_episode_length = average(average_episode_length, episode_length, k)
            run = False
            print("Ejecución finalizada")
            np.save("results/average_episode_length_ActorCritic.npy", average_episode_length)
        elif selection == "b2": 
            average_episode_length = np.zeros(1000)
            for i in tqdm(range(30)): 
                episode_length = run_SAC()
                episode_length = episode_length[:1000]
                average_episode_length = average(average_episode_length, episode_length, i)
            run = False
            print("Ejecución finalizada") 
            np.save("results/average_episode_length_SAC.npy", average_episode_length)
        elif selection == "c":
            selection = input("Seleccione el set de parámetros a explorar:\nDefault\nOriginal\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\nBest\n")
            params = init_parameters_SAC(selection)
            print(f"Explorando parámetros, con set de parámetros {selection}")
            episode_length = explore_SAC(params)
            run = False
            print("Ejecución finalizada")
            np.save(f"results/episode_length_SAC_{selection}.npy", episode_length)
        elif selection == "a1":
            average_episode_length = np.zeros(1000)
            for k in tqdm(range(30)):
                episode_length = run_qlearning()
                average_episode_length = average(average_episode_length, episode_length, k)
            run = False
            print("Ejecución finalizada")
            np.save("results/average_episode_length_Qlearning.npy", average_episode_length)
        elif selection == "a2":
            average_episode_length = np.zeros(1000)
            for k in tqdm(range(30)):
                episode_length = run_sarsa()
                average_episode_length = average(average_episode_length, episode_length, k)
            run = False
            print("Ejecución finalizada")
            np.save("results/average_episode_length_Sarsa.npy", average_episode_length)
        elif selection == "a3":
            average_episode_length = np.zeros(1000)
            for k in tqdm(range(30)):
                episode_length = run_sarsa_lambda()
                average_episode_length = average(average_episode_length, episode_length, k)
            run = False
            print("Ejecución finalizada")
            np.save("results/average_episode_length_SarsaLambda.npy", average_episode_length)
        else:
            print("Selección inválida")
    