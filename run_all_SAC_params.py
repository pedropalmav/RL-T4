from running_loops import explore_SAC, init_parameters_SAC
import numpy as np
from tqdm.auto import tqdm

keys = ["Original", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
for key in tqdm(keys):
    params = init_parameters_SAC(key)
    print(f"Explorando parámetros, con set de parámetros {key}")
    episode_length = explore_SAC(params)
    print("Ejecución finalizada")
    np.save(f"results/episode_length_SAC_{key}.npy", episode_length)