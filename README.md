# RL-T4
## Integrantes: Javier Campos y Pedro Palma

 Para correr los experimentos, basta con correr `main.py` y seleccionar la opción correspondiente a la pregunta a replicar (Imngresar "a1" para Q-Learning, "b1" para actor critic, etc).

 En caso de seleccionar c, deberás ingresar también el set de parámetros. Algunas opciones interesantes son "Original" y "Best". Dirigirse al archivo `running_loops.py` para revisar en detalle los diccionarios que representan a cada configuración de parámetros.

 Existe la opcion de correr el archivo `run_all_SAC_params.py` para correr todos los experimentos de c de una sola vez (sin incluir "best"), y evitar la interacción repetida con el selector del main. Lo recomendamos.

Los gráficos se pueden generar corriendo `graph.py` y descomentando las opciones disponibles en las ultimas tres lineas del archivo.

Los resultados de los experimentos se guardan en la carpeta results (como .npy), mientras que los gráficos se guardan en formato png en la carpeta img.
