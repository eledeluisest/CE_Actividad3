"""
Algoritmo genético
"""

from utilis_gramatica import *
import random
import time

# GENERALES
N_POB = 1500
N_ITERACCIONES = 500
LON_CODON = 75

# BUSQUEDA LOCAL
DIVERSIDAD_INICIAL = 3
N_BEST = 5

# TORNEO
K = 2

# MUTACION
SWAP_PROB = 0.3

# FUNCION
NFUNCION = 6
f = funciones[NFUNCION]
A, B = limites[NFUNCION]

N = 50
h = 1e-5
dx = (B - A) / N


res = []
for i in range(N):
    res.append(df(f, A, dx, h, i))
f_values = pd.Series(res)
f_min = f_values.min()
f_max = f_values.max()

resumen_min = []
resumen_mean = []
resumen_sd = []
time_list = []
for i_iter in range(100):
    # CONDICION DE SALIDA
    umbral = .1
    amp = (f_max - f_min)
    print('CONDICION DE SALIDA: mejor fitness < ' + str(0.01))
    t0 = time.time()
    poblacion, expresiones = genera_poblacion(N_POB, longitud=LON_CODON, diversidad_inicial=DIVERSIDAD_INICIAL)
    # Restringimos las soluciones imposibles penalizando el fitness
    df_gen = evalua(poblacion, dx, N, A)
    df_gen['fitness'] = df_gen.apply(error, axis=1, args=(f_values,))
    df_gen = df_gen.sort_values('fitness')
    for j in range(N_ITERACCIONES):
        poblacion, df_gen = busc_local(df_gen, poblacion, dx=dx, N=N, A=A, f_values=f_values, N_best=N_BEST)
        padres = []
        for i in range(N_POB):
            rand_index = random.choices(list(range(N_POB)), k=K)
            padres.append(torneo(df_gen, poblacion, rand_index))
        descendencia = []
        for i in (range(N_POB // 2)):
            padre1, padre2 = random.choices(padres, k=K)
            hijo1, hijo2 = two_poin_cross(padre1, padre2)
            descendencia = descendencia + [swap_mutation(hijo1, swap_prob=SWAP_PROB),
                                           swap_mutation(hijo2, swap_prob=SWAP_PROB)]
        # # Actualizamos y Elitimo
        df_desc = evalua(descendencia, dx, N, A)
        df_desc['fitness'] = df_desc.apply(error, axis=1, args=(f_values,))
        df_desc = df_desc.sort_values('fitness')
        # El mejor de la población pasa directamente 5 veces a la descendencia
        for i in range(3):
            descendencia[df_desc.iloc[-1, :].name] = poblacion[df_gen.iloc[0, :].name]
            df_desc.iloc[df_desc.iloc[-1, :].name, :] = df_gen.iloc[df_gen.iloc[0, :].name, :]
        poblacion = descendencia.copy()
        df_gen = df_desc.copy()
        resumen_min.append(df_gen.fitness.min())
        resumen_mean.append(df_gen.fitness.mean())
        resumen_sd.append(df_gen.fitness.std())
        time_list.append(time.time())
        if (j % 10) == 0:
            print(df_gen.fitness.min(), df_gen.fitness.mean(),
                  ret_op_write(poblacion[df_gen.loc[df_gen.fitness == df_gen.fitness.min(), :].iloc[0].name], x='x'))
        if (df_gen.fitness.min() < 0.01) or all(np.array(resumen_min[-100:]) == resumen_min[-1]):
            print(' EJECUCION FINAL ')
            print(df_gen.fitness.min(), df_gen.fitness.mean(),
                  ret_op_write(poblacion[df_gen.loc[df_gen.fitness == df_gen.fitness.min(), :].iloc[0].name], x='x'))
            break

    # fu = lambda x:  (x*(x+(x+1*x)))+np.sin((x+(x+1*x)))
    fu_or = derivadas[NFUNCION]
    # valores = [fu(A + dx * i) for i in range(N)]
    valores2 = [fu_or(A + dx * i) for i in range(N)]
    valores_exp = df_gen.loc[df_gen.fitness == df_gen.fitness.min(), :].iloc[0].iloc[:-1].values
    f_res = ret_op_write(poblacion[df_gen.loc[df_gen.fitness == df_gen.fitness.min(), :].iloc[0].name], x='x')
    resumen_min.append(df_gen.fitness.min())
    resumen_mean.append(df_gen.fitness.mean())
    resumen_sd.append(df_gen.fitness.std())
    time_list.append(time.time())
    dict_resultados = {'resumen_min': resumen_min, 'resumen_mean': resumen_mean, 'resumen_sd': resumen_sd,
                       'time_list': time_list}
    with open('resultados/funcion' + str(NFUNCION) + '_' + 'ITER' + str(i_iter) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(N_POB) +'nBEST'+str(N_BEST)+'LON_CODON'+str(LON_CODON)+'_FINAL.txt', 'w') as f:
        f.write('\n'.join([f_res, ';'.join([str(x) for x in valores_exp])]))
    pd.DataFrame(dict_resultados).to_csv(
        'resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i_iter) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(N_POB) +'nBEST'+str(N_BEST)+'LON_CODON'+str(LON_CODON)+'_FINAL.txt', sep=';')
