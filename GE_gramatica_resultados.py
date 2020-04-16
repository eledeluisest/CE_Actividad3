from utilis_gramatica import *
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

# GENERALES
N_POB = 1500
N_ITERACCIONES = 500
LON_CODON = 30

# BUSQUEDA LOCAL
DIVERSIDAD_INICIAL = 3
N_BEST = 5

# TORNEO
K = 2

# MUTACION
SWAP_PROB = 0.3

# FUNCION
NFUNCION = 4

funcion_resultado_sr = []
funcion_resultado_mbf = []
funcion_resultado_aes = []
resultado_mbf = {}
resultado_sr = {}
resultado_aes = {}
primero = True
SR = 0
MBF = 0
AES = 0

for j in [600, 1000, 1500, 2000]:
    resultado_sr[j] = []
    resultado_mbf[j] = []
    resultado_aes[j] = []
for NFUNCION in range(7):
    A, B = limites[NFUNCION]
    N = 50
    h = 1e-5
    dx = (B - A) / N

    f = funciones[NFUNCION]
    res = []
    for i in range(N):
        res.append(df(f, A, dx, h, i))

    f_values = pd.Series(res)
    f_min = f_values.min()
    f_max = f_values.max()

    # CONDICION DE SALIDA
    umbral = .1
    amp = (f_max - f_min)

    print('CONDICION DE SALIDA: mejor fitness < ' + str(amp * 0.1))
    derivada = derivadas[NFUNCION]

    with open('resultados/funcion' + str(NFUNCION) + '_' + 'ITER' + str(2) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(NFUNCION) + '.txt', 'r') as f:
        linea = f.read().split('\n')

    terminado = []
    min_fit = []
    n_iter_AES = []
    n_iter = []
    longitudes = [0]
    for i in range(10):
        fit_df = pd.read_csv('resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(NFUNCION) + '.txt', sep=';', index_col=0)
        l_excluir = longitudes[-1]
        fit_df_tmp = fit_df.iloc[l_excluir:, :]
        terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]) > 0)
        min_fit.append(fit_df_tmp.resumen_min.min())
        n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]))
        n_iter.append(len(fit_df_tmp))
        longitudes.append(len(fit_df))

    SR = np.nan_to_num(np.mean(terminado))
    MBF = np.nan_to_num(np.mean(min_fit))
    AES = np.nan_to_num(np.mean(n_iter))
    print(SR, MBF, AES)
    resultado_sr[600].append(SR)
    resultado_mbf[600].append(MBF)
    resultado_aes[600].append(AES)

    for j in [1000, 1500, 2000]:
        funcion_resultado_sr = []
        funcion_resultado_mbf = []
        funcion_resultado_aes = []
        terminado = []
        min_fit = []
        n_iter_AES = []
        n_iter = []
        SR = 0
        MBF = 0
        AES = 0
        longitudes = [0]
        for i in range(10):
            fit_df = pd.read_csv('resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
                DIVERSIDAD_INICIAL) + '_' + str(j) + '.txt', sep=';', index_col=0)
            l_excluir = longitudes[-1]
            fit_df_tmp = fit_df.iloc[l_excluir:, :]
            terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]) > 0)
            min_fit.append(fit_df_tmp.resumen_min.min())
            n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]))
            n_iter.append(len(fit_df_tmp))
            longitudes.append(len(fit_df))

        SR = np.nan_to_num(np.mean(terminado))
        MBF = np.nan_to_num(np.mean(min_fit))
        AES = np.nan_to_num(np.mean(n_iter))
        print(SR, MBF, AES)
        resultado_sr[j].append(SR)
        resultado_mbf[j].append(MBF)
        resultado_aes[j].append(AES)

width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']
fig, ax = plt.subplots()

x = np.arange(len(labels)) * 2  # the label locations
rects1 = ax.bar(x - 3 * width / 2, resultado_sr[600], width, label='Pob = 600', color='mistyrose')
rects2 = ax.bar(x - width / 2, resultado_sr[1000], width, label='Pob = 1000', color='salmon')
rects3 = ax.bar(x + width / 2, resultado_sr[1500], width, label='Pob = 1500', color='red')
rects4 = ax.bar(x + 3 * width / 2, resultado_sr[2000], width, label='Pob = 2000', color='darkred')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SR')
ax.set_xlabel('Función')
ax.set_title('SR para cada función para diferentes tamaños de población')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/SR_NPOB.png')
plt.show()
width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']
fig, ax = plt.subplots()

x = np.arange(len(labels)) * 2  # the label locations
rects1 = ax.bar(x - 3 * width / 2, resultado_mbf[600], width, label='Pob = 600', color='mistyrose')
rects2 = ax.bar(x - width / 2, resultado_mbf[1000], width, label='Pob = 1000', color='salmon')
rects3 = ax.bar(x + width / 2, resultado_mbf[1500], width, label='Pob = 1500', color='red')
rects4 = ax.bar(x + 3 * width / 2, resultado_mbf[2000], width, label='Pob = 2000', color='darkred')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MBF')
ax.set_xlabel('Función')
ax.set_title('MBF para cada función para diferentes tamaños de población')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/MBF_NPOB.png')
plt.show()
width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']
fig, ax = plt.subplots()

x = np.arange(len(labels)) * 2  # the label locations
rects1 = ax.bar(x - 3 * width / 2, resultado_aes[600], width, label='Pob = 600', color='mistyrose')
rects2 = ax.bar(x - width / 2, resultado_aes[1000], width, label='Pob = 1000', color='salmon')
rects3 = ax.bar(x + width / 2, resultado_aes[1500], width, label='Pob = 1500', color='red')
rects4 = ax.bar(x + 3 * width / 2, resultado_aes[2000], width, label='Pob = 2000', color='darkred')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AES')
ax.set_xlabel('Función')
ax.set_title('AES para cada función para diferentes tamaños de población')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/AES_NPOB.png')
plt.show()
############################
# HASTA AQUÍ METRICAS VS NPOB
############################

############################
# AHORA METRICAS VS NBEST
############################

from utilis_gramatica import *
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

# GENERALES
N_POB = 1500
N_ITERACCIONES = 500
LON_CODON = 30

# BUSQUEDA LOCAL
DIVERSIDAD_INICIAL = 3
N_BEST = 5

# TORNEO
K = 2

# MUTACION
SWAP_PROB = 0.3

# FUNCION
NFUNCION = 4

funcion_resultado_sr = []
funcion_resultado_mbf = []
funcion_resultado_aes = []
resultado_mbf = {}
resultado_sr = {}
resultado_aes = {}
primero = True
SR = 0
MBF = 0
AES = 0

for j in [5, 10, 20]:
    resultado_sr[j] = []
    resultado_mbf[j] = []
    resultado_aes[j] = []
for NFUNCION in range(7):
    A, B = limites[NFUNCION]
    N = 50
    h = 1e-5
    dx = (B - A) / N

    f = funciones[NFUNCION]
    res = []
    for i in range(N):
        res.append(df(f, A, dx, h, i))

    f_values = pd.Series(res)
    f_min = f_values.min()
    f_max = f_values.max()

    # CONDICION DE SALIDA
    umbral = .1
    amp = (f_max - f_min)

    print('CONDICION DE SALIDA: mejor fitness < ' + str(amp * 0.1))
    derivada = derivadas[NFUNCION]

    with open('resultados/funcion' + str(NFUNCION) + '_' + 'ITER' + str(2) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(NFUNCION) + '.txt', 'r') as f:
        linea = f.read().split('\n')

    terminado = []
    min_fit = []
    n_iter_AES = []
    n_iter = []
    longitudes = [0]
    for i in range(10):
        fit_df = pd.read_csv('resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(NFUNCION) + '.txt', sep=';', index_col=0)
        l_excluir = longitudes[-1]
        fit_df_tmp = fit_df.iloc[l_excluir:, :]
        terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]) > 0)
        min_fit.append(fit_df_tmp.resumen_min.min())
        n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]))
        n_iter.append(len(fit_df_tmp))
        longitudes.append(len(fit_df))

    SR = np.nan_to_num(np.mean(terminado))
    MBF = np.nan_to_num(np.mean(min_fit))
    AES = np.nan_to_num(np.mean(n_iter))
    print(SR, MBF, AES)
    resultado_sr[10].append(SR)
    resultado_mbf[10].append(MBF)
    resultado_aes[10].append(AES)

    for j in [5, 20]:
        funcion_resultado_sr = []
        funcion_resultado_mbf = []
        funcion_resultado_aes = []
        terminado = []
        min_fit = []
        n_iter_AES = []
        n_iter = []
        SR = 0
        MBF = 0
        AES = 0
        longitudes = [0]
        for i in range(10):
            fit_df = pd.read_csv('resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
                DIVERSIDAD_INICIAL) + '_' + str(1500) + 'nBEST' + str(j) + '.txt', sep=';', index_col=0)
            l_excluir = longitudes[-1]
            fit_df_tmp = fit_df.iloc[l_excluir:, :]
            terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]) > 0)
            min_fit.append(fit_df_tmp.resumen_min.min())
            n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]))
            n_iter.append(len(fit_df_tmp))
            longitudes.append(len(fit_df))

        SR = np.nan_to_num(np.mean(terminado))
        MBF = np.nan_to_num(np.mean(min_fit))
        AES = np.nan_to_num(np.mean(n_iter))
        print(SR, MBF, AES)
        resultado_sr[j].append(SR)
        resultado_mbf[j].append(MBF)
        resultado_aes[j].append(AES)

width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']
fig, ax = plt.subplots()

x = np.arange(len(labels)) * 2  # the label locations
rects1 = ax.bar(x - width, resultado_sr[5], width, label='N Best = 5', color='mistyrose')
rects2 = ax.bar(x, resultado_sr[10], width, label='N Best = 10', color='salmon')
rects4 = ax.bar(x + width, resultado_sr[20], width, label='N Best = 20', color='darkred')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SR')
ax.set_xlabel('Función')
ax.set_title('SR para cada función para diferentes tamaños de población')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/SR_NBEST.png')
plt.show()
width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']
fig, ax = plt.subplots()

x = np.arange(len(labels)) * 2  # the label locations
rects1 = ax.bar(x - width, resultado_mbf[5], width, label='N Best = 5', color='mistyrose')
rects2 = ax.bar(x, resultado_mbf[10], width, label='N Best = 10', color='salmon')
rects4 = ax.bar(x + width, resultado_mbf[20], width, label='N Best = 20', color='darkred')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MBF')
ax.set_xlabel('Función')
ax.set_title('MBF para cada función para diferentes tamaños de población')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/MBF_NBEST.png')
plt.show()
width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']
fig, ax = plt.subplots()

x = np.arange(len(labels)) * 2  # the label locations

rects1 = ax.bar(x - width, resultado_aes[5], width, label='N Best = 5', color='mistyrose')
rects2 = ax.bar(x, resultado_aes[10], width, label='N Best = 10', color='salmon')
rects4 = ax.bar(x + width, resultado_aes[20], width, label='N Best = 20', color='darkred')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AES')
ax.set_xlabel('Función')
ax.set_title('AES para cada función para diferentes tamaños de población')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/AES_NBEST.png')
plt.show()

############################
# AHORA METRICAS VS LON CODON
############################

from utilis_gramatica import *
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

# GENERALES
N_POB = 1500
N_ITERACCIONES = 500
LON_CODON = 30

# BUSQUEDA LOCAL
DIVERSIDAD_INICIAL = 3
N_BEST = 5

# TORNEO
K = 2

# MUTACION
SWAP_PROB = 0.3

# FUNCION
NFUNCION = 4

funcion_resultado_sr = []
funcion_resultado_mbf = []
funcion_resultado_aes = []
resultado_mbf = {}
resultado_sr = {}
resultado_aes = {}
primero = True
SR = 0
MBF = 0
AES = 0

for j in [10, 20, 30, 40, 50, 75, 100]:
    resultado_sr[j] = []
    resultado_mbf[j] = []
    resultado_aes[j] = []
for NFUNCION in range(7):
    A, B = limites[NFUNCION]
    N = 50
    h = 1e-5
    dx = (B - A) / N

    f = funciones[NFUNCION]
    res = []
    for i in range(N):
        res.append(df(f, A, dx, h, i))

    f_values = pd.Series(res)
    f_min = f_values.min()
    f_max = f_values.max()

    # CONDICION DE SALIDA
    umbral = .1
    amp = (f_max - f_min)

    print('CONDICION DE SALIDA: mejor fitness < ' + str(amp * 0.1))
    derivada = derivadas[NFUNCION]

    with open('resultados/funcion' + str(NFUNCION) + '_' + 'ITER' + str(2) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(NFUNCION) + '.txt', 'r') as f:
        linea = f.read().split('\n')

    terminado = []
    min_fit = []
    n_iter_AES = []
    n_iter = []
    longitudes = [0]
    for i in range(10):
        fit_df = pd.read_csv('resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(NFUNCION) + '.txt', sep=';', index_col=0)
        l_excluir = longitudes[-1]
        fit_df_tmp = fit_df.iloc[l_excluir:, :]
        terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]) > 0)
        min_fit.append(fit_df_tmp.resumen_min.min())
        n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]))
        n_iter.append(len(fit_df_tmp))
        longitudes.append(len(fit_df))

    SR = np.nan_to_num(np.mean(terminado))
    MBF = np.nan_to_num(np.mean(min_fit))
    AES = np.nan_to_num(np.mean(n_iter))
    print(SR, MBF, AES)
    resultado_sr[30].append(SR)
    resultado_mbf[30].append(MBF)
    resultado_aes[30].append(AES)

    for j in [10, 20, 40, 50, 75, 100]:
        funcion_resultado_sr = []
        funcion_resultado_mbf = []
        funcion_resultado_aes = []
        terminado = []
        min_fit = []
        n_iter_AES = []
        n_iter = []
        SR = 0
        MBF = 0
        AES = 0
        longitudes = [0]
        for i in range(10):
            try:
                fit_df = pd.read_csv(
                    'resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
                        DIVERSIDAD_INICIAL) + '_' + str(1500) + 'nBEST' + str(5) + 'LON_CODON' + str(j) + '.txt',
                    sep=';', index_col=0)
                l_excluir = longitudes[-1]
                fit_df_tmp = fit_df.iloc[l_excluir:, :]
                terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]) > 0)
                min_fit.append(fit_df_tmp.resumen_min.min())
                n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]))
                n_iter.append(len(fit_df_tmp))
                longitudes.append(len(fit_df))
            except FileNotFoundError:
                terminado = []

        if len(terminado) > 1:
            SR = np.nan_to_num(np.mean(terminado))
            MBF = np.nan_to_num(np.mean(min_fit))
            AES = np.nan_to_num(np.mean(n_iter))
        print(SR, MBF, AES)
        resultado_sr[j].append(SR)
        resultado_mbf[j].append(MBF)
        resultado_aes[j].append(AES)

width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']
fig, ax = plt.subplots()

x = np.arange(len(labels)) * 3  # the label locations
rects1 = ax.bar(x - 3 * width, resultado_sr[10], width, label='Longitud Codon = 10', color='rosybrown')
rects2 = ax.bar(x - 2 * width, resultado_sr[20], width, label='N Best = 20', color='mistyrose')
rects3 = ax.bar(x - width, resultado_sr[30], width, label='N Best = 30', color='salmon')
rects4 = ax.bar(x, resultado_sr[40], width, label='N Best = 40', color='tomato')
rects5 = ax.bar(x + width, resultado_sr[50], width, label='N Best = 50', color='red')
rects6 = ax.bar(x + 2 * width, resultado_sr[75], width, label='N Best = 75', color='darkred')
rects7 = ax.bar(x + 3 * width, resultado_sr[100], width, label='N Best = 100', color='brown')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SR')
ax.set_xlabel('Función')
ax.set_title('SR para cada función para diferentes tamaños de población')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/SR_CODON_LON.png')
plt.show()
width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']
fig, ax = plt.subplots()

x = np.arange(len(labels)) * 3  # the label locations
rects1 = ax.bar(x - 3 * width, resultado_mbf[10], width, label='Longitud Codon = 10', color='rosybrown')
rects2 = ax.bar(x - 2 * width, resultado_mbf[20], width, label='N Best = 20', color='mistyrose')
rects3 = ax.bar(x - width, resultado_mbf[30], width, label='N Best = 30', color='salmon')
rects4 = ax.bar(x, resultado_mbf[40], width, label='N Best = 40', color='tomato')
rects5 = ax.bar(x + width, resultado_mbf[50], width, label='N Best = 50', color='red')
rects6 = ax.bar(x + 2 * width, resultado_mbf[75], width, label='N Best = 75', color='darkred')
rects7 = ax.bar(x + 3 * width, resultado_mbf[100], width, label='N Best = 100', color='brown')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MBF')
ax.set_xlabel('Función')
ax.set_title('MBF para cada función para diferentes tamaños de población')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/MBF_CODON_LON.png')
plt.show()
width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']
fig, ax = plt.subplots()

x = np.arange(len(labels)) * 3  # the label locations

rects1 = ax.bar(x - 3 * width, resultado_aes[10], width, label='Longitud Codon = 10', color='rosybrown')
rects2 = ax.bar(x - 2 * width, resultado_aes[20], width, label='Longitud Codon = 20', color='mistyrose')
rects3 = ax.bar(x - width, resultado_aes[30], width, label='Longitud Codon = 30', color='salmon')
rects4 = ax.bar(x, resultado_aes[40], width, label='Longitud Codon = 40', color='tomato')
rects5 = ax.bar(x + width, resultado_aes[50], width, label='Longitud Codon = 50', color='red')
rects6 = ax.bar(x + 2 * width, resultado_aes[75], width, label='Longitud Codon = 75', color='darkred')
rects7 = ax.bar(x + 3 * width, resultado_aes[100], width, label='Longitud Codon = 100', color='brown')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AES')
ax.set_xlabel('Función')
ax.set_title('AES para cada función para diferentes tamaños de población')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/AES_CODON_LON.png')
plt.show()

############################
# EJECUCIONES FINALES
############################

from utilis_gramatica import *
import matplotlib.pyplot as plt
import pandas as pd
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
NFUNCION = 4

resultado_mbf = []
resultado_sr = []
resultado_aes = []
primero = True
SR = 0
MBF = 0
AES = 0

for NFUNCION in range(7):
    A, B = limites[NFUNCION]
    N = 50
    h = 1e-5
    dx = (B - A) / N

    f = funciones[NFUNCION]
    res = []
    for i in range(N):
        res.append(df(f, A, dx, h, i))

    f_values = pd.Series(res)
    f_min = f_values.min()
    f_max = f_values.max()

    # CONDICION DE SALIDA
    umbral = .1
    amp = (f_max - f_min)

    print('CONDICION DE SALIDA: mejor fitness < ' + str(amp * 0.1))
    derivada = derivadas[NFUNCION]

    with open('resultados/funcion' + str(NFUNCION) + '_' + 'ITER' + str(2) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(NFUNCION) + '.txt', 'r') as f:
        linea = f.read().split('\n')

    terminado = []
    min_fit = []
    n_iter_AES = []
    n_iter = []
    longitudes = [0]
    for i in range(100):
        try:
            fit_df = pd.read_csv('resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
                DIVERSIDAD_INICIAL) + '_' + str(1500) + 'nBEST' + str(5) + 'LON_CODON' + str(75) + '_FINAL.txt',
                                 sep=';',
                                 index_col=0)
            l_excluir = longitudes[-1]
            fit_df_tmp = fit_df.iloc[l_excluir:, :]
            terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]) > 0)
            min_fit.append(fit_df_tmp.resumen_min.min())
            n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]))
            n_iter.append(len(fit_df_tmp))
            longitudes.append(len(fit_df))
        except FileNotFoundError:
            print('Excepcion')

    if len(terminado) > 1:
        SR = np.nan_to_num(np.mean(terminado))
        MBF = np.nan_to_num(np.mean(min_fit))
        AES = np.nan_to_num(np.mean(n_iter))

    resultado_sr.append(SR)
    resultado_mbf.append(MBF)
    resultado_aes.append(AES)

resultado_mbf_exact = []
resultado_sr_exact = []
resultado_aes_exact = []
primero = True
SR = 0
MBF = 0
AES = 0

for NFUNCION in range(7):
    A, B = limites[NFUNCION]
    N = 50
    h = 1e-5
    dx = (B - A) / N

    f = funciones[NFUNCION]
    res = []
    for i in range(N):
        res.append(df(f, A, dx, h, i))

    f_values = pd.Series(res)
    f_min = f_values.min()
    f_max = f_values.max()

    # CONDICION DE SALIDA
    umbral = .1
    amp = (f_max - f_min)

    print('CONDICION DE SALIDA: mejor fitness < ' + str(0.1))
    derivada = derivadas[NFUNCION]

    with open('resultados/funcion' + str(NFUNCION) + '_' + 'ITER' + str(2) + '_DivIni' + str(
            DIVERSIDAD_INICIAL) + '_' + str(NFUNCION) + '.txt', 'r') as f:
        linea = f.read().split('\n')

    terminado = []
    min_fit = []
    n_iter_AES = []
    n_iter = []
    longitudes = [0]
    for i in range(100):
        try:
            fit_df = pd.read_csv('resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
                DIVERSIDAD_INICIAL) + '_' + str(1500) + 'nBEST' + str(5) + 'LON_CODON' + str(75) + '_FINAL.txt',
                                 sep=';',
                                 index_col=0)
            l_excluir = longitudes[-1]
            fit_df_tmp = fit_df.iloc[l_excluir:, :]
            terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (0.1)]) > 0)
            min_fit.append(fit_df_tmp.resumen_min.min())
            n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (0.1)]))
            n_iter.append(len(fit_df_tmp))
            longitudes.append(len(fit_df))
        except FileNotFoundError:
            print('Excepcion')

    if len(terminado) > 1:
        SR = np.nan_to_num(np.mean(terminado))
        MBF = np.nan_to_num(np.mean(min_fit))
        AES = np.nan_to_num(np.mean(n_iter))

    resultado_sr_exact.append(SR)
    resultado_mbf_exact.append(MBF)
    resultado_aes_exact.append(AES)

width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']

x = np.arange(len(labels))  # the label locations
fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, resultado_sr_exact, width, label='Condición de salida del enunciado: 0.1',
                color='salmon')
rects2 = ax.bar(x + width / 2, resultado_sr, width, label='Condicion de salida ralajada: AMP*0.1', color='red')
ax.set_ylabel('SR')
ax.set_xlabel('Función')
ax.set_title('SR')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/SR_FINAL.png')
plt.show()

width = 0.35  # the width of the bars
labels = ['1', '2', '3', '4', '5', '6', '7']

x = np.arange(len(labels))  # the label locations
fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, resultado_mbf_exact, width, label='Condición de salida del enunciado: 0.1',
                color='salmon')
rects2 = ax.bar(x + width / 2, resultado_mbf, width, label='Condicion de salida ralajada: AMP*0.1', color='red')
ax.set_ylabel('MBF')
ax.set_xlabel('Función')
ax.set_title('MBF')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/MBF_FINAL.png')
plt.show()

x = np.arange(len(labels))  # the label locations
fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, resultado_aes_exact, width, label='Condición de salida del enunciado: 0.1',
                color='salmon')
rects2 = ax.bar(x + width / 2, resultado_aes, width, label='Condicion de salida ralajada: AMP*0.1', color='red')
ax.set_ylabel('AES')
ax.set_xlabel('Función')
ax.set_title('AES')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('resultados/AES_FINAL.png')
plt.show()

############################
# FUNCIONES GRÁFICAS
############################

funcion_resultado_sr = []
funcion_resultado_mbf = []
funcion_resultado_aes = []
resultado_sr_best = {}
resultado_mbf_best = {}
resultado_aes_best = {}
primero = True
SR = 0
MBF = 0
AES = 0
bool_bf = False
if bool_bf == True:
    best_funcs = []
best_exp = []
for NFUNCION in range(7):
    A, B = limites[NFUNCION]
    N = 50
    h = 1e-5
    dx = (B - A) / N

    f = funciones[NFUNCION]
    res = []
    for i in range(N):
        res.append(df(f, A, dx, h, i))

    f_values = pd.Series(res)
    f_min = f_values.min()
    f_max = f_values.max()

    # CONDICION DE SALIDA
    umbral = .1
    amp = (f_max - f_min)

    print('CONDICION DE SALIDA: mejor fitness < ' + str(amp * 0.1))
    derivada = derivadas[NFUNCION]
    fig = plt.figure()
    plt.plot([i * (B - A) / 50 + A for i in range(50)], [derivada(i * (B - A) / 50 + A) for i in range(50)],
             linewidth=5, color='black', label='Función a ajustar')
    best_fit = 999
    best_exp_index = 999
    for i in range(100):
        try:
            with open('resultados/funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
                    DIVERSIDAD_INICIAL) + '_' + str(1500) + 'nBEST' + str(5) + 'LON_CODON' + str(75) + '_FINAL.txt',
                      'r') as f:
                linea = f.read().split('\n')
            fit_df = pd.read_csv('resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
                DIVERSIDAD_INICIAL) + '_' + str(1500) + 'nBEST' + str(5) + 'LON_CODON' + str(75) + '_FINAL.txt',
                                 sep=';',
                                 index_col=0)
            if fit_df.resumen_min.min() < best_fit:
                best_exp_index = i
            exp = linea[0]
            func = lambda x: exec(exp)
            valores = [float(x) for x in linea[1].split(';')]
            derivada = derivadas[NFUNCION]
            if i == best_funcs[NFUNCION]:
                best_exp.append(exp)
                plt.plot([i * (B - A) / 50 + A for i in range(50)], valores, '--', linewidth=2, color='blue',
                         label='Best Fit')
            elif i == 0:
                plt.plot([i * (B - A) / 50 + A for i in range(50)], valores, '--', linewidth=1, color='red',
                         label='Resultado de la función')
            else:
                plt.plot([i * (B - A) / 50 + A for i in range(50)], valores, '--', linewidth=1, color='red')
        except:
            print(' Excepcion ')
    plt.title('FUNCIÓN ' + str(NFUNCION + 1))
    plt.xlim((A, B))
    plt.legend()
    fig.savefig('resultados/FINAL'+str(NFUNCION)+'.png')
    plt.show()
    if bool_bf == True:
        best_funcs.append(best_exp_index)

terminado = []
min_fit = []
n_iter_AES = []
n_iter = []

for NBEST_ITER in [5, 10, 20]:
    longitudes = [0]
    if NBEST_ITER == 10:
        for i in range(10):
            it_df = pd.read_csv('resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
                DIVERSIDAD_INICIAL) + '_' + str(1500) + 'nBEST' + str(5) + 'LON_CODON' + str(75) + '_FINAL.txt',
                                sep=';',
                                index_col=0)
            l_excluir = longitudes[-1]
            fit_df_tmp = fit_df.iloc[l_excluir:, :]
            terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]) > 0)
            min_fit.append(fit_df_tmp.resumen_min.min())
            n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]))
            n_iter.append(len(fit_df_tmp))
            longitudes.append(len(fit_df))

        SR = np.nan_to_num(np.mean(terminado))
        MBF = np.nan_to_num(np.mean(min_fit))
        AES = np.nan_to_num(np.mean(n_iter))
        print(SR, MBF, AES)
        resultado_sr_best[10].append(SR)
        resultado_mbf_best[10].append(MBF)
        resultado_aes_best[10].append(AES)
    else:
        for i in range(10):
            fit_df = pd.read_csv(
                'resultados/data_funcion' + str(NFUNCION) + '_' + 'ITER' + str(i) + '_DivIni' + str(
                    DIVERSIDAD_INICIAL) + '_' + str(1500) + 'nBEST' + str(NBEST_ITER) + '.txt', sep=';',
                index_col=0)
            l_excluir = longitudes[-1]
            fit_df_tmp = fit_df.iloc[l_excluir:, :]
            terminado.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]) > 0)
            min_fit.append(fit_df_tmp.resumen_min.min())
            n_iter_AES.append(len(fit_df_tmp[fit_df_tmp.resumen_min < (amp * 0.1)]))
            n_iter.append(len(fit_df_tmp))
            longitudes.append(len(fit_df))

        SR = np.nan_to_num(np.mean(terminado))
        MBF = np.nan_to_num(np.mean(min_fit))
        AES = np.nan_to_num(np.mean(n_iter))
        print(SR, MBF, AES)
        resultado_sr_best[NBEST_ITER].append(SR)
        resultado_mbf_best[NBEST_ITER].append(MBF)
        resultado_aes_best[NBEST_ITER].append(AES)

#########
# Otras cosas
########


