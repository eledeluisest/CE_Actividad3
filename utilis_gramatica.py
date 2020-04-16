import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


def op_0(a, b):
    try:
        return a + b
    except Exception:
        raise ValueError(' Propagacion del error en la operación')


def op_1(a, b):
    try:
        return a - b
    except Exception:
        raise ValueError(' Propagacion del error en la operación')


def op_2(a, b):
    try:
        return a * b
    except Exception:
        raise ValueError(' Propagacion del error en la operación')


def op_3(a, b):
    try:
        return a / b
    except Exception:
        raise ValueError(' Propagacion del error en la operación')


def pre_op_0(x):
    try:
        return np.sin(x)
    except:
        raise ValueError('Pre Op no compatible. Cadena no compatible')


def pre_op_1(x):
    try:
        return np.cos(x)
    except:
        raise ValueError('Pre Op no compatible. Cadena no compatible')


def pre_op_2(x):
    try:
        return np.exp(x)
    except:
        raise ValueError('Pre Op no compatible. Cadena no compatible')


def pre_op_3(x):
    if x > 0:
        try:
            return np.log(x)
        except:
            raise ValueError('Pre Op no compatible. Cadena no compatible')
    else:
        raise ValueError(' Operacion incorrecta ')


def expr_0(exp1, op, exp2):
    try:
        return op(exp1, exp2)
    except:
        raise ValueError(' Cadena no compatible ')


def expr_1(exp1, op, exp2):
    try:
        return (op(exp1, exp2))
    except:
        raise ValueError(' Cadena no compatible ')


def expr_2(pre, exp1):
    try:
        return pre(exp1)
    except:
        raise ValueError(' Cadena no compatible ')


def expr_3(var):
    try:
        return var
    except:
        raise ValueError(' Cadena no compatible ')


expr = ([expr_0, 3], [expr_1, 3], [expr_2, 2], [expr_3, 1])
dic_pre_op = (pre_op_0, pre_op_1, pre_op_2, pre_op_3)
dict_var = (lambda x: x, lambda x: x, 1, 1)
dict_op = (op_0, op_1, op_2, op_3)


def ret_op(codon, x, cod_index=0):
    if cod_index < len(codon):
        cod_tmp = codon[cod_index]
        if cod_tmp == '0':
            if (cod_index + 3) < len(codon) and (int(codon[cod_index + 1 + 1]) < 4):
                codon_aux = codon[expr[int(cod_tmp)][1]:]
                exp = expr[int(cod_tmp)][0](ret_op(codon_aux, x, cod_index + 1 ), dict_op[int(codon[cod_index + 1 + 1])],
                                            ret_op(codon_aux, x, cod_index + 1 + 2))
            else:
                raise ValueError('Cadena no compatible con la gramática')

        elif cod_tmp == '1':
            if (cod_index + 3) < len(codon) and (int(codon[cod_index + 1 + 1]) < 4):
                codon_aux = codon[expr[int(cod_tmp)][1]:]
                exp = expr[int(cod_tmp)][0](ret_op(codon_aux, x, cod_index + 1 ), dict_op[int(codon[cod_index + 1 + 1])],
                                            ret_op(codon_aux, x, cod_index + 1 + 2))
            else:
                raise ValueError('Cadena no compatible con la gramática')
        elif cod_tmp == '2':
            if (cod_index + 2) < len(codon) and (int(codon[cod_index + 1]) < 4):
                codon_aux = codon[expr[int(cod_tmp)][1]:]
                exp = expr[int(cod_tmp)][0](dic_pre_op[int(codon[cod_index + 1])], ret_op(codon_aux, x, cod_index + 1 + 1))
            else:
                raise ValueError('Cadena no compatible con la gramática')
        elif cod_tmp == '3':
            if int(codon[cod_index + 1]) < 2:
                # print(' imprimo x ')
                exp = dict_var[int(codon[cod_index + 1])](x)
            elif int(codon[cod_index + 1]) < 4:
                exp = dict_var[int(codon[cod_index + 1])]
            else:
                raise ValueError('Cadena no compatible con la gramática')
        if abs(exp) != np.inf and ~np.isnan(exp):
            return exp
        else:
            raise ValueError(' Cadena da valor infinito')
    else:
        raise ValueError('Cadena no compatible con la gramática')


def genera_codon(min=0, max=4, longitud=60):
    return ''.join([str(np.random.randint(min, max)) for x in range(longitud)])


def df(f, A, dx, h, i):
    return (f((A + i * dx) + h) - f(A + i * dx)) / h


def error(gen_values, f_values, U=0.1, k0=1, k1=10):
    t1 = sum(abs(f_values - gen_values)[abs(f_values - gen_values) <= U] * k0)
    t2 = sum(abs(f_values - gen_values)[abs(f_values - gen_values) > U] * k1)
    return (t1 + t2) / (len(f_values) + 1)


# Cruzamos
def one_poin_cross(padre1, padre2):
    indice = random.randint(0, len(padre1))
    return padre1[:indice] + padre2[indice:], padre2[:indice] + padre1[indice:]


def two_poin_cross(padre1, padre2):
    indice = random.randint(0, len(padre1))
    indice2 = random.randint(0, len(padre1))
    if indice < indice2:
        return padre1[:indice] + padre2[indice:indice2] + padre1[indice2:], padre2[:indice] + padre1[
                                                                                              indice:indice2] + padre2[
                                                                                                                indice2:]
    elif indice > indice2:
        return padre1[:indice2] + padre2[indice2:indice] + padre1[indice:], padre2[:indice2] + padre1[
                                                                                               indice2:indice] + padre2[
                                                                                                                 indice:]
    else:
        return padre1, padre2


# Mutamos
def swap_mutation(individuo, swap_prob=.5):
    """
    Implementación de la mutación
    :param individuo: Individuo de la población
    :param swap_prob: Probabilidad de que se produzca la mutación
    :return: Devuelve el indivudo mutado
    """
    individuo = list(individuo)
    # Método de la altura, generamos un número aletario y si es menor que la probabilidades hacemos la mutación.
    if random.uniform(0, 1) <= swap_prob:
        punto_1 = int(random.uniform(0, len(individuo)))
        punto_2 = int(random.uniform(0, len(individuo)))
        tmp = individuo[punto_1]
        individuo[punto_1] = individuo[punto_2]
        individuo[punto_2] = tmp
    return ''.join(individuo)


def torneo(df_gen, poblacion, rand_index):
    if df_gen.iloc[rand_index[0], -1] < df_gen.iloc[rand_index[1], -1]:
        return poblacion[rand_index[0]]
    else:
        return poblacion[rand_index[1]]


def evalua(poblacion, dx, N, A, penalty=10e9):
    todo = []
    for p in poblacion:
        p1 = []
        i = 0
        while len(p1) < N:
            x = A + dx * i
            try:
                p1.append(ret_op(p, x))
            except:
                p1.append(penalty)
            i += 1
        todo.append(p1)
    return pd.DataFrame(todo)


def ret_op_write(codon, x, cod_index=0):
    def expr_0_write(exp1, op, exp2):
        try:
            return exp1 + op + exp2
        except:
            raise ValueError(' Cadena no compatible ')

    def expr_1_write(exp1, op, exp2):
        try:
            return '(' + exp1 + op + exp2 + ')'
        except:
            raise ValueError(' Cadena no compatible ')

    def expr_2_write(pre, exp1):
        try:
            return pre + '(' + exp1 + ')'
        except:
            raise ValueError(' Cadena no compatible ')

    def expr_3_write(var):
        try:
            return var
        except:
            raise ValueError(' Cadena no compatible ')

    expr_write = ([expr_0_write, 3], [expr_1_write, 3], [expr_2_write, 2], [expr_3_write, 1])
    dic_pre_op_write = ('np.sin', 'np.cos', 'np.exp', 'np.log')
    dict_var_write = ('x', 'x', '1', '1')
    dict_op_write = ('+', '-', '*', '/')
    if cod_index < len(codon):
        cod_tmp = codon[cod_index]
        if cod_tmp == '0':
            if (cod_index + 1 + 2) < len(codon) and (int(codon[cod_index + 1 + 1]) < 4):
                codon_aux = codon[expr_write[int(cod_tmp)][1]:]
                exp = expr_write[int(cod_tmp)][0](ret_op_write(codon_aux, x, cod_index + 1),
                                                  dict_op_write[int(codon[cod_index + 1 + 1])],
                                                  ret_op_write(codon_aux, x, cod_index + 1 + 2))
            else:
                raise ValueError('Cadena no compatible con la gramática')

        elif cod_tmp == '1':
            if (cod_index + 1 + 2) < len(codon) and (int(codon[cod_index + 1 + 1]) < 4):
                codon_aux = codon[expr_write[int(cod_tmp)][1]:]
                exp = expr_write[int(cod_tmp)][0](ret_op_write(codon_aux, x, cod_index + 1),
                                                  dict_op_write[int(codon[cod_index + 1 + 1])],
                                                  ret_op_write(codon_aux, x, cod_index + 1 + 2))
            else:
                raise ValueError('Cadena no compatible con la gramática')
        elif cod_tmp == '2':
            if (cod_index + 1 + 1) < len(codon) and (int(codon[cod_index + 1]) < 4):
                codon_aux = codon[expr_write[int(cod_tmp)][1]:]
                exp = expr_write[int(cod_tmp)][0](dic_pre_op_write[int(codon[cod_index + 1])],
                                                  ret_op_write(codon_aux, x, cod_index + 1 + 1))
            else:
                raise ValueError('Cadena no compatible con la gramática')
        elif cod_tmp == '3':
            if int(codon[cod_index + 1]) < 2:
                # print(' imprimo x - write')
                exp = dict_var_write[int(codon[cod_index + 1])]
            elif int(codon[cod_index + 1]) < 4:
                exp = dict_var_write[int(codon[cod_index + 1])]
            else:
                raise ValueError('Cadena no compatible con la gramática')

        return exp
    else:
        raise ValueError('Cadena no compatible con la gramática')


def genera_poblacion(N_POB, min=0, max=4, longitud=60, contador_max=100000, diversidad_inicial = 5):
    poblacion = []
    expresiones = []
    exc = 0
    contador = 0
    while len(poblacion) < N_POB:
        contador += 1
        tmp = genera_codon(min=min, max=max, longitud=longitud)
        try:
            # print(ret_op_write(tmp, x='x'))
            ret_op(tmp, 0)
            if contador < contador_max:
                if sum([ret_op_write(tmp, x='x') == exp for exp in expresiones]) < diversidad_inicial:
                    expresiones.append(ret_op_write(tmp, x='x'))
                    poblacion.append(tmp)
            else:
                poblacion.append(tmp)
                expresiones.append(ret_op_write(tmp, x='x'))
        except:
            exc += 1
    print(' Cadenas excluidas: ' + str(exc + 1))
    return poblacion, expresiones


def busc_local(df_gen, poblacion, dx, N, A, f_values, N_best = 10):
    id = df_gen.head(N_best).index.values
    poblacion = np.array(poblacion)
    best = np.unique(poblacion[id])
    new = []
    for i, p1 in enumerate(best):
        for j, p2 in enumerate(best):
            if p1 != p2 and i < j:
                hij1, hij2 = two_poin_cross(p1, p2)
                new.append(hij1)
                new.append(hij2)
    id_worst = df_gen.tail(len(new)).index.values
    poblacion[id_worst] = new
    df_gen_2 = evalua(poblacion, dx, N, A)
    df_gen_2['fitness'] = df_gen_2.apply(error, axis=1, args=(f_values,))
    df_gen_2 = df_gen_2.sort_values('fitness')
    return list(poblacion), df_gen_2

def f_en1(x):
    return x**3 + 8
def df_en1(x):
    return 3*(x**2)

def f_en2(x):
    return (x-2)/(x+2)
def df_en2(x):
    return 4/(x+2)**2

def f_en3(x):
    return 0.2*(x**2+1)*(x-1)
def df_en3(x):
    return 0.2*(3*x**2 - 2*x +1)

def f_en4(x):
    return -np.exp(-2*x**2+2)
def df_en4(x):
    return 4*x*np.exp(-2*x**2+2)

def f_en5(x):
    return (np.exp(2*x)+np.exp(-6*x))/2
def df_en5(x):
    return np.exp(2*x)-3*np.exp(-6*x)

def f_en6(x):
    return x*np.log(1+2*x)
def df_en6(x):
    return np.log(1+2*x)+(2*x)/(1+2*x)

def f_en7(x):
    return np.exp(2*x)*np.sin(x)
def df_en7(x):
    return np.exp(2*x)*(2*np.sin(x)+np.cos(x))

funciones = [f_en1, f_en2, f_en3, f_en4, f_en5, f_en6, f_en7]
derivadas = [df_en1, df_en2, df_en3, df_en4, df_en5, df_en6, df_en7]
limites = [(0,5), (0,5), (-2,2), (0,3), (0,2), (0,5), (-2,2)]
