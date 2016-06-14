import numpy as np
import math
import sys

def quantize(signal, partitions, codebook):
    indices = []
    quanta = []
    for datum in signal:
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        indices.append(index)
        quanta.append(codebook[index])
    return indices, quanta


def pattern_mat(x, m):
    """
    Construct a matrix of `m`-length segments of `x`.
    Parameters
    ----------
    x : (N, ) array_like
        Array of input data.
    m : int
        Length of segment. Must be at least 1. In the case that `m` is 1, the
        input array is returned.
    Returns
    -------
    patterns : (m, N-m+1)
        Matrix whose first column is the first `m` elements of `x`, the second
        column is `x[1:m+1]`, etc.
    Examples
    --------
    > p = pattern_mat([1, 2, 3, 4, 5, 6, 7], 3])
    array([[ 1.,  2.,  3.,  4.,  5.],
           [ 2.,  3.,  4.,  5.,  6.],
           [ 3.,  4.,  5.,  6.,  7.]])
    """
    x = np.asarray(x).ravel()
    if m == 1:
        return x
    else:
        N = len(x)
        patterns = np.zeros((m, N - m + 1))
        for i in range(m):
            patterns[i, :] = x[i:N - m + i + 1]
        return patterns


def en_shannon(series, L, num_int):
    if not series:
        raise ValueError("No hay serie definida")
    if not L:
        raise ValueError("No hay dimension (L) definida")
    if not num_int:
        raise ValueError("num_int sin definir")
    # Normalizacion
    series = (series - np.mean(series)) / np.std(series)
    # We the values of the parameters required for the quantification:
    epsilon = (max(series) - min(series)) / num_int
    partition = np.arange(min(series), math.ceil(max(series)), epsilon)
    codebook = np.arange(-1, num_int + 1)
    # Uniform quantification of the time series:
    _, quants = quantize(series, partition, codebook)
    # The minimum value of the signal quantified assert passes -1 to 0:
    quants = [0 if x == -1 else x for x in quants]
    N = len(quants)
    # We compose the patterns of length 'L':
    X = pattern_mat(quants, L)
    # We get the number of repetitions of each pattern:
    num = np.ones(N - L + 1)
    # This loop goes over the columns of 'X':
    if L == 1:
        X = np.atleast_2d(X)
    for j in range(0, N - L + 1):
        for i2 in range(j + 1, N - L + 1):
            tmp = [0 if x == -1 else 1 for x in X[:, j]]
            if (tmp[0] == 1) and (X[:, j] == X[:, i2]).all():
                num[j] += 1
                X[:, i2] = -1
            tmp = -1

    # We get those patterns which are not NaN:
    aux = [0 if x == -1 else 1 for x in X[0, :]]
    # Now, we can compute the number of different patterns:
    new_num = []
    for j, a in enumerate(aux):
        if a != 0:
            new_num.append(num[j])
    new_num = np.asarray(new_num)

    # We get the number of patterns which have appeared only once:
    unique = sum(new_num[new_num == 1])
    # We compute the probability of each pattern:
    p_i = new_num / (N - L + 1)
    # Finally, the Shannon Entropy is computed as:
    SE = np.dot((- 1) * p_i, (np.log(p_i)))

    return SE, unique


def cond_en(series, L, num_int):
    if not series:
        raise ValueError("No hay serie definida")
    if not L:
        raise ValueError("No hay dimension (L) definida")
    if not num_int:
        raise ValueError("num_int sin definir")
    # Processing:
    # First, we call the Shannon Entropy function:
    # 'L' as embedding dimension:
    SE, unique = en_shannon(series, L, num_int)
    # 'L-1' as embedding dimension:
    SE_1, _ = en_shannon(series, L - 1, num_int)
    # The Conditional Entropy is defined as a differential entropy:
    CE = SE - SE_1
    return CE, unique


def correc_cond_en(series, Lmax, num_int):
    if not series:
        raise ValueError("No hay serie definida")
    if not Lmax:
        raise ValueError("No hay dimension (L) definida")
    if not num_int:
        raise ValueError("num_int sin definir")
    N = len(series)
    # We will use this for the correction term: (L=1)
    E_est_1, _ = en_shannon(series, 1, num_int)
    # Incializacin de la primera posicin del vector que almacena la CCE a un
    # numero elevado para evitar que se salga del bucle en L=2 (primera
    # iteracin):
    # CCE is a vector that will contian the several CCE values computed:
    CCE = sys.maxsize * np.ones(Lmax+1)
    CCE[0] = 100
    CE = np.ones(Lmax+1)
    uniques = np.ones(Lmax+1)
    correc_term = np.ones(Lmax+1)
    for L in range(2, Lmax+1):
        # First, we compute the CE for the current embedding dimension: ('L')
        CE[L], uniques[L] = cond_en(series, L, num_int)
        # Second, we compute the percentage of patterns which are not repeated:
        perc_L = uniques[L] / (N - L + 1)
        correc_term[L] = perc_L * E_est_1
        # Third, the CCE is the CE plus the correction term:
        CCE[L] = CE[L] + correc_term[L]

    # Finally, the best estimation of the CCE is the minimum value of all the
    # CCE that have been computed:
    CCE_min = min(CCE)
    return CCE_min
