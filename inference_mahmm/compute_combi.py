import itertools
from itertools import chain
from collections import Counter
import pickle
import numpy as np 
import copy

def generate_combinations(K, n):
    """
    Generate combinations and permutations of elements.

    Parameters
    ----------
    K : int
        The number of elements to choose from.
    n : int
        The number of elements in each combination.

    Returns
    -------
    numpy.ndarray
        The array of combinations (with replacement) of `K` elements taken `n` at a time.
    numpy.ndarray 
        Array of permutations (with repetition) of `K` elements taken `n` at a time.
    collections.Counter: 
        Dictionary mapping each unique combination to the number of times it appears in the permutations array, 
        without considering the order.

    """
    elements = list(range(K))
    combi = itertools.combinations_with_replacement(elements, n)
    permu = itertools.product(elements, repeat=n)
    sorted_permu = map(sorted, copy.deepcopy(permu))
    combi_dict = Counter(map(tuple, sorted_permu))
    return np.array(list(combi)), np.array(list(permu)), combi_dict

def save_all_combi(max_k, max_n, path=''):
    """
    Generate and save all combinations and permutations for all values of K and n up to the given maximums.

    Parameters
    ----------
    max_k : int
        The maximum number of distinct elements to choose from (inclusive).
    max_n : int
        The maximum length of combinations.
    path : str
        The path where the resulting dictionary should be saved (defaults is the current directory).

    """
    dict_k = {}
    for k in max_k:
        dict_n = {}
        for n in max_n:
            print(k, n)
            dict_n[n] = generate_combinations(k, (n+1))
        dict_k[k] = dict_n
        
    with open(path + 'all_combi.pkl', 'wb') as fp:
        pickle.dump(dict_k, fp)
        
