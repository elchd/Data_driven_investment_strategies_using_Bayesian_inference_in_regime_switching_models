import itertools
from itertools import chain
from collections import Counter
import pickle
import numpy as np 
import copy

def generate_combinations(K, n):
    elements = list(range(K))

    combi = itertools.combinations_with_replacement(elements, n)
    permu = itertools.product(elements, repeat=n)
    sorted_permu = map(sorted, copy.deepcopy(permu))
    combi_dict = Counter(map(tuple, sorted_permu))
    return np.array(list(combi)), np.array(list(permu)), combi_dict

def get_combinations(K, n): 
    """
    Generate combinations and permutations of elements.

    Parameters
    ----------
    K : int
        Number of elements to choose from.
    n : int
        Number of elements in each combination.

    Returns
    -------
    tuple
        A tuple containing three lists: 'combi', 'permu', and 'combi_dict'.

    Notes
    -----
    This function generates combinations and permutations of elements. 
    It returns a tuple containing the combinations, permutations, and a dictionary mapping permutations to combinations.

    """
    
    elements = list(range(K))
    combi = list(itertools.combinations_with_replacement(elements, n))
    
    permu = list(itertools.product(elements, repeat=n))
    sorted_permu = list(map(sorted, permu))
    combi_dict = Counter(map(tuple, sorted_permu))
    
    return(combi, permu, combi_dict)

def save_all_combi(K, N):
    """
    Generates and save all combinations and permutations for all values of K and
    n up to the given maximums.

    Parameters
    ----------
    max_k : int, optional
        The maximum number of distinct elements to choose from (inclusive), by
        default 10.
    max_n : int, optional
        The maximum length of combinations, by default 10.

    """
    
    """
    Generate and save combinations and permutations of elements for given parameters.

    Parameters
    ----------
    K : list
        List of integers representing different numbers of elements to choose from.
    N : list
        List of integers representing different numbers of elements in each combination.

    Notes
    -----
    This function generates combinations and permutations of elements for various values of 'K' and 'N'.
    It saves the results in a pickle file named 'all_combi.pkl'.

    """
    
    dict_k = {}
    for k in K:
        dict_n = {}
        for n in N:
            print(k, n)
            dict_n[n] = generate_combinations(k, (n+1))
        dict_k[k] = dict_n
        
    with open('all_combi.pkl', 'wb') as fp:
        pickle.dump(dict_k, fp)
        
#%%
#save_all_combi(K = [2, 3], N = np.arange(19))