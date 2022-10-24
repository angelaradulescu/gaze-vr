import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys
from itertools import combinations

UTILS_DIR = 'FC_geodesic/utils/distance_FC'
sys.path.insert(0, UTILS_DIR)
from distance_FC import distance_FC

def geodesic_distance(M1, M2):

    dist = distance_FC(M1, M2); 

    return dist.geodesic()

def pearson_distance(M1, M2):

    dist = distance_FC(M1, M2); 

    return dist.pearson()

def compute_null(reference_matrix, n_permutations=1000, distance='geodesic'):
    """ Computes a null distribution for geodesic distance to the reference matrix 
    by random permutation."""

    null_distribution = np.ones(n_permutations)*np.nan

    for p in np.arange(n_permutations):

        ## Permute matrix. 
        perm = np.random.permutation(reference_matrix.shape[0])
        permuted = reference_matrix[perm,:][:,perm]
        
        ## Get distance. 
        dist = distance_FC(reference_matrix, permuted)

        if distance == 'geodesic': null_distribution[p] = dist.geodesic()
        if distance == 'pearson': null_distribution[p] = dist.pearson()
        if distance == 'spearman': null_distribution[p] = dist.spearman()

    return null_distribution

def compute_average(matrices, distance='geodesic'):
    """ Computes the average pairwise distance between several matrices. Also returns the 2nd order
    similarity matrix (i.e. the matrix of pairwise distances between the matrices). Handles pairs of matrices 
    of different lengths by restricting to the minimum length."""

    matrix_pairs = list(combinations(np.arange(len(matrices)), 2))
    rdm = np.zeros((len(matrices), len(matrices)))

    pairwise_dist = np.ones(len(matrix_pairs))*np.nan

    ## Loop through matrix pairs.
    for i in np.arange(len(matrix_pairs)): 
    
        A, B = matrices[matrix_pairs[i][0]], matrices[matrix_pairs[i][1]]
        m_dims = [A.shape[0], B.shape[0]]

        if (A.shape[0] != B.shape[0]):
    
            min_dim = m_dims[np.where(m_dims == np.min(m_dims))[0][0]]

            dist = distance_FC(A[:min_dim, :min_dim], B[:min_dim, :min_dim])

        else: 

            dist = distance_FC(A, B)

        if distance == 'geodesic': 
            pairwise_dist[i] = dist.geodesic()
            rdm[matrix_pairs[i][0], matrix_pairs[i][1]] = dist.geodesic()
        if distance == 'pearson': 
            pairwise_dist[i] = dist.pearson()
            rdm[matrix_pairs[i][0], matrix_pairs[i][1]] = dist.pearson()

    return np.mean(pairwise_dist), rdm + rdm.T

def permute_matrices(matrices):

    matrices_permuted = []

    for m in matrices:

        perm = np.random.permutation(m.shape[0])
        m_permuted = m[perm,:][:,perm]
        matrices_permuted.append(m_permuted)

    return matrices_permuted

def compute_average_null(matrices, n_permutations=1000, distance='geodesic'):
    """ Uses random permutation to computes a null distribution for average geodesic distance 
    between several matrices."""

    average_null = np.ones(n_permutations)*np.nan

    for p in np.arange(n_permutations):

        matrices_permuted = permute_matrices(matrices)
        average_null[p], d = compute_average(matrices_permuted)

    return average_null







