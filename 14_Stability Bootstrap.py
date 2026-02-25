#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: K-Means Stability Bootstrap Batch 1
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Identifying stability of hyper-parameter combinations for K-Means, Hierarchical Agglomerative, and Spectral Clustering.

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4
- sklearn version: 1.6.1

Notes:
- Written for NIH HPC Biowulf
- Bootstraps in batches of 200. 
"""
print("Hello!")

#core
import pandas as pd
import numpy as np
import argparse

#statistical suite
from sklearn.utils import resample
from itertools import combinations
from sklearn.cluster import KMeans

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Processes PCA files")
    parser.add_argument('--tasks_t0', type=str, required=True,
                        help="Path Tasks T0")
    parser.add_argument('--tasks_t2', type=str, required=True,
                        help="Path to Tasks T2")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    path_tasks_t0 = args.tasks_t0
    tasks_t0 = pd.read_csv(
        path_tasks_t0, sep=",")

    path_tasks_t2 = args.tasks_t2
    tasks_t2 = pd.read_csv(
        path_tasks_t2, sep=",")

    del path_tasks_t0
    del path_tasks_t2
    
    print("Data Loading Complete")

    """
    Keep Components at 90% Threshold
    """
    # sort dataframes
    tasks_t0.sort_values(
        by='src_subject_id', inplace=True)
    tasks_t2.sort_values(
        by='src_subject_id', inplace=True)

    tasks_t0.reset_index(
        drop=True, inplace=True)
    tasks_t2.reset_index(
        drop=True, inplace=True)

    print("Data Sorting Complete")
    
    # ids
    kmeans_stability_t0 = pd.DataFrame(
        tasks_t0['src_subject_id'])
    kmeans_stability_t2 = pd.DataFrame(
        tasks_t2['src_subject_id'])

    """
    Functions
    (1) randomly bootstraps data with replacement, using a pre-determined and different seed for each iteration, 
    (2) fits algorithm function to each bootstrapped sample,
    (3) saves labels with subject ids as a dataframe within a nested data dictionary 
    """
    def kmeans_bootstrap(dataframe, seed_range, km_clusters, name_cols):
        km_bootstrap = {}
        column = [
            col for col in dataframe.columns if col.startswith(name_cols)]
        for i, set_seed in enumerate(seed_range, 1):
            bootstrap_i = resample(
                dataframe, random_state=set_seed, n_samples=len(dataframe), replace=True)
            kmeans = KMeans(
                n_clusters=km_clusters,
                init='random',
                n_init=10,
                random_state=set_seed
            )
            labels = kmeans.fit_predict(
                bootstrap_i[column])
            result = bootstrap_i[[
                'src_subject_id']].copy()
            result[f'bootstrap{i}'] = labels
            km_bootstrap[f'km_{km_clusters}_bootstrap{i}'] = result
        return km_bootstrap

    def merge_bootstraps(dictionary):
        for key in dictionary:
            dictionary[key] = dictionary[key].drop_duplicates(
            )
        df_merge = list(
            dictionary.values())[0]
        for key, dataframe in list(dictionary.items())[1:]:
            df_merge = pd.merge(df_merge, dataframe, on='src_subject_id',
                                how='outer', suffixes=('', f'_{key}'))
        return df_merge

    def similarity_value(i_val, j_val):
        similarity = (
            i_val == j_val) & (i_val != -1)
        if np.isnan(i_val) or np.isnan(j_val):
            similarity = np.nan
        return similarity

    def similarity_matrix(dataframe):
        nboots = dataframe.shape[1] - 1
        vector = np.zeros(
            (int(len(dataframe)*(len(dataframe)-1)/2), nboots))
        for nb in range(nboots):
            size = dataframe.iloc[:, nb+1]
            print(nb, end=",")
            result = [similarity_value(a, b) for a, b in combinations(
                dataframe['bootstrap' + str(nb+1)], r=2)]
            vector[:,
                   nb] = result
        similarity_vect = np.nanstd(
            vector, axis=1)
        similarity_mat = np.zeros(
            (len(size), len(size)))
        similarity_mat[np.triu_indices(
            len(size), k=1)] = similarity_vect
        similarity_mat += similarity_mat.T
        np.fill_diagonal(
            similarity_mat, np.nan)
        bootstrap_vect = np.count_nonzero(
            ~np.isnan(vector), axis=1)
        bootstrap_mat = np.zeros(
            (len(size), len(size)))
        bootstrap_mat[np.triu_indices(
            len(size), k=1)] = bootstrap_vect
        bootstrap_mat += bootstrap_mat.T
        np.fill_diagonal(
            bootstrap_mat, np.nan)

        similarity_mat = pd.DataFrame(
            similarity_mat, columns=None)
        bootstrap_mat = pd.DataFrame(
            bootstrap_mat, columns=None)
        return {'similarity_mat': similarity_mat, 'bootstrap_mat': bootstrap_mat}

    seed_range = range(0, 200)
    # seed = 843

    """
    K-Means
    """
    # tasks baseline
    name_cols = 'tasks_'

    tasks_t0_km_bootstrap = {}
    for cluster in range(2, 16):
        key = f'tasks_t0_km_{cluster}_bootstrap'
        tasks_t0_km_bootstrap[key] = kmeans_bootstrap(
            tasks_t0, seed_range, cluster, name_cols=name_cols)

    tasks_t0_km_bootstrap_merge = {}
    for outer_key, inner_dict in tasks_t0_km_bootstrap.items():
        tasks_t0_km_bootstrap_merge[outer_key] = merge_bootstraps(
            inner_dict)

    tasks_t0_km_similarity = {}
    
    print('Starting Similarities')
    
    for key, dataframe in tasks_t0_km_bootstrap_merge.items():
        tasks_t0_km_similarity[key] = similarity_matrix(
            dataframe)
        print('One Similarity Bootstrap Complete')
        
    print('Finished Similarities')
    
    for key, dictionary in tasks_t0_km_similarity.items():
        if 'similarity_mat' in dictionary:
            row_means = dictionary['similarity_mat'].mean(
                axis=1)
            kmeans_stability_t0[key] = row_means

    kmeans_stability_t0.to_csv(
        'kmeans_stability_t0_v1.csv', index=False)
    
    print('KMeans T0 Complete')
    
    del tasks_t0_km_bootstrap
    del tasks_t0_km_bootstrap_merge
    del tasks_t0_km_similarity
    
    # tasks 2-year follow-up
    tasks_t2_km_bootstrap = {}
    for cluster in range(2, 16):
        key = f'tasks_t2_km_{cluster}_bootstrap'
        tasks_t2_km_bootstrap[key] = kmeans_bootstrap(
            tasks_t2, seed_range, cluster, name_cols=name_cols)

    tasks_t2_km_bootstrap_merge = {}
    for outer_key, inner_dict in tasks_t2_km_bootstrap.items():
        tasks_t2_km_bootstrap_merge[outer_key] = merge_bootstraps(
            inner_dict)

    tasks_t2_km_similarity = {}
    for key, dataframe in tasks_t2_km_bootstrap_merge.items():
        tasks_t2_km_similarity[key] = similarity_matrix(
            dataframe)

    for key, dictionary in tasks_t2_km_similarity.items():
        if 'similarity_mat' in dictionary:
            row_means = dictionary['similarity_mat'].mean(
                axis=1)
            kmeans_stability_t2[key] = row_means
    
    kmeans_stability_t2.to_csv(
        'kmeans_stability_t2_v1.csv', index=False)
    
    print('KMeans T2 Complete')
    
    del tasks_t2_km_bootstrap
    del tasks_t2_km_bootstrap_merge
    del tasks_t2_km_similarity
    
if __name__ == "__main__":
    main()
