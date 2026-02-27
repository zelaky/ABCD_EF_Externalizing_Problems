#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Algorithm Hyper-parameter Tuning (including Sensitivity Analyses)
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Identifying optimal hyper-parameter combinations for K-Means, Hierarchical Agglomerative, DBSCAN, HDBSCAN, and Spectral Clustering.

Dataset(s):
- sst_t0_all_pca_90.csv
- nback_t0_all_pca_90.csv
- sst_t2_all_pca_90.csv
- nback_t2_all_pca_90.csv
- sst_t0_only_pca_90.csv
- nback_t0_only_pca_90.csv
- sst_t2_only_pca_90.csv
- nback_t2_only_pca_90.csv
- sst_t0_rt_pca_90.csv
- nback_t0_rt_pca_90.csv
- sst_t2_rt_pca_90.csv
- nback_t2_rt_pca_90.csv
- sst_t0_singleton_pca_90.csv
- nback_t0_singleton_pca_90.csv
- sst_t2_singleton_pca_90.csv
- nback_t2_singleton_pca_90.csv

Output(s):
- f"{name}_km_labels.csv"
- f"{name}_km_metrics.csv"
- f"{name}_hier_labels.csv"
- f"{name}_hier_metrics.csv"
- f"{name}_spect_labels.csv"
- f"{name}_spect_metrics.csv"

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4
- scipy version: 1.15.3
- sklearn version: 1.6.1
- matplotlib version: 3.10.0

Notes:
"""
#core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os 
import random 

#statistical suite
import scipy
from scipy.spatial.distance import pdist
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

#plotting
import matplotlib
import matplotlib.pyplot as plt

#load data
waves = ['t0', 't2']
imputations = ['all', 'only', 'rt', 'singleton']

task_pcs = {}

for wave in waves:
    for impute in imputations:
        sst_key = f'sst_{wave}_{impute}_pca_90'
        nback_key = f'nback_{wave}_{impute}_pca_90'
        sst_df = pd.read_csv(import_directory / f'{sst_key}.csv')
        nback_df = pd.read_csv(import_directory / f'{nback_key}.csv')
        
        merged_df = sst_df.merge(nback_df, on='src_subject_id', how='outer')
        merged_df = merged_df.sort_values('src_subject_id').reset_index(drop=True)

        col_rename = ['src_subject_id'] + [f'tasks_{i+1}' for i in range(len(merged_df.columns) - 1)]
        merged_df.columns = col_rename

        task_key = f'tasks_{wave}_{impute}'
        task_pcs[task_key] = merged_df

del system, working_directory, wave, waves, impute, imputations, sst_df, nback_df, merged_df

"""
Function(s)
"""
def kmeans_clustering(dataframe, clusters, cols, task):
    metrics = []
    km_labels = pd.DataFrame({'src_subject_id': dataframe['src_subject_id']})
    pcas = dataframe[cols]

    for cluster in clusters:
        kmeans = KMeans(
            n_clusters=cluster,
            init='random',
            n_init=10,
            random_state=843
        )
        labels = kmeans.fit_predict(pcas)
        silhouette = silhouette_score(pcas, labels)
        inertia = kmeans.inertia_
        calinski_harabasz = calinski_harabasz_score(pcas, labels)
        davies_bouldin = davies_bouldin_score(pcas, labels)

        km_labels[f"{task}_km_{cluster}"] = labels
        metrics.append({
            'task': task,
            'clusters': cluster,
            'silhouette': silhouette,
            'inertia': inertia,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin
        })
    km_metrics = pd.DataFrame(metrics)
    return km_labels, km_metrics

def hierarchical_clustering(dataframe, clusters, cols, task):
    metrics = []
    hier_labels = pd.DataFrame({'src_subject_id': dataframe['src_subject_id']})
    pcas = dataframe[cols]

    for cluster in clusters:
        hierarch = AgglomerativeClustering(n_clusters=cluster)
        labels = hierarch.fit_predict(pcas)
        silhouette = silhouette_score(pcas, labels)
        calinski_harabasz = calinski_harabasz_score(pcas, labels)
        davies_bouldin = davies_bouldin_score(pcas, labels)

        hier_labels[f"{task}_hier_{cluster}"] = labels
        metrics.append({
            'task': task,
            'clusters': cluster,
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin
        })

    hier_metrics = pd.DataFrame(metrics)
    return hier_labels, hier_metrics

def spectral_clustering(dataframe, cluster, components, cols, task):
    metrics = []
    spect_labels = pd.DataFrame({'src_subject_id': dataframe['src_subject_id']})
    pcas = dataframe[cols]

    for component in components:
        spect = SpectralClustering(
            n_clusters=cluster,
            n_components=component,
            assign_labels='cluster_qr',
            eigen_tol=4e-04,
            random_state=843
        )
        labels = spect.fit_predict(pcas)
        silhouette = silhouette_score(pcas, labels)
        calinski_harabasz = calinski_harabasz_score(pcas, labels)
        davies_bouldin = davies_bouldin_score(pcas, labels)

        spect_labels[f"{task}_spect_{component}"] = labels
        metrics.append({
            'task': task,
            'components': component,
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin
        })

    spect_metrics = pd.DataFrame(metrics)
    return spect_labels, spect_metrics

def hdbscan_clustering(dataframe, sizes, samples, cols, task):
    metrics = []
    hdb_labels = pd.DataFrame({'src_subject_id': dataframe['src_subject_id']})
    pcas = dataframe[cols]

    for size in sizes:
        for sample in samples:
            hdb = HDBSCAN(min_cluster_size=size, min_samples=sample)
            labels = hdb.fit_predict(pcas)
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                silhouette = silhouette_score(pcas, labels)
                calinski_harabasz = calinski_harabasz_score(pcas, labels)
                davies_bouldin = davies_bouldin_score(pcas, labels)
            else: #if one cluster (or noise), assign None or skip
                silhouette = None
                calinski_harabasz = None
                davies_bouldin = None

            hdb_labels[f"{task}_hdb_{size}_{sample}"] = labels
            metrics.append({
                'task': task,
                'size': size,
                'sample': sample,
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin
            })

    hdb_metrics = pd.DataFrame(metrics)
    return hdb_labels, hdb_metrics

def dbscan_clustering(dataframe, epsilons, clusters, cols, task):
    metrics = []
    db_labels = pd.DataFrame({'src_subject_id': dataframe['src_subject_id']})
    pcas = dataframe[cols]

    for epsilon in epsilons:
        for cluster in clusters:
            db = DBSCAN(eps=epsilon, min_samples=cluster)
            labels = db.fit_predict(pcas)
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                silhouette = silhouette_score(pcas, labels)
                calinski_harabasz = calinski_harabasz_score(pcas, labels)
                davies_bouldin = davies_bouldin_score(pcas, labels)
            else: #if one cluster (or noise), assign None or skip
                silhouette = None
                calinski_harabasz = None
                davies_bouldin = None

            db_labels[f"{task}_db_{epsilon:.2f}_{cluster}"] = labels
            metrics.append({
                'task': task,
                'epsilon': epsilon,
                'cluster': cluster,
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin
            })

    db_metrics = pd.DataFrame(metrics)
    return db_labels, db_metrics

"""
Random Data
"""
#dbscan 
np.random.seed(0)
X = np.random.rand(300, 2)
eps_range = np.arange(0.01, 0.2, 0.01)
min_range = range(4, 20)

fig, axes = plt.subplots(len(eps_range), len(min_range), figsize=(20, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = axes.ravel()

for i, epsilons in enumerate(eps_range):
    for j, samples in enumerate(min_range):
        db = DBSCAN(eps=epsilons, min_samples=samples)
        labels = db.fit_predict(X)
        ax = axes[i * len(min_range) + j]
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=25)
        ax.set_title(f'eps: ${epsilons:.2f}, samp: ${samples:.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()

#hdbscan
np.random.seed(0)  
X = np.random.rand(300, 2)

cluster_range = range(10, 100, 10)
min_range = range(1, 11)

fig, axes = plt.subplots(len(cluster_range), len(min_range), figsize=(20, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = axes.ravel()

for i, size in enumerate(cluster_range):
    for j, samples in enumerate(min_range):
        hdb = HDBSCAN(
            min_cluster_size=size,
            min_samples=samples
        )
        labels = hdb.fit_predict(X)
        ax = axes[i * len(min_range) + j]
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=25)
        ax.set_title(f'size: ${size:.2f}, samp: ${samples:.2f}')
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()

#spectral
np.random.seed(0)
X = np.random.rand(300, 2)
cluster_range = range(2, 11)
component_range = range(1, 11)

fig, axes = plt.subplots(len(cluster_range), len(component_range), figsize=(20, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = axes.ravel()

for i, cluster in enumerate(cluster_range):
    for j, component in enumerate(component_range):
        spect = SpectralClustering(
            n_clusters=cluster,
            n_components=component,
            assign_labels='cluster_qr',
            eigen_tol=4e-04,
            random_state=0
        )
        labels = spect.fit_predict(X)
        ax = axes[i * len(component_range) + j]
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=25)
        ax.set_title(f'clust: {cluster}, comp: {component}')
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()

del fig, ax, axes, X, i, j, 
del db, hdb, spect
del samples, labels, size, eps_range, epsilons, cluster, cluster_range, component, component_range, min_range

"""
Thresholds
"""
sample_10pct = 5501*0.10
sample_20pct = 5501*0.20
sample_90pct = 5501*0.90

cluster_range = range(2, 16)
component_range = range(2, 16)

"""
Tuning K-Means
"""
km_labels = {}
km_metrics = {}

for task_name, df in task_pcs.items():
    pca_cols = [col for col in df.columns if col.startswith('tasks_')]
    labels, metrics = kmeans_clustering(df, cluster_range, pca_cols, task_name)
    km_labels[task_name] = labels
    km_metrics[task_name] = metrics

for task_name, df in km_metrics.items():
    add_row = pd.DataFrame([{
        'task': task_name,
        'clusters': 1,
        'silhouette': np.nan,
        'inertia': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }])
    km_metrics[task_name] = pd.concat([add_row, df], ignore_index=True)

for task_name, df in km_labels.items():
    add_col = f"{task_name}_km_1"
    df[add_col] = 0

"""
Tuning Hierarchical
"""
hier_labels = {}
hier_metrics = {}

for task_name, df in task_pcs.items():
    pca_cols = [col for col in df.columns if col.startswith('tasks_')]
    labels, metrics = hierarchical_clustering(df, cluster_range, pca_cols, task_name)
    hier_labels[task_name] = labels
    hier_metrics[task_name] = metrics

for task_name, df in hier_metrics.items():
    add_row = pd.DataFrame([{
        'task': task_name,
        'clusters': 1,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }])
    hier_metrics[task_name] = pd.concat([add_row, df], ignore_index=True)

for task_name, df in hier_labels.items():
    add_col = f"{task_name}_hier_1"
    df[add_col] = 0
    
"""
Tuning Spectral
"""
spect_labels = {}
spect_metrics = {}

for task_name, df in task_pcs.items():
    if '_t0' in task_name:
        cluster_n = 25 
    elif '_t2' in task_name:
        cluster_n = 24
    else:
        raise ValueError(f"Unknown timepoint: {task_name}")
    pca_cols = [col for col in df.columns if col.startswith('tasks_')]
    labels, metrics = spectral_clustering(df, cluster_n, component_range, pca_cols, task_name)
    spect_labels[task_name] = labels
    spect_metrics[task_name] = metrics
    
for task_name, df in spect_metrics.items():
    add_row = pd.DataFrame([{
        'task': task_name,
        'components': 1,
        'silhouette': np.nan,
        'calinski_harabasz': np.nan,
        'davies_bouldin': np.nan
    }])
    spect_metrics[task_name] = pd.concat([add_row, df], ignore_index=True)

for task_name, df in spect_labels.items():
    add_col = f"{task_name}_spect_1"
    df[add_col] = 0
    
"""
Tuning DBSCAN
- set min_samples to 1 and identify epsilon where 90% of participants are unclustered (minimum values for min_samples and epsilon)
- maximum value of min_samples is 10% of sample size
- set min_samples to maximum value and identify the value of epsilon for which 90% of the participants are clustered in the same group 
- tune both hyper-parameter ranges using grid search to identify up to 15 models
"""
#baseline all
task_t0_all = task_pcs['tasks_t0_all']
pca_cols = [col for col in task_t0_all.columns if col.startswith('tasks_')]

#pairwise distances to estimate epsilon range
tasks_t0_pcs = task_t0_all.drop('src_subject_id', axis=1).values
tasks_t0_pair_dist = pdist(tasks_t0_pcs, metric='euclidean')
median_eps = np.median(tasks_t0_pair_dist) #8.945377273090982
min_eps = median_eps * 0.1 #0.8945377273090982

#set min_samples to 1 and identify epsilon where 90% of participants are unclustered (minimum values for min_samples and epsilon)
#notes: 
    #(1) no point when 90% are unclustered (-1), 
    #(2) more than 15 clusters or a several clusters with 1 participant and the rest in one large cluster
epsilon_range = np.arange(0.5, 15, 0.5)
cluster_n = [1]
task_name = 'tasks_t0_all'
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_all, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#setting minimum epsilon as 6.0 and min_samples to 10% of sample
#note: when epsilon is 8.0, 99.64% of the sample is in the same cluster, with the rest unclustered

epsilon_range = np.arange(6, 50, 2) #over 90% clustered in the same group at 8.0
cluster_n = [550] #10% of sample
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_all, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#setting both epsilon and cluster ranges
#note: 
    #(1) when min_samples is 1 no participant is unclustered and majority cluster becomes bigger as epsilon increases
    #(2) interaction: holding epsilon constant, an increase in min_samples is associated with more unclustered participants, 
    #but for each increase in epsilon, there is a lower number of unclustered participants 
epsilon_range = np.arange(6, 8, 0.5) 
cluster_range = np.arange(1, 80, 5) 
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_all, epsilon_range, cluster_range, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#2-year follow-up all
task_t2_all = task_pcs['tasks_t2_all']
pca_cols = [col for col in task_t2_all.columns if col.startswith('tasks_')]

#pairwise distances to estimate epsilon range
tasks_t2_pcs = task_t2_all.drop('src_subject_id', axis=1).values
tasks_t2_pair_dist = pdist(tasks_t2_pcs, metric='euclidean') #8.77238590005231
median_eps = np.median(tasks_t2_pair_dist) #0.8772385900052311
min_eps = median_eps * 0.1

#set min_samples to 1 and identify epsilon where 90% of participants are unclustered (minimum values for min_samples and epsilon)
#notes: 
    #(1) no point when 90% are unclustered (-1), 
    #(2) more than 15 clusters or a several clusters with 1 participant and the rest in one large cluster
epsilon_range = np.arange(0.5, 15, 0.5)
cluster_n = [1]
task_name = 'tasks_t2_all'
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t2_all, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#setting minimum epsilon as 6.0 and min_samples to 10% of sample
#note: when epsilon is 8.0, over 90% of the sample is in the same cluster, with the rest unclustered

epsilon_range = np.arange(6, 50, 2) #over 90% clustered in the same group at 8.0
cluster_n = [550] #10% of sample
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t2_all, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#setting both epsilon and cluster ranges
#note: 
    #(1) when min_samples is 1 no participant is unclustered and majority cluster becomes bigger as epsilon increases
    #(2) interaction: holding epsilon constant, an increase in min_samples is associated with more unclustered participants, 
    #but for each increase in epsilon, there is a lower number of unclustered participants 
epsilon_range = np.arange(6, 8, 0.5) 
cluster_range = np.arange(1, 80, 5) 
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t2_all, epsilon_range, cluster_range, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#baseline only
task_t0_only = task_pcs['tasks_t0_only']
pca_cols = [col for col in task_t0_only.columns if col.startswith('tasks_')]

tasks_t0_pcs = task_t0_only.drop('src_subject_id', axis=1).values
tasks_t0_pair_dist = pdist(tasks_t0_pcs, metric='euclidean')
median_eps = np.median(tasks_t0_pair_dist) 
min_eps = median_eps * 0.1 

epsilon_range = np.arange(0.5, 15, 0.5)
cluster_n = [1]
task_name = 'tasks_t0_only'
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_only, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

epsilon_range = np.arange(6, 50, 2) 
cluster_n = [550] 
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_only, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

epsilon_range = np.arange(6, 8, 0.5) 
cluster_range = np.arange(1, 80, 5) 
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_only, epsilon_range, cluster_range, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#2-year follow-up only
tasks_t2_only = task_pcs['tasks_t2_only']
pca_cols = [col for col in tasks_t2_only.columns if col.startswith('tasks_')]

tasks_t2_pcs = tasks_t2_only.drop('src_subject_id', axis=1).values
tasks_t2_pair_dist = pdist(tasks_t2_pcs, metric='euclidean')
median_eps = np.median(tasks_t2_pair_dist) 
min_eps = median_eps * 0.1

epsilon_range = np.arange(0.5, 15, 0.5)
cluster_n = [1]
task_name = 'tasks_t2_only'
dbscan_labels, dbscan_metrics = dbscan_clustering(tasks_t2_only, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

epsilon_range = np.arange(6, 50, 2) 
cluster_n = [550] 
dbscan_labels, dbscan_metrics = dbscan_clustering(tasks_t2_only, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

epsilon_range = np.arange(6, 8, 0.5) 
cluster_range = np.arange(1, 80, 5) 
dbscan_labels, dbscan_metrics = dbscan_clustering(tasks_t2_only, epsilon_range, cluster_range, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#baseline rt
task_t0_rt = task_pcs['tasks_t0_rt']
pca_cols = [col for col in task_t0_rt.columns if col.startswith('tasks_')]

tasks_t0_pcs = task_t0_rt.drop('src_subject_id', axis=1).values
tasks_t0_pair_dist = pdist(tasks_t0_pcs, metric='euclidean')
median_eps = np.median(tasks_t0_pair_dist) 
min_eps = median_eps * 0.1 

epsilon_range = np.arange(0.5, 15, 0.5)
cluster_n = [1]
task_name = 'tasks_t0_rt'
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_rt, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

epsilon_range = np.arange(6, 50, 2) 
cluster_n = [550] 
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_rt, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

epsilon_range = np.arange(6, 8, 0.5) 
cluster_range = np.arange(1, 80, 5) 
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_rt, epsilon_range, cluster_range, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#2-year follow-up rt
tasks_t2_rt = task_pcs['tasks_t2_rt']
pca_cols = [col for col in tasks_t2_rt.columns if col.startswith('tasks_')]

tasks_t2_pcs = tasks_t2_rt.drop('src_subject_id', axis=1).values
tasks_t2_pair_dist = pdist(tasks_t2_pcs, metric='euclidean')
median_eps = np.median(tasks_t2_pair_dist) 
min_eps = median_eps * 0.1

epsilon_range = np.arange(0.5, 15, 0.5)
cluster_n = [1]
task_name = 'tasks_t2_rt'
dbscan_labels, dbscan_metrics = dbscan_clustering(tasks_t2_rt, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

epsilon_range = np.arange(6, 50, 2) 
cluster_n = [550] 
dbscan_labels, dbscan_metrics = dbscan_clustering(tasks_t2_rt, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

epsilon_range = np.arange(6, 8, 0.5) 
cluster_range = np.arange(1, 80, 5) 
dbscan_labels, dbscan_metrics = dbscan_clustering(tasks_t2_rt, epsilon_range, cluster_range, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#baseline singleton
task_t0_singleton = task_pcs['tasks_t0_singleton']
pca_cols = [col for col in task_t0_singleton.columns if col.startswith('tasks_')]

#pairwise distances to estimate epsilon range
tasks_t0_pcs = task_t0_singleton.drop('src_subject_id', axis=1).values
tasks_t0_pair_dist = pdist(tasks_t0_pcs, metric='euclidean')
median_eps = np.median(tasks_t0_pair_dist) #8.977033401237724
min_eps = median_eps * 0.1 #0.8977033401237725

#set min_samples to 1 and identify epsilon where 90% of participants are unclustered (minimum values for min_samples and epsilon)
#notes: 
    #(1) no point when 90% are unclustered (-1), 
    #(2) more than 15 clusters or a several clusters with 1 participant and the rest in one large cluster
epsilon_range = np.arange(0.5, 15, 0.5)
cluster_n = [1]
task_name = 'tasks_t0_singleton'
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_singleton, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#setting minimum epsilon as 6.0 and min_samples to 10% of sample
#note: when epsilon is 8.0, 99.64% of the sample is in the same cluster, with the rest unclustered
epsilon_range = np.arange(6, 50, 2) #over 90% clustered in the same group at 8.0
cluster_n = [550] #10% of sample
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_singleton, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#setting both epsilon and cluster ranges
#note: 
    #(1) when min_samples is 1 no participant is unclustered and majority cluster becomes bigger as epsilon increases
    #(2) interaction: holding epsilon constant, an increase in min_samples is associated with more unclustered participants, 
    #but for each increase in epsilon, there is a lower number of unclustered participants 
epsilon_range = np.arange(6, 8, 0.5) 
cluster_range = np.arange(1, 80, 5) 
dbscan_labels, dbscan_metrics = dbscan_clustering(task_t0_singleton, epsilon_range, cluster_range, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#2-year follow-up singleton
tasks_t2_singleton = task_pcs['tasks_t2_singleton']
pca_cols = [col for col in tasks_t2_singleton.columns if col.startswith('tasks_')]

#pairwise distances to estimate epsilon range
tasks_t2_pcs = tasks_t2_singleton.drop('src_subject_id', axis=1).values
tasks_t2_pair_dist = pdist(tasks_t2_pcs, metric='euclidean') #8.77238590005231
median_eps = np.median(tasks_t2_pair_dist) #8.977033401237724
min_eps = median_eps * 0.1 #0.8977033401237725

#set min_samples to 1 and identify epsilon where 90% of participants are unclustered (minimum values for min_samples and epsilon)
#notes: 
    #(1) no point when 90% are unclustered (-1), 
    #(2) more than 15 clusters or a several clusters with 1 participant and the rest in one large cluster
epsilon_range = np.arange(0.5, 15, 0.5)
cluster_n = [1]
task_name = 'tasks_t0_singleton'
dbscan_labels, dbscan_metrics = dbscan_clustering(tasks_t2_singleton, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#setting minimum epsilon as 6.0 and min_samples to 10% of sample
#note: when epsilon is 8.0, over 90% of the sample is in the same cluster, with the rest unclustered
epsilon_range = np.arange(6, 50, 2) #over 90% clustered in the same group at 8.0
cluster_n = [550] #10% of sample
dbscan_labels, dbscan_metrics = dbscan_clustering(tasks_t2_singleton, epsilon_range, cluster_n, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#setting both epsilon and cluster ranges
#note: 
    #(1) when min_samples is 1 no participant is unclustered and majority cluster becomes bigger as epsilon increases
    #(2) interaction: holding epsilon constant, an increase in min_samples is associated with more unclustered participants, 
    #but for each increase in epsilon, there is a lower number of unclustered participants 
epsilon_range = np.arange(6, 8, 0.5) 
cluster_range = np.arange(1, 80, 5) 
dbscan_labels, dbscan_metrics = dbscan_clustering(tasks_t2_singleton, epsilon_range, cluster_range, pca_cols, task_name)
count_cols = dbscan_labels.filter(regex=rf'^{task_name}_db')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

"""
Tuning HDBSCAN
- set min_samples to 1 and tune min_cluster_size. 
- when the cluster number stops changing, this is maximum min_cluster_size value. 
- set min_cluster_size to maximum min_cluster_size value
- tune min_samples until 90% of the participants are unclustered (maximum value for min_samples).
"""
#baseline all
task_t0_all = task_pcs['tasks_t0_all']
pca_cols = [col for col in task_t0_all.columns if col.startswith('tasks_')]

#when sample_num is 1, cluster size stops changing at 67, but all have >90% unclustered
size_range = range(2, 100, 5) 
sample_num = [1]
task_name = 'tasks_t0_all'
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(task_t0_all, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_num = [30]
sample_range = range(2, 50) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(task_t0_all, size_num, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_range = range(2, 20, 1) 
sample_num = [2]
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(task_t0_all, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#note: increases in min_samples is associated with more unclustered participants in a repeating pattern 
size_range = range(2, 20, 1) 
sample_range = range(2, 4, 1) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(task_t0_all, size_range, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#2-year follow-up all
tasks_t2_all = task_pcs['tasks_t2_all']
pca_cols = [col for col in tasks_t2_all.columns if col.startswith('tasks_')]

#when sample_num is 1, cluster size stops changing at 67, but all have >90% unclustered
size_range = range(2, 100, 5) 
sample_num = [1]
task_name = 'tasks_t0_all'
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_all, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_num = [7]
sample_range = range(2, 20) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_all, size_num, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_range = range(2, 20, 1) 
sample_range = range(2, 4, 1) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_all, size_range, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#baseline only
tasks_t0_only = task_pcs['tasks_t0_only']
pca_cols = [col for col in tasks_t0_only.columns if col.startswith('tasks_')]

size_range = range(2, 100, 5) 
sample_num = [1]
task_name = 'tasks_t0_only'
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_only, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_num = [30]
sample_range = range(2, 50) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_only, size_num, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_range = range(2, 20, 1) 
sample_num = [2]
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_only, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_range = range(2, 20, 1) 
sample_range = range(2, 4, 1) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_only, size_range, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#2-year follow-up only
tasks_t2_only = task_pcs['tasks_t2_only']
pca_cols = [col for col in tasks_t2_only.columns if col.startswith('tasks_')]

size_range = range(2, 100, 5) 
sample_num = [1]
task_name = 'tasks_t2_only'
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_only, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_num = [7]
sample_range = range(2, 20) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_only, size_num, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_range = range(2, 20, 1) 
sample_range = range(2, 4, 1) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_only, size_range, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#baseline rt
tasks_t0_rt = task_pcs['tasks_t0_rt']
pca_cols = [col for col in tasks_t0_rt.columns if col.startswith('tasks_')]

size_range = range(2, 100, 5) 
sample_num = [1]
task_name = 'tasks_t0_rt'
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_rt, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_range = range(2, 20, 1) 
sample_num = [2]
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_rt, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#note: increases in min_samples is associated with more unclustered participants in a repeating pattern 
size_range = range(2, 20, 1) 
sample_range = range(2, 4, 1) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_rt, size_range, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#2-year follow-up rt
tasks_t2_rt = task_pcs['tasks_t2_rt']
pca_cols = [col for col in tasks_t2_rt.columns if col.startswith('tasks_')]

#when sample_num is 1, cluster size stops changing at 67, but all have >90% unclustered
size_range = range(2, 100, 5) 
sample_num = [1]
task_name = 'tasks_t2_rt'
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_rt, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_range = range(2, 20, 1) 
sample_range = range(2, 4, 1) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_rt, size_range, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#baseline singleton
tasks_t0_singleton = task_pcs['tasks_t0_singleton']
pca_cols = [col for col in tasks_t0_singleton.columns if col.startswith('tasks_')]

#when sample_num is 1, cluster size stops changing at 37, but all have >90% unclustered
size_range = range(2, 100, 5) 
sample_num = [1]
task_name = 'tasks_t0_singleton'
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_singleton, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_num = [30]
sample_range = range(2, 50) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_singleton, size_num, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_range = range(2, 20, 1) 
sample_num = [2]
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_singleton, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#note: increases in min_samples is associated with more unclustered participants in a repeating pattern 
size_range = range(2, 20, 1) 
sample_range = range(2, 4, 1) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t0_singleton, size_range, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

#2-year follow-up singleton
tasks_t2_singleton = task_pcs['tasks_t2_singleton']
pca_cols = [col for col in tasks_t2_singleton.columns if col.startswith('tasks_')]

#when sample_num is 1, cluster size stops changing at 67, but all have >90% unclustered
size_range = range(2, 100, 5) 
sample_num = [1]
task_name = 'tasks_t2_singleton'
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_singleton, size_range, sample_num, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_num = [7]
sample_range = range(2, 20) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_singleton, size_num, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

size_range = range(2, 20, 1) 
sample_range = range(2, 4, 1) 
hdbscan_labels, hdbscan_metrics = hdbscan_clustering(tasks_t2_singleton, size_range, sample_range, pca_cols, task_name)
count_cols = hdbscan_labels.filter(regex=rf'^{task_name}_hdb')
unique_counts = count_cols.apply(lambda x: x.value_counts()).fillna(0).astype(int)

"""
Export Files
"""
for name, df in km_labels.items():
    file_name = f"{name}_km_labels.csv"
    file_path = export_directory/file_name
    df.to_csv(file_path, index=False)  
    
for name, df in km_metrics.items():
    file_name = f"{name}_km_metrics.csv"
    file_path = export_directory/file_name
    df.to_csv(file_path, index=False) 
    
for name, df in hier_labels.items():
    file_name = f"{name}_hier_labels.csv"
    file_path = export_directory/file_name
    df.to_csv(file_path, index=False)  
    
for name, df in hier_metrics.items():
    file_name = f"{name}_hier_metrics.csv"
    file_path = export_directory/file_name
    df.to_csv(file_path, index=False)
    
for name, df in spect_labels.items():
    file_name = f"{name}_spect_labels.csv"
    file_path = export_directory/file_name
    df.to_csv(file_path, index=False)  
    
for name, df in spect_metrics.items():
    file_name = f"{name}_spect_metrics.csv"
    file_path = export_directory/file_name
    df.to_csv(file_path, index=False) 
    





