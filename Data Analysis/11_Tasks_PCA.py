#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Principal Component Analysis (PCA) on Behavioral Tasks
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Applying PCA to baseline and 2-year follow-up SST and EFnBack tasks.

Input(s):
- demo_t0_split.csv
- sst_t0_all.csv
- nback_t0_all.csv
- sst_t2_all.csv
- nback_t2_all.csv
- sst_t0_only.csv
- nback_t0_only.csv
- sst_t2_only.csv
- nback_t2_only.csv
- sst_t0_rt.csv
- nback_t0_rt.csv
- sst_t2_rt.csv
- nback_t2_rt.csv

Output(s):
- f"{name}_pca_full.csv"
- f"{name}_pca_90.csv"

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4
- sklearn version: 1.6.1
- matplotlib version: 3.10.0
- seaborn version: 0.13.2

Notes:
"""
#core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os  

#analyses
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#plotting
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

#load data
file_list = [
    'sst_t0_all', 'nback_t0_all', 'sst_t2_all', 'nback_t2_all',
    'sst_t0_only', 'nback_t0_only', 'sst_t2_only', 'nback_t2_only',
    'sst_t0_rt', 'nback_t0_rt', 'sst_t2_rt', 'nback_t2_rt'
]

sst_nback_t0_t2 = {name: pd.read_csv(os.path.join(import_directory/'1_Task_Cleaning'/f"{name}.csv")) for name in file_list}
demo_t0_split = pd.read_csv(import_directory/'5_Split'/'demo_t0_split.csv')

del system, working_directory, file_list

"""
Function(s)
"""
scaler = StandardScaler()
def process_data(df):
    df_sorted = df.sort_values('src_subject_id').reset_index(drop=True)
    ids = df_sorted['src_subject_id']
    df_feat = df_sorted.drop(['src_subject_id', 'eventname'], axis=1)
    df_scale = scaler.fit_transform(df_feat)
    return df_scale, ids

component_thresholds_list = [] 
#set n_components as a percentage < 1.00 or as an integer indicating the number of desired components 
def fit_pca_plot(name, real_scaled, random_noise, gaussian_noise, n_components=None): 
    #fit models
    pca_real = PCA()
    pcs_real = pca_real.fit_transform(real_scaled)
    pca_random = PCA()
    pcs_random = pca_random.fit_transform(random_noise)
    pca_gaussian = PCA()
    pcs_gaussian = pca_gaussian.fit_transform(gaussian_noise)
    all_components = len(pca_real.explained_variance_)

    #determine number of components for scatter plot
    if isinstance(n_components, float) and 0 < n_components < 1:
        cumsum = np.cumsum(pca_real.explained_variance_ratio_)
        scatter_comps = np.argmax(cumsum >= n_components) + 1
    elif isinstance(n_components, int):
        scatter_comps = min(all_components, n_components)
    else:
        scatter_comps = all_components
    component_thresholds_list.append({'dataframe': name, 'components_90': scatter_comps})

    #scree plot
    x = np.arange(1, all_components + 1)
    bar_width = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width, pca_real.explained_variance_, width=bar_width, label='Scaled', alpha=0.7)
    plt.bar(x, pca_random.explained_variance_, width=bar_width, label='Random Noise', alpha=0.7)
    plt.bar(x + bar_width, pca_gaussian.explained_variance_, width=bar_width, label='Gaussian Noise', alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (Eigenvalues)')
    plt.title(f'Scree Plot Comparison - {name}')
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #cumulative variance plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, np.cumsum(pca_real.explained_variance_ratio_), marker='o', linestyle='--', color='blue', label='Scaled')
    plt.plot(x, np.cumsum(pca_random.explained_variance_ratio_), marker='o', linestyle='--', color='orange', label='Random Noise')
    plt.plot(x, np.cumsum(pca_gaussian.explained_variance_ratio_), marker='o', linestyle='--', color='green', label='Gaussian Noise')
    plt.axhline(y=0.95, color='red', linestyle='-', linewidth=1.5)
    plt.text(1, 0.96, '95% threshold', color='red', fontsize=12)
    plt.axhline(y=0.90, color='green', linestyle='-', linewidth=1.5)
    plt.text(1, 0.91, '90% threshold', color='green', fontsize=12)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Cumulative Variance Comparison - {name}')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='x')
    plt.show()

    #scatter matrix (real data PCs only)
    df_scatter = pd.DataFrame(pcs_real[:, :scatter_comps], columns=[f"PC{i+1}" for i in range(scatter_comps)])
    axes = pd.plotting.scatter_matrix(df_scatter, alpha=0.3, figsize=(30, 20), diagonal='kde')
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(f'Scatter Matrix of Top {scatter_comps} Components - {name}', fontsize=20)
    plt.show()

    return {
        'scaled': pcs_real,
        'random_noise': pcs_random,
        'gaussian_noise': pcs_gaussian
    }

"""
Prepare Dataframes
"""
sample_ids = demo_t0_split[demo_t0_split['sample_ids']==1]
keep_ids = set(sample_ids['src_subject_id'])

sst_nback_t0_t2 = {name: df[df['src_subject_id'].isin(keep_ids)].reset_index(drop=True)
    for name, df in sst_nback_t0_t2.items()
}

del sample_ids, keep_ids

tasks_scaled = {}
tasks_ids = {}
tasks_random_noise = {}
tasks_gaussian_noise = {}

#scale data
for name, df in sst_nback_t0_t2.items():
    scaled, ids = process_data(df)
    tasks_scaled[name] = scaled
    tasks_ids[name] = ids

#create random noise and gaussian noise dataframes
for name, scaled_df in tasks_scaled.items():
    n_samples, n_features = scaled_df.shape
    random = np.random.randint(0, 100, size=(n_samples, n_features))
    gaussian = np.random.normal(0, 1, size=(n_samples, n_features))
    tasks_random_noise[name] = random
    tasks_gaussian_noise[name] = gaussian

#scale random noise and gaussian noise dataframes
for name, array in tasks_random_noise.items():
    scaler = StandardScaler()
    tasks_random_noise[name] = scaler.fit_transform(array)
for name, array in tasks_gaussian_noise.items():
    scaler = StandardScaler()
    tasks_gaussian_noise[name] = scaler.fit_transform(array)

"""
Principal Component Analysis (PCA)
"""
tasks_pcs_scaled = {}
tasks_pcs_random_noise = {}
tasks_pcs_gaussian_noise = {}

for name in tasks_scaled:
    pcs = fit_pca_plot(
        name=name,
        real_scaled=tasks_scaled[name],
        random_noise=tasks_random_noise[name],
        gaussian_noise=tasks_gaussian_noise[name],
        n_components=0.90
    )
    
    tasks_pcs_scaled[name] = pcs['scaled']
    tasks_pcs_random_noise[name] = pcs['random_noise']
    tasks_pcs_gaussian_noise[name] = pcs['gaussian_noise']

component_thresholds = pd.DataFrame(component_thresholds_list)
pcs_threshold_lookup = dict(zip(component_thresholds['dataframe'], component_thresholds['components_90']))

tasks_pcs_scaled_full = {}

for name in tasks_pcs_scaled:
    scaled_array = tasks_pcs_scaled[name]
    id_df = tasks_ids[name].to_frame()
    pc_cols = [f"pc_{i+1}" for i in range(scaled_array.shape[1])]
    scaled_df = pd.DataFrame(scaled_array, columns=pc_cols)
    if name.startswith("sst_t0"):
        prefix = "sst_t0_"
    elif name.startswith("sst_t2"):
        prefix = "sst_t2_"
    elif name.startswith("nback_t0"):
        prefix = "nback_t0_"
    elif name.startswith("nback_t2"):
        prefix = "nback_t2_"
    else:
        continue
    rename_cols = {col: f"{prefix}{i+1}" for i, col in enumerate(pc_cols)}
    scaled_df = scaled_df.rename(columns=rename_cols)
    df_combined = pd.concat([id_df, scaled_df], axis=1)
    df_combined = df_combined[["src_subject_id"] + [col for col in df_combined.columns if col != "src_subject_id"]]
    tasks_pcs_scaled_full[name] = df_combined

tasks_pcs_scaled_90 = {}

for name, df in tasks_pcs_scaled_full.items():
    n_comps = pcs_threshold_lookup.get(name)
    if n_comps is None:
        continue
    cols = ["src_subject_id"] + [col for col in df.columns if col != "src_subject_id"][:n_comps]
    tasks_pcs_scaled_90[name] = df[cols]

"""
Export Files
"""
for name, df in tasks_pcs_scaled_full.items():
    file_name = f"{name}_pca_full.csv"
    file_path = export_directory / file_name
    df.to_csv(file_path, index=False)

for name, df in tasks_pcs_scaled_90.items():
    file_name = f"{name}_pca_90.csv"
    file_path = export_directory / file_name
    df.to_csv(file_path, index=False)
    
    
    