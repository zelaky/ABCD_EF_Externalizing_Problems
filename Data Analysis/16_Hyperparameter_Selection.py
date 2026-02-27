#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Cluster Label Counts
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Calculating counts of cluster labels for k-Means, Hierarchical, and Spectral clustering.

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
"""
#core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform 

methods = ['hier', 'km', 'spect']
waves = ['t0', 't2']
imputes = ['all', 'only', 'rt', 'singleton']

model_labels = {}

for method in methods: 
    for wave in waves: 
        for impute in imputes: 
            filename = f'tasks_{wave}_{impute}_{method}_labels.csv'
            path = import_directory / method / filename
            key = f'tasks_{wave}_{impute}_{method}_labels'
            model_labels[key] = pd.read_csv(path)

"""
Count Labels
"""
count_labels = {}

for key, df in model_labels.items():
    label_df = df.iloc[:, 1:]  # exclude ID column
    
    parts = key.split('_')  # e.g., ['tasks', 't0', 'all', 'km', 'labels']
    wave = parts[1]
    impute = parts[2]
    method = parts[3]

    cluster_rows = {}
    num_cols = label_df.shape[1]

    for i, col in enumerate(label_df.columns):
        # Calculate cluster count: 
        # if last col -> 1 cluster
        # else cluster count = i + 2
        cluster_count = 1 if i == num_cols - 1 else i + 2
        
        label_counts = label_df[col].value_counts().sort_index()
        row = {label: label_counts.get(label, np.nan) for label in range(16)}

        new_key = f"{method}{cluster_count}_{wave}_{impute}_counts"
        cluster_rows[new_key] = row

    count_labels[key] = pd.DataFrame.from_dict(cluster_rows, orient='index')

model_label_counts = pd.concat(count_labels.values(), axis=0)

#check exclusion criteria
threshold = 5501 * 0.10  # 550.1
small_max_cluster = model_label_counts.max(axis=1) <= threshold
small_max_cluster_rows = model_label_counts[small_max_cluster]

"""
Export Files
"""
model_label_counts.to_csv(export_directory/'model_label_counts.csv', index=True)










