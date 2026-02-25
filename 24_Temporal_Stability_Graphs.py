#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Cluster Temporal Stability Graphs
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Graphing the proportion of participants in the same relative cluster across timepoints (i.e., baseline and 2-year follow-up).

Dataset(s):
- abcd_all.csv

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
import os
import matplotlib.pyplot as plt
import seaborn as sns

#load data
abcd_all = pd.read_csv(import_directory/'abcd_all.csv')
all_t0_high_t2_high_ids = pd.read_csv(export_directory/'IDs'/'all_tasks_t0_km_2_t2_t0_0_t2_0_ids.csv')
all_t0_high_t2_low_ids = pd.read_csv(export_directory/'IDs'/'all_tasks_t0_km_2_t2_t0_0_t2_1_ids.csv')
all_t0_low_t2_high_ids = pd.read_csv(export_directory/'IDs'/'all_tasks_t0_km_2_t2_t0_1_t2_0_ids.csv')
all_t0_low_t2_low_ids = pd.read_csv(export_directory/'IDs'/'all_tasks_t0_km_2_t2_t0_1_t2_1_ids.csv')

"""
Prepare Data
"""
ids_11 = set(all_t0_high_t2_high_ids['src_subject_id'])   # always high EF
ids_10 = set(all_t0_high_t2_low_ids['src_subject_id'])    # high → low
ids_01 = set(all_t0_low_t2_high_ids['src_subject_id'])    # low → high
ids_00 = set(all_t0_low_t2_low_ids['src_subject_id'])     # always low

abcd_all['cluster_stability'] = (
    abcd_all['src_subject_id'].isin(ids_11) * 1 +
    abcd_all['src_subject_id'].isin(ids_10) * 2 +
    abcd_all['src_subject_id'].isin(ids_01) * 3 +
    abcd_all['src_subject_id'].isin(ids_00) * 4
)

abcd_all['cluster_stability'] = abcd_all['cluster_stability'].replace(0, np.nan)

"""
Cluster Graphs
"""
# T0 Clusters and T0 SST PCs
abcd_all['cluster_label'] = abcd_all['tasks_t0_km_2'].map({0: "Higher EF", 1: "Lower EF"})
palette = {"Higher EF": "black", "Lower EF": "purple"}
plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=abcd_all,
    x='sst_t0_pc1',
    y='sst_t0_pc2',
    hue='cluster_label',
    palette=palette,
    s=60,          
    edgecolor='white',
    linewidth=0.5,
    alpha=0.75)
plt.title("Baseline SST PC1 and PC2 by Baseline Cluster Assignment", fontsize=18)
plt.xlabel("SST PC1", fontsize=14)
plt.ylabel("SST PC2", fontsize=14)
plt.legend(title="Clusters", title_fontsize=12, fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()

# T0 Clusters and T0 NBack PCs
abcd_all['cluster_label'] = abcd_all['tasks_t0_km_2'].map({0: "Higher EF", 1: "Lower EF"})
palette = {"Higher EF": "black", "Lower EF": "purple"}
plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=abcd_all,
    x='nback_t0_pc1',
    y='nback_t0_pc2',
    hue='cluster_label',
    palette=palette,
    s=60,          
    edgecolor='white',
    linewidth=0.5,
    alpha=0.75)
plt.title("Baseline EN-Back PC1 and PC2 by Baseline Cluster Assignment", fontsize=18)
plt.xlabel("EN-Back PC1", fontsize=14)
plt.ylabel("EN-Back PC2", fontsize=14)
plt.legend(title="Clusters", title_fontsize=12, fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()

# T2 Clusters and T0 SST PCs
abcd_all['cluster_label'] = abcd_all['tasks_t2_km_2'].map({0: "Higher EF", 1: "Lower EF"})
palette = {"Higher EF": "black", "Lower EF": "purple"}
plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=abcd_all,
    x='sst_t0_pc1',
    y='sst_t0_pc2',
    hue='cluster_label',
    palette=palette,
    s=60,          
    edgecolor='white',
    linewidth=0.5,
    alpha=0.75)
plt.title("Baseline SST PC1 and PC2 by 2-year Cluster Assignment", fontsize=18)
plt.xlabel("SST PC1", fontsize=14)
plt.ylabel("SST PC2", fontsize=14)
plt.legend(title="Clusters", title_fontsize=12, fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()

# T2 Clusters and T0 NBack PCs
abcd_all['cluster_label'] = abcd_all['tasks_t2_km_2'].map({0: "Higher EF", 1: "Lower EF"})
palette = {"Higher EF": "black", "Lower EF": "purple"}
plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=abcd_all,
    x='nback_t0_pc1',
    y='nback_t0_pc2',
    hue='cluster_label',
    palette=palette,
    s=60,          
    edgecolor='white',
    linewidth=0.5,
    alpha=0.75)
plt.title("Baseline EN-Back PC1 and PC2 by 2-year Cluster Assignment", fontsize=18)
plt.xlabel("EN-Back PC1", fontsize=14)
plt.ylabel("EN-Back PC2", fontsize=14)
plt.legend(title="Clusters", title_fontsize=12, fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()

# T0 Clusters and T2 SST PCs
abcd_all['cluster_label'] = abcd_all['tasks_t0_km_2'].map({0: "Higher EF", 1: "Lower EF"})
palette = {"Higher EF": "black", "Lower EF": "purple"}
plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=abcd_all,
    x='sst_t2_pc1',
    y='sst_t2_pc2',
    hue='cluster_label',
    palette=palette,
    s=60,          
    edgecolor='white',
    linewidth=0.5,
    alpha=0.75)
plt.title("2-year SST PC1 and PC2 by Baseline Cluster Assignment", fontsize=18)
plt.xlabel("SST PC1", fontsize=14)
plt.ylabel("SST PC2", fontsize=14)
plt.legend(title="Clusters", title_fontsize=12, fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()

# T0 Clusters and T2 NBack PCs
abcd_all['cluster_label'] = abcd_all['tasks_t0_km_2'].map({0: "Higher EF", 1: "Lower EF"})
palette = {"Higher EF": "black", "Lower EF": "purple"}
plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=abcd_all,
    x='nback_t2_pc1',
    y='nback_t2_pc2',
    hue='cluster_label',
    palette=palette,
    s=60,          
    edgecolor='white',
    linewidth=0.5,
    alpha=0.75)
plt.title("2-year EN-Back PC1 and PC2 by Baseline Cluster Assignment", fontsize=18)
plt.xlabel("EN-Back PC1", fontsize=14)
plt.ylabel("EN-Back PC2", fontsize=14)
plt.legend(title="Clusters", title_fontsize=12, fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()

# T2 Clusters and T2 SST PCs
abcd_all['cluster_label'] = abcd_all['tasks_t2_km_2'].map({0: "Higher EF", 1: "Lower EF"})
palette = {"Higher EF": "black", "Lower EF": "purple"}
plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=abcd_all,
    x='sst_t2_pc1',
    y='sst_t2_pc2',
    hue='cluster_label',
    palette=palette,
    s=60,          
    edgecolor='white',
    linewidth=0.5,
    alpha=0.75)
plt.title("2-year SST PC1 and PC2 by 2-year Cluster Assignment", fontsize=18)
plt.xlabel("SST PC1", fontsize=14)
plt.ylabel("SST PC2", fontsize=14)
plt.legend(title="Clusters", title_fontsize=12, fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()

# T2 Clusters and T2 NBack PCs
abcd_all['cluster_label'] = abcd_all['tasks_t2_km_2'].map({0: "Higher EF", 1: "Lower EF"})
palette = {"Higher EF": "black", "Lower EF": "purple"}
plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=abcd_all,
    x='nback_t2_pc1',
    y='nback_t2_pc2',
    hue='cluster_label',
    palette=palette,
    s=60,          
    edgecolor='white',
    linewidth=0.5,
    alpha=0.75)
plt.title("2-year EN-Back PC1 and PC2 by 2-year Cluster Assignment", fontsize=18)
plt.xlabel("EN-Back PC1", fontsize=14)
plt.ylabel("EN-Back PC2", fontsize=14)
plt.legend(title="Clusters", title_fontsize=12, fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()

"""
Stability Graphs
"""
#SST T0 PCs
palette = { 1: 'green', 2: 'black', 3: 'blue', 4: 'red'}
label_map = { 1: "Always High EF", 2: "High EF to Low EF", 3: "Low EF to High EF", 4: "Always Low EF"}
plt.figure(figsize=(12, 9))
for profile, color in palette.items():
    subset = abcd_all[abcd_all['cluster_stability'] == profile]
    plt.scatter(
        subset['sst_t0_pc1'],
        subset['sst_t0_pc2'],
        s=60,                  
        alpha=0.85,
        color=color,
        label=label_map[profile],   
        edgecolors='white',     
        linewidth=0.5)
plt.title("Baseline SST PC1 and PC2 by Temporal Stability Profiles", fontsize=18)
plt.xlabel("SST PC1", fontsize=14)
plt.ylabel("SST PC2", fontsize=14)
plt.legend(title="Profiles", title_fontsize=12, fontsize=11, loc='lower right')
plt.tight_layout()
plt.show()

#SST T2 PCs
palette = {1: 'green', 2: 'black', 3: 'blue', 4: 'red'}
label_map = {1: "Always High EF", 2: "High EF to Low EF", 3: "Low EF to High EF", 4: "Always Low EF"}
plt.figure(figsize=(12, 9))
for profile, color in palette.items():
    subset = abcd_all[abcd_all['cluster_stability'] == profile]
    plt.scatter(
        subset['sst_t2_pc1'],
        subset['sst_t2_pc2'],
        s=60,                  
        alpha=0.85,
        color=color,
        label=label_map[profile],   
        edgecolors='white',     
        linewidth=0.5)
plt.title("2-year Follow-up SST PC1 and PC2 by Temporal Stability Profiles", fontsize=18)
plt.xlabel("SST PC1", fontsize=14)
plt.ylabel("SST PC2", fontsize=14)
plt.legend(title="Profiles", title_fontsize=12, fontsize=11, loc='lower right')
plt.tight_layout()
plt.show()

#NBack T0 PCs
palette = {1: 'green', 2: 'black', 3: 'blue', 4: 'red'}
label_map = {1: "Always High EF", 2: "High EF to Low EF", 3: "Low EF to High EF", 4: "Always Low EF"}
plt.figure(figsize=(12, 9))
for profile, color in palette.items():
    subset = abcd_all[abcd_all['cluster_stability'] == profile]
    plt.scatter(
        subset['nback_t0_pc1'],
        subset['nback_t0_pc2'],
        s=60,                  
        alpha=0.85,
        color=color,
        label=label_map[profile],   
        edgecolors='white',     
        linewidth=0.5)
plt.title("Baseline EN-Back PC1 and PC2 by Temporal Stability Profiles", fontsize=18)
plt.xlabel("EN-Back PC1", fontsize=14)
plt.ylabel("EN-Back PC2", fontsize=14)
plt.legend(title="Profiles", title_fontsize=12, fontsize=11, loc='lower right')
plt.tight_layout()
plt.show()

#NBack T2 PCs
palette = {1: 'green', 2: 'black', 3: 'blue', 4: 'red'}
label_map = {1: "Always High EF", 2: "High EF to Low EF", 3: "Low EF to High EF", 4: "Always Low EF"}
plt.figure(figsize=(12, 9))
for profile, color in palette.items():
    subset = abcd_all[abcd_all['cluster_stability'] == profile]
    plt.scatter(
        subset['nback_t2_pc1'],
        subset['nback_t2_pc2'],
        s=60,                  
        alpha=0.85,
        color=color,
        label=label_map[profile],   
        edgecolors='white',     
        linewidth=0.5)
plt.title("2-year Follow-up EN-Back  PC1 and PC2 by Temporal Stability Profiles", fontsize=18)
plt.xlabel("EN-Back PC1", fontsize=14)
plt.ylabel("EN-Back PC2", fontsize=14)
plt.legend(title="Profiles", title_fontsize=12, fontsize=11, loc='lower right')
plt.tight_layout()
plt.show()
























