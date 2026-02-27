#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Cluster Stability
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Calculating stability from bootstraped samples for each clustering algorithm.

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

#load data
methods = ['hierarchical', 'kmeans', 'spectral']
versions = ['v1', 'v2', 'v3', 'v4', 'v5']
waves = ['t0', 't2']
imputes = ['all', 'single', 'only', 'rt']

bootstraped_models = {}

for method in methods:
    for version in versions:
        for wave in waves:
            for impute in imputes:
                imp_key = impute.lower().replace('_impute', '')
                filename = f"{method}_stability_{wave}_{version}_{imp_key}.csv"
                path = import_directory / filename
                key = f"{method}_stability_{wave}_{version}_{imp_key}"
                bootstraped_models[key] = pd.read_csv(path)
                bootstraped_models[key] = (bootstraped_models[key].sort_values("src_subject_id").reset_index(drop=True))

"""
Functions
"""    
def bootstrap_version_mean(dfs, start_boot=2, end_boot=15, id_col='src_subject_id'):
    result = dfs[0][[id_col]].copy()
    for i in range(start_boot, end_boot + 1):
        cols_i = [col for col in dfs[0].columns if f"_{i}_" in col]
        for col in cols_i:
            result[col] = sum(df[col] for df in dfs) / len(dfs)
    return result
   
"""
Average Versions
"""   
models = ["kmeans", "hierarchical", "spectral"]
waves = ["t0", "t2"]
versions = ["all", "only", "rt", "single"]

bootstrapped_avg = {}

for model in models:
    for wave in waves:
        for version in versions:
            dfs = [
                df for key, df in bootstraped_models.items()
                if (model in key) and (wave in key) and (version in key)]
            if len(dfs) == 0:
                continue
            avg_df = bootstrap_version_mean(dfs)
            out_key = f"{model}_{wave}_{version}"
            bootstrapped_avg[out_key] = avg_df
            
"""
Average Instability by Model
"""
instability = {}

for key, df in bootstrapped_avg.items():
    instability_df = (df.iloc[:, 1:].mean().to_frame().reset_index())
    instability_df.columns = (instability_df.columns.astype(str).str.replace("index", "model").str.replace("0", "instability"))
    instability[key] = instability_df

"""
Export Files
"""
for key, df in instability.items():
    filename = f"{key}_instability.csv"
    df.to_csv(export_directory / filename, index=False)









