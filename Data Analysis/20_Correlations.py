#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Baseline Participant Comparisons
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Correlating all variables and timepoints of interest.

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4
- scipy version: 1.15.3

Notes:
- Pairwise deletion used.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os  

#statistical suite
import scipy 
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

#load data
sst_t0 = pd.read_csv(import_directory/'1_Task_Cleaning'/'sst_bissett_garavan_t0.csv')
nback_t0 = pd.read_csv(import_directory/'1_Task_Cleaning'/'nback_manual_t0.csv')
sst_t2 = pd.read_csv(import_directory/'1_Task_Cleaning'/'sst_bissett_garavan_t2.csv')
nback_t2 = pd.read_csv(import_directory/'1_Task_Cleaning'/'nback_manual_t2.csv')
cbcl_ss = pd.read_csv(import_directory/'2_Clinical_Cleaning'/'cbcl_ss.csv')
demo_t0_split = pd.read_csv(import_directory/'5_Split'/'demo_t0_split.csv')

"""
Preparing Dataframes
"""
t_cols = ['src_subject_id', 'eventname'] + [col for col in cbcl_ss if col.endswith('_t')] 
cbcl_ss = cbcl_ss[t_cols]

#select waves of clinical assessments
rename_cols = [col for col in sst_t0.columns if col not in ['src_subject_id', 'eventname']]
sst_t0 = sst_t0.rename(columns={col: f'{col}_0' for col in rename_cols})
sst_t0 = sst_t0.drop(columns='eventname')

rename_cols = [col for col in sst_t2.columns if col not in ['src_subject_id', 'eventname']]
sst_t2 = sst_t2.rename(columns={col: f'{col}_2' for col in rename_cols})
sst_t2 = sst_t2.drop(columns='eventname')

rename_cols = [col for col in nback_t0.columns if col not in ['src_subject_id', 'eventname']]
nback_t0 = nback_t0.rename(columns={col: f'{col}_0' for col in rename_cols})
nback_t0 = nback_t0.drop(columns='eventname')

rename_cols = [col for col in nback_t2.columns if col not in ['src_subject_id', 'eventname']]
nback_t2 = nback_t2.rename(columns={col: f'{col}_2' for col in rename_cols})
nback_t2 = nback_t2.drop(columns='eventname')

tasks_t0 = pd.merge(sst_t0, nback_t0, on='src_subject_id', how='outer')
tasks_t2 = pd.merge(sst_t2, nback_t2, on='src_subject_id', how='outer')
tasks_t02 = pd.merge(tasks_t0, tasks_t2, on='src_subject_id', how='outer')

cbcl_t0 = cbcl_ss[cbcl_ss['eventname'] == 'baseline_year_1_arm_1']
rename_cols = [col for col in cbcl_t0.columns if col not in ['src_subject_id', 'eventname']]
cbcl_t0 = cbcl_t0.rename(columns={col: f'{col}_0' for col in rename_cols})
cbcl_t0 = cbcl_t0.drop(columns='eventname')

cbcl_t1 = cbcl_ss[cbcl_ss['eventname'] == '1_year_follow_up_y_arm_1']
rename_cols = [col for col in cbcl_t1.columns if col not in ['src_subject_id', 'eventname']]
cbcl_t1 = cbcl_t1.rename(columns={col: f'{col}_1' for col in rename_cols})
cbcl_t1 = cbcl_t1.drop(columns='eventname')

cbcl_t2 = cbcl_ss[cbcl_ss['eventname'] == '2_year_follow_up_y_arm_1']
rename_cols = [col for col in cbcl_t2.columns if col not in ['src_subject_id', 'eventname']]
cbcl_t2 = cbcl_t2.rename(columns={col: f'{col}_2' for col in rename_cols})
cbcl_t2 = cbcl_t2.drop(columns='eventname')

cbcl_t3 = cbcl_ss[cbcl_ss['eventname'] == '3_year_follow_up_y_arm_1']
rename_cols = [col for col in cbcl_t3.columns if col not in ['src_subject_id', 'eventname']]
cbcl_t3 = cbcl_t3.rename(columns={col: f'{col}_3' for col in rename_cols})
cbcl_t3 = cbcl_t3.drop(columns='eventname')

cbcl_t01 = pd.merge(cbcl_t0, cbcl_t1, on='src_subject_id', how='outer')
cbcl_t012 = pd.merge(cbcl_t01, cbcl_t2, on='src_subject_id', how='outer')
cbcl_t0123 = pd.merge(cbcl_t012, cbcl_t3, on='src_subject_id', how='outer')

#get participant lists
sample_ids = demo_t0_split[demo_t0_split['sample_ids'] == 1]
sample_ids = sample_ids['src_subject_id'].tolist()

cbcl_t0123 = cbcl_t0123[cbcl_t0123['src_subject_id'].isin(sample_ids)]
tasks_t02 = tasks_t02[tasks_t02['src_subject_id'].isin(sample_ids)]

#combine all dataframes
tasks_cbcl_t0123 = pd.merge(tasks_t02, cbcl_t0123, on='src_subject_id', how='outer')

"""
Correlation Tables
"""
cols = tasks_cbcl_t0123.select_dtypes(include=[np.number]).columns

r_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
p_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
n_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

for i in cols:
    for j in cols:
        valid = tasks_cbcl_t0123[[i, j]].dropna()
        n = len(valid)
        n_matrix.loc[i, j] = n
        if n > 1:
            x = valid[i].to_numpy().ravel()
            y = valid[j].to_numpy().ravel()
            r, p = pearsonr(x, y)
        else:
            r, p = np.nan, np.nan
        r_matrix.loc[i, j] = r
        p_matrix.loc[i, j] = p

#flatten upper triangle of p-values for FDR
p_vals = p_matrix.where(np.triu(np.ones(p_matrix.shape), k=1).astype(bool)).stack().values

reject, p_fdr, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

fdr_p_matrix = p_matrix.copy()
fdr_p_matrix[:] = np.nan 
fdr_p_matrix.values[np.triu_indices_from(fdr_p_matrix, k=1)] = p_fdr
fdr_p_matrix = fdr_p_matrix.combine_first(fdr_p_matrix.T)

fdr_reject_matrix = (fdr_p_matrix < 0.05) #boolean FDR mask

np.fill_diagonal(r_matrix.values, np.nan)
np.fill_diagonal(p_matrix.values, np.nan)
np.fill_diagonal(n_matrix.values, np.nan)
np.fill_diagonal(fdr_reject_matrix.values, np.nan)

"""
Export Files
"""
r_matrix.to_csv(export_directory/'r_matrix.csv', index=False)
p_matrix.to_csv(export_directory/'p_matrix.csv', index=False)
n_matrix.to_csv(export_directory/'n_matrix.csv', index=False)
fdr_p_matrix.to_csv(export_directory/'fdr_p_matrix.csv', index=False)
fdr_reject_matrix.to_csv(export_directory/'fdr_reject_matrix.csv', index=False)
