#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Principal Component Analysis (PCA) on Behavioral Tasks (Sibling Sensitivity)
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Applying PCA to baseline and 2-year follow-up SST and EFnBack tasks from randomly selected siblings (one child per family)
    
Input(s):
- sst_t0_all_pca_90.csv
- nback_t0_all_pca_90.csv
- sst_t2_all_pca_90.csv
- nback_t2_all_pca_90.csv
- demo_t0_split.csv

Output(s):
- sst_t0_singleton_pca_90.csv
- sst_t2_singleton_pca_90.csv
- nback_t0_singleton_pca_90.csv
- nback_t2_singleton_pca_90.csv
    
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

#statistical suite
from scipy.stats import chi2_contingency

#load data
demo_t0_split = pd.read_csv(import_directory/'5_Split'/'demo_t0_split.csv')
sst_t0_all_pca_90 = pd.read_csv(export_directory/'sst_t0_all_pca_90.csv')
sst_t2_all_pca_90 = pd.read_csv(export_directory/'sst_t2_all_pca_90.csv')
nback_t0_all_pca_90 = pd.read_csv(export_directory/'nback_t0_all_pca_90.csv')
nback_t2_all_pca_90 = pd.read_csv(export_directory/'nback_t2_all_pca_90.csv')

"""
Select One Family Member
"""
demo_t0_split = demo_t0_split[demo_t0_split['sample_ids'] == 1]
demo_t0_singleton = (demo_t0_split.groupby('rel_family_id', group_keys=False)[demo_t0_split.columns].apply(lambda x: x.sample(1, random_state=843)).reset_index(drop=True))

sex_split_all = demo_t0_singleton.groupby(['split_ids', 'demo_sex_rc'])['src_subject_id'].count()
sex_split_all = sex_split_all.reset_index()

#chi-square test on sex
test_freq = [1292, 1139]
train_freq = [1304, 1127]
table = pd.DataFrame({'Train': train_freq, 'Test': test_freq}, index=['Male', 'Female'])
stat, p_value, _, _ = chi2_contingency(table)

# stat = 0.10000822774395261, p-value = 0.7518197606560428

singleton_ids = demo_t0_singleton['src_subject_id'].tolist()

"""
Reduce Dataframe
"""
sst_t0_singleton_pca_90 = sst_t0_all_pca_90[sst_t0_all_pca_90['src_subject_id'].isin(singleton_ids)]
sst_t2_singleton_pca_90 = sst_t2_all_pca_90[sst_t2_all_pca_90['src_subject_id'].isin(singleton_ids)]
nback_t0_singleton_pca_90 = nback_t0_all_pca_90[nback_t0_all_pca_90['src_subject_id'].isin(singleton_ids)]
nback_t2_singleton_pca_90 = nback_t2_all_pca_90[nback_t2_all_pca_90['src_subject_id'].isin(singleton_ids)]

"""
Export Files
"""
sst_t0_singleton_pca_90.to_csv(export_directory/'sst_t0_singleton_pca_90.csv', index=False)
sst_t2_singleton_pca_90.to_csv(export_directory/'sst_t2_singleton_pca_90.csv', index=False)
nback_t0_singleton_pca_90.to_csv(export_directory/'nback_t0_singleton_pca_90.csv', index=False)
nback_t2_singleton_pca_90.to_csv(export_directory/'nback_t2_singleton_pca_90.csv', index=False)




