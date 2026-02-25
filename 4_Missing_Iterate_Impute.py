#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Misisng Imputation: Scikit-Learn Iterative Imputer  
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- IterativeImputer imputation of missing in the inhibitory control and working memory behavioral task data from the Adolescent Brain Cognitive Development (ABCD) Study baseline and 2-year follow-up waves. 

Inputs(s):
- sst_t0_shift_impute.csv
- sst_t2_shift_impute.csv
- nback_manual_t0.csv
- nback_manual_t2.csv

Output(s):
- sst_t0_all_impute.csv
- sst_t2_all_impute.csv
- nback_t0_all_impute.csv
- nback_t2_all_impute.csv
- sst_t0_only_impute.csv
- sst_t2_only_impute.csv
- nback_t0_only_impute.csv
- nback_t2_only_impute.csv
- sst_t0_rt_impute.csv
- sst_t2_rt_impute.csv
- nback_t0_rt_impute.csv
- nback_t2_rt_impute.csv

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4
- scikit-learn version: 1.6.1

Notes:
- 'IterativeImputer is experimental and the API might change without any deprecation cycle. 
  To use it, you need to explicitly import enable_iterative_imputer: from sklearn.experimental import enable_iterative_imputer'
"""
#core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os
from tqdm import tqdm
import warnings

#imputation 
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#load data
sst_t0_shift_impute = pd.read_csv(export_directory / 'sst_t0_shift_impute.csv')
sst_t2_shift_impute = pd.read_csv(export_directory / 'sst_t2_shift_impute.csv')
nback_manual_t0 = pd.read_csv(export_directory / 'nback_manual_t0.csv')
nback_manual_t2 = pd.read_csv(export_directory / 'nback_manual_t2.csv')

"""
Function(s)
"""
def impute_missing(df, id_df, cols, seed=843):
    imputer = IterativeImputer(random_state=seed)
    imputed_array = imputer.fit_transform(df[cols])
    imputed_df = pd.DataFrame(imputed_array, columns=cols, index=df.index)
    merged_df = pd.concat([id_df, imputed_df], axis=1)
    return merged_df

"""
Prepare Data
"""
#sort and re-index
dfs = [sst_t0_shift_impute, sst_t2_shift_impute, nback_manual_t0, nback_manual_t2]
dfs = [df.sort_values(by='src_subject_id').reset_index(drop=True) for df in dfs]
sst_t0_shift_impute, sst_t2_shift_impute, nback_manual_t0, nback_manual_t2 = dfs

del dfs

#keep participants who completed both tasks at baseline 
sst_nback_t0 = pd.merge(
    sst_t0_shift_impute, 
    nback_manual_t0, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)

#at baseline 8830 participants have SST and EFnback 
sst_nback_t0.shape[0]

#keep participants who completed both tasks at 2-year follow-up 
sst_nback_t2 = pd.merge(
    sst_t2_shift_impute, 
    nback_manual_t2, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)
#at 2-year follow-up 6868 participants have SST and EFnback 
sst_nback_t2.shape[0]

#check missing
sst_nback_t0_miss_cols = (sst_nback_t0.isnull().mean() * 100).round(2)
sst_nback_t2_miss_cols = (sst_nback_t2.isnull().mean() * 100).round(2)

#drop incorrect late go
incrlg_cols = ['sst_incrlg_mrt', 'sst_incrlg_stdrt']
sst_nback_t0 = sst_nback_t0.drop(columns=incrlg_cols) #missing 69.2%
sst_nback_t2 = sst_nback_t2.drop(columns=incrlg_cols) #missing 81.67%

del incrlg_cols

#get list of ids by timepoint
id_cols = ['src_subject_id', 'eventname']

t0_ids = sst_nback_t0[id_cols]
t2_ids = sst_nback_t2[id_cols]

del sst_t0_shift_impute, sst_t2_shift_impute, nback_manual_t0, nback_manual_t2

"""
Impute Missing: Method 1
- SST with All (i.e., SST and EFnBack) Features
- EFnBack with SST and EFnBack Features
"""
sst_nback_cols = [col for col in sst_nback_t0 if (col.startswith('sst_') or col.startswith('nback_'))]

#baseline
all_t0 = impute_missing(sst_nback_t0, t0_ids, sst_nback_cols)

#2-year follow-up
all_t2 = impute_missing(sst_nback_t2, t2_ids, sst_nback_cols)

#check missing and export files
all_t0_miss_cols = (all_t0.isnull().mean() * 100).round(2)
all_t2_miss_cols = (all_t2.isnull().mean() * 100).round(2)

del all_t0_miss_cols, all_t2_miss_cols

sst_cols = ['src_subject_id', 'eventname'] + [col for col in sst_nback_t0 if col.startswith('sst_')]
nback_cols = ['src_subject_id', 'eventname'] + [col for col in sst_nback_t0 if col.startswith('nback_')]

sst_t0_all_impute = all_t0[sst_cols]
sst_t2_all_impute = all_t2[sst_cols]
nback_t0_all_impute = all_t0[nback_cols]
nback_t2_all_impute = all_t2[nback_cols]

sst_t0_all_impute.to_csv(export_directory/'sst_t0_all_impute.csv', index=False)
sst_t2_all_impute.to_csv(export_directory/'sst_t2_all_impute.csv', index=False)
nback_t0_all_impute.to_csv(export_directory/'nback_t0_all_impute.csv', index=False)
nback_t2_all_impute.to_csv(export_directory/'nback_t2_all_impute.csv', index=False)

del sst_nback_cols
del all_t0, all_t2

"""
Impute Missing: Method 2
- SST with Only SST Features 
- EFnBack with Only EFnBack Features
"""
sst_cols = [col for col in sst_nback_t0 if col.startswith("sst_")]
nback_cols = [col for col in sst_nback_t0 if col.startswith("nback_")]

#baseline
sst_only_t0 = impute_missing(sst_nback_t0, t0_ids, sst_cols)
nback_only_t0 = impute_missing(sst_nback_t0, t0_ids, nback_cols)

#2-year follow-up
sst_only_t2 = impute_missing(sst_nback_t2, t2_ids, sst_cols)
nback_only_t2 = impute_missing(sst_nback_t2, t2_ids, nback_cols)

#check missing and export files
sst_t0_miss_cols = (sst_only_t0.isnull().mean() * 100).round(2)
sst_t2_miss_cols = (sst_only_t2.isnull().mean() * 100).round(2)
nback_t0_miss_cols = (nback_only_t0.isnull().mean() * 100).round(2)
nback_t2_miss_cols = (nback_only_t2.isnull().mean() * 100).round(2)

sst_t0_only_impute = sst_only_t0
sst_t2_only_impute = sst_only_t2 
nback_t0_only_impute = nback_only_t0 
nback_t2_only_impute = nback_only_t2

sst_t0_only_impute.to_csv(export_directory/'sst_t0_only_impute.csv', index=False)
sst_t2_only_impute.to_csv(export_directory/'sst_t2_only_impute.csv', index=False)
nback_t0_only_impute.to_csv(export_directory/'nback_t0_only_impute.csv', index=False)
nback_t2_only_impute.to_csv(export_directory/'nback_t2_only_impute.csv', index=False)

del sst_cols, nback_cols
del sst_t0_miss_cols, sst_t2_miss_cols, nback_t0_miss_cols, nback_t2_miss_cols

"""
Impute Missing: Method 3
- SST Mean Reaction Times (MRT) with Only SST MRT Features 
- SST Standard Deviation of Reaction Times (STDRT) with Only SST STDRT Features 
- EFnBack MRT with Only EFnBack MRT Features 
- EFnBack STDRT with Only EFnBack STDRT Features 
"""
sst_mrt_cols = [col for col in sst_nback_t0 if (col.startswith('sst_') and col.endswith('_mrt'))]
sst_stdrt_cols = [col for col in sst_nback_t0 if (col.startswith('sst_') and col.endswith('_stdrt'))]

exclude_mrt = ['nback_c0b_mrt', 'nback_c2b_mrt']
exclude_stdrt = ['nback_c0b_stdrt', 'nback_c2b_stdrt']

nback_mrt_cols = [col for col in sst_nback_t0 if col.startswith('nback_') and col.endswith('_mrt') and col not in exclude_mrt]
nback_stdrt_cols = [col for col in sst_nback_t0 if col.startswith('nback_') and col.endswith('_stdrt') and col not in exclude_stdrt]

del exclude_mrt, exclude_stdrt

#baseline
#sst
sst_mrt_t0 = impute_missing(sst_nback_t0, t0_ids, sst_mrt_cols)
sst_stdrt_t0 = impute_missing(sst_nback_t0, t0_ids, sst_stdrt_cols)

sst_rt_t0 = pd.merge(
    sst_mrt_t0, 
    sst_stdrt_t0, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)

#nback
nback_mrt_t0 = impute_missing(sst_nback_t0, t0_ids, nback_mrt_cols)
nback_stdrt_t0 = impute_missing(sst_nback_t0, t0_ids, nback_stdrt_cols)

nback_rt_t0 = pd.merge(
    nback_mrt_t0, 
    nback_stdrt_t0, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)

#2-year follow-up
#sst
sst_mrt_t2 = impute_missing(sst_nback_t2, t2_ids, sst_mrt_cols)
sst_stdrt_t2 = impute_missing(sst_nback_t2, t2_ids, sst_stdrt_cols)

sst_rt_t2 = pd.merge(
    sst_mrt_t2, 
    sst_stdrt_t2, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)

#nback
nback_mrt_t2 = impute_missing(sst_nback_t2, t2_ids, nback_mrt_cols) #no missing
nback_stdrt_t2 = impute_missing(sst_nback_t2, t2_ids, nback_stdrt_cols) #no missing

nback_rt_t2 = pd.merge(
    nback_mrt_t2, 
    nback_stdrt_t2, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)

del sst_mrt_cols, sst_stdrt_cols, nback_mrt_cols, nback_stdrt_cols

#check missing, merge, and export files
sst_rate_total_cols = ['src_subject_id', 
                       'eventname'
                       ] + [col for col in sst_nback_t0 if (col.startswith('sst_') and col.endswith('_rate'))] + ['sst_mssd', 
                                                                                                                             'sst_mssrt', 
                                                                                                                             'sst_issrt'
                                                                                                                            ]
sst_rate_total_t0 = sst_nback_t0[sst_rate_total_cols]
sst_t0_rt_impute = pd.merge(
    sst_rt_t0, 
    sst_rate_total_t0, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)
sst_rate_total_t2 = sst_nback_t2[sst_rate_total_cols]
sst_t2_rt_impute = pd.merge(
    sst_rt_t2, 
    sst_rate_total_t2, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)

nback_rate_total_cols = ['src_subject_id', 
                       'eventname'
                       ] + [col for col in sst_nback_t0 if (col.startswith('nback_') and col.endswith('_rate'))] + ['nback_c0b_mrt', 
                                                                                                                               'nback_c2b_mrt', 
                                                                                                                               'nback_c0b_stdrt', 
                                                                                                                               'nback_c2b_stdrt'
                                                                                                                               ]
nback_rate_total_t0 = sst_nback_t0[nback_rate_total_cols]
nback_t0_rt_impute = pd.merge(
    nback_rt_t0, 
    nback_rate_total_t0, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)
nback_rate_total_t2 = sst_nback_t2[nback_rate_total_cols]
nback_t2_rt_impute = pd.merge(
    nback_rt_t2, 
    nback_rate_total_t2, 
    how='inner', 
    on=['src_subject_id', 'eventname']
)

del sst_rate_total_cols, nback_rate_total_cols
del sst_rt_t0, sst_rate_total_t0, nback_rt_t0, nback_rate_total_t0, sst_rt_t2, sst_rate_total_t2, nback_rt_t2, nback_rate_total_t2

sst_t0_miss_cols = (sst_t0_rt_impute.isnull().mean() * 100).round(2)
sst_t2_miss_cols = (sst_t2_rt_impute.isnull().mean() * 100).round(2)
nback_t0_miss_cols = (nback_t0_rt_impute.isnull().mean() * 100).round(2)
nback_t2_miss_cols = (nback_t2_rt_impute.isnull().mean() * 100).round(2)

del sst_t0_miss_cols, sst_t2_miss_cols, nback_t0_miss_cols, nback_t2_miss_cols

sst_t0_rt_impute.to_csv(export_directory/'sst_t0_rt_impute.csv', index=False)
sst_t2_rt_impute.to_csv(export_directory/'sst_t2_rt_impute.csv', index=False)
nback_t0_rt_impute.to_csv(export_directory/'nback_t0_rt_impute.csv', index=False)
nback_t2_rt_impute.to_csv(export_directory/'nback_t2_rt_impute.csv', index=False)

#compare dimensions
sst_t0_all_impute.shape
sst_t0_only_impute.shape
sst_t0_rt_impute.shape

sst_t2_all_impute.shape
sst_t2_only_impute.shape
sst_t2_rt_impute.shape



