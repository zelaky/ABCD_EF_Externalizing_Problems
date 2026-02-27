#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Sample Selection 
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Excluding participants missing ABCD Study baseline or 2-year follow-up inhibitory control and working memory behavioral tasks. 

Input(s):
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

Output(s):
- sample_ids.csv

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
names = [
    'sst_t0_all_impute', 'sst_t2_all_impute',
    'sst_t0_only_impute', 'sst_t2_only_impute',
    'sst_t0_rt_impute', 'sst_t2_rt_impute',
    'nback_t0_all_impute', 'nback_t2_all_impute',
    'nback_t0_only_impute', 'nback_t2_only_impute',
    'nback_t0_rt_impute', 'nback_t2_rt_impute']

for name in names:
    globals()[name] = pd.read_csv(export_directory/f"{name}.csv")

"""
Identify Sample:
- Complete task data at baseline and 2-year follow-up for both tasks
"""
#there are 5,509 common ids across baseline and 2-year follow-up for SST
sst_common_ids = set(sst_t0_all_impute['src_subject_id']) & set(sst_t2_all_impute['src_subject_id'])

#there are 5,509 common ids across baseline and 2-year follow-up for EFnBack
nback_common_ids = set(nback_t0_all_impute['src_subject_id']) & set(nback_t2_all_impute['src_subject_id'])

#these ids are identical for both tasks
task_common_ids = sst_common_ids & nback_common_ids

del sst_common_ids, nback_common_ids

"""
Create Final Task Dataframes
"""
for name, df in list(globals().items()):
    if isinstance(df, pd.DataFrame) and 'src_subject_id' in df.columns:
        globals()[name] = df[df['src_subject_id'].isin(task_common_ids)].reset_index(drop=True)

"""
Export Files
"""
for name, df in list(globals().items()):
    if isinstance(df, pd.DataFrame) and "_impute" in name:
        new_name = name.replace("_impute", "")
        globals()[new_name] = df
        del globals()[name]
        df.to_csv(export_directory/f"{new_name}.csv", index=False)
        
sample_ids = pd.DataFrame(list(task_common_ids), columns=['src_subject_id'])
sample_ids.to_csv(export_directory/"sample_ids.csv", index=False)




