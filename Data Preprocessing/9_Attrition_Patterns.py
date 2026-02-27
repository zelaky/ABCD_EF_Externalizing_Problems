#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Attrition Rate
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Exploring data completeness patterns in the ABCD Study baseline, and 1-year, 2-year, 3-year, and 4-year follow-up waves. 

Input(s):
- sample_ids.csv
- demo_t0.csv
- cbcl_ss.csv
- ksads_ss.csv

Output(s):
pattern_miss.csv
demo_t0_pattern.csv
ksads_ex.csv
cbcl_ex.csv

Packages: 
- Spyder version: 6.07
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
- Participants must have completed both clinical tasks at baseline and 2-year follow-up.
"""
#core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os  

#load data
sample_ids = pd.read_csv(import_directory/'1_Task_Cleaning'/'sample_ids.csv')
cbcl_ss = pd.read_csv(import_directory/'2_Clinical_Cleaning'/'cbcl_ss.csv')
ksads_ss = pd.read_csv(import_directory/'2_Clinical_Cleaning'/'ksads_ss.csv')
demo_t0 = pd.read_csv(import_directory/'3_Demographic_Cleaning'/'demo_t0.csv')

"""
Sample Selection
"""
sample_ids = sample_ids['src_subject_id'].unique()

ksads_ex = ksads_ss[ksads_ss['src_subject_id'].isin(sample_ids)]
cbcl_ex = cbcl_ss[cbcl_ss['src_subject_id'].isin(sample_ids)]

"""
CBCL
- ID lists by wave 
"""
cbcl_waves = {}  

for key, group in cbcl_ss.groupby('eventname'):
    cbcl_waves[key] = group

cbcl_t0_ids = list(cbcl_waves.get('baseline_year_1_arm_1', {}).get('src_subject_id', [])) #n = 5509
cbcl_t1_ids = list(cbcl_waves.get('1_year_follow_up_y_arm_1', {}).get('src_subject_id', [])) #n = 5415
cbcl_t2_ids = list(cbcl_waves.get('2_year_follow_up_y_arm_1', {}).get('src_subject_id', [])) #n = 5500
cbcl_t3_ids = list(cbcl_waves.get('3_year_follow_up_y_arm_1', {}).get('src_subject_id', [])) #n = 5159
cbcl_t4_ids = list(cbcl_waves.get('4_year_follow_up_y_arm_1', {}).get('src_subject_id', [])) #n = 2831

del cbcl_ss, cbcl_waves

"""
KSADS
- ID lists by wave
"""
ksads_waves = {}  

for key, group in ksads_ss.groupby('eventname'):
    ksads_waves[key] = group

ksads_t0_ids = list(ksads_waves.get('baseline_year_1_arm_1', {}).get('src_subject_id', [])) #n = 5437
ksads_t1_ids = list(ksads_waves.get('1_year_follow_up_y_arm_1', {}).get('src_subject_id', [])) #n = 5352
ksads_t2_ids = list(ksads_waves.get('2_year_follow_up_y_arm_1', {}).get('src_subject_id', [])) #n = 5423
ksads_t3_ids = list(ksads_waves.get('3_year_follow_up_y_arm_1', {}).get('src_subject_id', [])) #n = 5236
ksads_t4_ids = list(ksads_waves.get('4_year_follow_up_y_arm_1', {}).get('src_subject_id', [])) #n = 2794

del ksads_waves, ksads_ss
del group
del key

"""
Missing Patterns 
- Clinical data from 4-year follow-up dropped, given high rattes of attrition and no cleaned task data. 
- Possible attrition profiles = 256 (i.e., 2 ** 8)
"""
cbcl_ex = cbcl_ex[cbcl_ex['eventname'] != '4_year_follow_up_y_arm_1']
ksads_ex = ksads_ex[ksads_ex['eventname'] != '4_year_follow_up_y_arm_1']
del cbcl_t4_ids, ksads_t4_ids 

cbcl_ids = {
    't0': cbcl_t0_ids,
    't1': cbcl_t1_ids,
    't2': cbcl_t2_ids,
    't3': cbcl_t3_ids 
}

ksads_ids = {
    't0': ksads_t0_ids,
    't1': ksads_t1_ids,
    't2': ksads_t2_ids,
    't3': ksads_t3_ids 
}

pattern_miss = demo_t0[['src_subject_id']].copy()

for timepoint, ids in cbcl_ids.items():
    pattern_miss[f'{timepoint}_cbcl'] = pattern_miss['src_subject_id'].isin(ids).astype(int)

for timepoint, ids in ksads_ids.items():
    pattern_miss[f'{timepoint}_ksads'] = pattern_miss['src_subject_id'].isin(ids).astype(int)

del cbcl_t0_ids, cbcl_t1_ids, cbcl_t2_ids, cbcl_t3_ids, cbcl_ids
del ksads_t0_ids, ksads_t1_ids, ksads_t2_ids, ksads_t3_ids, ksads_ids

mask = pattern_miss['src_subject_id'].isin(sample_ids).astype(int)
pattern_miss[['t0_tasks', 't2_tasks', 'sample_ids']] = pd.DataFrame([mask]*3).T.values

#pattern across all columns
all_cols = [col for col in pattern_miss.columns if col.startswith('t')]
pattern_miss['pattern_all'] = pattern_miss[all_cols].astype(str).apply(''.join, axis=1)
groups = pattern_miss.groupby('pattern_all')
pattern_miss['miss_category'] = groups['src_subject_id'].transform('ngroup') + 1  

del all_cols, ids, mask, timepoint, groups

pattern_miss['complete_ids'] = 0

pattern_miss.loc[pattern_miss['pattern_all'] == '1111111111', 'complete_ids'] = 1
unique_counts = pattern_miss['complete_ids'].value_counts()

#merge with demographics
pattern_ids = pattern_miss[['src_subject_id', 'pattern_all', 'miss_category', 'complete_ids', 'sample_ids']]
demo_t0 = pd.merge(demo_t0, pattern_ids, on='src_subject_id', how='outer')
demo_t0['complete_ids'] = demo_t0['complete_ids'].fillna(0)

del pattern_ids

"""
Export Files
"""
pattern_miss.to_csv(export_directory/'pattern_miss.csv', index=False)
demo_t0.to_csv(export_directory/'demo_t0_pattern.csv', index=False)

ksads_ex.to_csv(import_directory/'2_Clinical_Cleaning'/'ksads_ex.csv', index=False)
cbcl_ex.to_csv(import_directory/'2_Clinical_Cleaning'/'cbcl_ex.csv', index=False)


