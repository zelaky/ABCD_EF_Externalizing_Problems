#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Data Compilation
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Merging dataframes for final analyses.

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

#load task data
sst_t0_all = pd.read_csv(import_directory/'1_Task_Cleaning'/'sst_t0_all.csv')
sst_t2_all = pd.read_csv(import_directory/'1_Task_Cleaning'/'sst_t2_all.csv')

nback_t0_all = pd.read_csv(import_directory/'1_Task_Cleaning'/'nback_t0_all.csv')
nback_t2_all = pd.read_csv(import_directory/'1_Task_Cleaning'/'nback_t2_all.csv')

sst_t0_only = pd.read_csv(import_directory/'1_Task_Cleaning'/'sst_t0_only.csv')
sst_t2_only = pd.read_csv(import_directory/'1_Task_Cleaning'/'sst_t2_only.csv')

nback_t0_only = pd.read_csv(import_directory/'1_Task_Cleaning'/'nback_t0_only.csv')
nback_t2_only = pd.read_csv(import_directory/'1_Task_Cleaning'/'nback_t2_only.csv')

sst_t0_rt = pd.read_csv(import_directory/'1_Task_Cleaning'/'sst_t0_rt.csv')
sst_t2_rt = pd.read_csv(import_directory/'1_Task_Cleaning'/'sst_t2_rt.csv')

nback_t0_rt = pd.read_csv(import_directory/'1_Task_Cleaning'/'nback_t0_rt.csv')
nback_t2_rt = pd.read_csv(import_directory/'1_Task_Cleaning'/'nback_t2_rt.csv')

#load clinical data
cbcl_ss = pd.read_csv(import_directory/'2_Clinical_Cleaning'/'cbcl_ss.csv')
ksads_ss = pd.read_csv(import_directory/'2_Clinical_Cleaning'/'ksads_ss.csv')

#load demoraphics
demo_t0_split = pd.read_csv(import_directory/'5_Split'/'demo_t0_split.csv')
demo_full = pd.read_csv(import_directory/'3_Demographic_Cleaning'/'demo_full.csv')

#load pc data
sst_t0_all_pca_90 = pd.read_csv(import_directory/'6_PCA'/'sst_t0_all_pca_90.csv')
sst_t2_all_pca_90 = pd.read_csv(import_directory/'6_PCA'/'sst_t2_all_pca_90.csv')

nback_t0_all_pca_90 = pd.read_csv(import_directory/'6_PCA'/'nback_t0_all_pca_90.csv')
nback_t2_all_pca_90 = pd.read_csv(import_directory/'6_PCA'/'nback_t2_all_pca_90.csv')

sst_t0_only_pca_90 = pd.read_csv(import_directory/'6_PCA'/'sst_t0_only_pca_90.csv')
sst_t2_only_pca_90 = pd.read_csv(import_directory/'6_PCA'/'sst_t2_only_pca_90.csv')

nback_t0_only_pca_90 = pd.read_csv(import_directory/'6_PCA'/'nback_t0_only_pca_90.csv')
nback_t2_only_pca_90 = pd.read_csv(import_directory/'6_PCA'/'nback_t2_only_pca_90.csv')

sst_t0_rt_pca_90 = pd.read_csv(import_directory/'6_PCA'/'sst_t0_rt_pca_90.csv')
sst_t2_rt_pca_90 = pd.read_csv(import_directory/'6_PCA'/'sst_t2_rt_pca_90.csv')

nback_t0_rt_pca_90 = pd.read_csv(import_directory/'6_PCA'/'nback_t0_rt_pca_90.csv')
nback_t2_rt_pca_90 = pd.read_csv(import_directory/'6_PCA'/'nback_t2_rt_pca_90.csv')

#load model labels
tasks_t0_all_km_labels = pd.read_csv(import_directory/'7_Algorithm_Tuning'/'KMeans'/'tasks_t0_all_km_labels.csv')
tasks_t2_all_km_labels = pd.read_csv(import_directory/'7_Algorithm_Tuning'/'KMeans'/'tasks_t2_all_km_labels.csv')

tasks_t0_only_km_labels = pd.read_csv(import_directory/'7_Algorithm_Tuning'/'KMeans'/'tasks_t0_only_km_labels.csv')
tasks_t2_only_km_labels = pd.read_csv(import_directory/'7_Algorithm_Tuning'/'KMeans'/'tasks_t2_only_km_labels.csv')

tasks_t0_rt_km_labels = pd.read_csv(import_directory/'7_Algorithm_Tuning'/'KMeans'/'tasks_t0_rt_km_labels.csv')
tasks_t2_rt_km_labels = pd.read_csv(import_directory/'7_Algorithm_Tuning'/'KMeans'/'tasks_t2_rt_km_labels.csv')

tasks_t0_singleton_km_labels = pd.read_csv(import_directory/'7_Algorithm_Tuning'/'KMeans'/'tasks_t0_singleton_km_labels.csv')
tasks_t2_singleton_km_labels = pd.read_csv(import_directory/'7_Algorithm_Tuning'/'KMeans'/'tasks_t2_singleton_km_labels.csv')

"""
Prepare Dataframes
"""
ksads_ss = ksads_ss[ksads_ss['eventname'] != '4_year_follow_up_y_arm_1'].reset_index(drop=True)
cbcl_ss = cbcl_ss[cbcl_ss['eventname'] != '4_year_follow_up_y_arm_1'].reset_index(drop=True)

ksads_ss_pres = ksads_ss[['src_subject_id',
                          'eventname',
                         'ksads_dmdd', 
                         'ksads_adhd_other', 
                         'ksads_adhd_present', 
                         'ksads_odd_present', 
                         'ksads_cd_present_child', 
                         'ksads_cd_present_adolescent']]

cbcl_cols = ['src_subject_id', 'eventname'] + [col for col in cbcl_ss.columns if col.endswith('_t')]

cbcl_ss_pres = cbcl_ss[cbcl_cols]

wave_map = {
    'baseline_year_1_arm_1': 't0',
    '1_year_follow_up_y_arm_1': 't1',
    '2_year_follow_up_y_arm_1': 't2',
    '3_year_follow_up_y_arm_1': 't3'}

ksads_ss_pres.loc[:, 'eventname'] = ksads_ss_pres['eventname'].replace(wave_map)
cbcl_ss_pres.loc[:, 'eventname'] = cbcl_ss_pres['eventname'].replace(wave_map)

ksads_ss_pres = ksads_ss_pres.pivot(index='src_subject_id', columns='eventname')
ksads_ss_pres.columns = ['{}_{}'.format(col[0], col[1]) for col in ksads_ss_pres.columns]
ksads_ss_pres = ksads_ss_pres.reset_index()

cbcl_ss_pres = cbcl_ss_pres.pivot(index='src_subject_id', columns='eventname')
cbcl_ss_pres.columns = ['{}_{}'.format(col[0], col[1]) for col in cbcl_ss_pres.columns]
cbcl_ss_pres = cbcl_ss_pres.reset_index()

nback_t0_all.columns = [col.replace("nback_", "nback_t0_") if col.startswith("nback_") else col for col in nback_t0_all.columns]
nback_t0_only.columns = [col.replace("nback_", "nback_t0_") if col.startswith("nback_") else col for col in nback_t0_only.columns]
nback_t0_rt.columns = [col.replace("nback_", "nback_t0_") if col.startswith("nback_") else col for col in nback_t0_rt.columns]
nback_t2_all.columns = [col.replace("nback_", "nback_t2_") if col.startswith("nback_") else col for col in nback_t2_all.columns]
nback_t2_only.columns = [col.replace("nback_", "nback_t2_") if col.startswith("nback_") else col for col in nback_t2_only.columns]
nback_t2_rt.columns = [col.replace("nback_", "nback_t2_") if col.startswith("nback_") else col for col in nback_t2_rt.columns]

sst_t0_all.columns = [col.replace("sst_", "sst_t0_") if col.startswith("sst_") else col for col in sst_t0_all.columns]
sst_t0_only.columns = [col.replace("sst_", "sst_t0_") if col.startswith("sst_") else col for col in sst_t0_only.columns]
sst_t0_rt.columns = [col.replace("sst_", "sst_t0_") if col.startswith("sst_") else col for col in sst_t0_rt.columns]
sst_t2_all.columns = [col.replace("sst_", "sst_t2_") if col.startswith("sst_") else col for col in sst_t2_all.columns]
sst_t2_only.columns = [col.replace("sst_", "sst_t2_") if col.startswith("sst_") else col for col in sst_t2_only.columns]
sst_t2_rt.columns = [col.replace("sst_", "sst_t2_") if col.startswith("sst_") else col for col in sst_t2_rt.columns]

demo_t0_sample = demo_t0_split[demo_t0_split['sample_ids']==1]
demo_t0_sample = demo_t0_sample[['src_subject_id', 
                                 'interview_age', 
                                 'demo_ed_v2', 
                                 'demo_sex_rc', 
                                 'demo_gender_id_v2', 
                                 'demo_comb_income_v2', 
                                 'race_ethnicity', 
                                 'site_id_l', 
                                 'rel_family_id', 
                                 'rel_birth_id', 
                                 'complete_ids', 
                                 'train_ids', 
                                 'test_ids']]
demo_t0_sample = demo_t0_sample.rename(columns={'interview_age': 'interview_age_t0'})
sample_ids = demo_t0_sample['src_subject_id'].tolist()
demo_full = demo_full[demo_full['src_subject_id'].isin(sample_ids)]
demo_t2_sample = demo_full[demo_full['eventname']=='2_year_follow_up_y_arm_1']
demo_t2_sample = demo_t2_sample[['src_subject_id', 'interview_age']]
demo_t2_sample = demo_t2_sample.rename(columns={'interview_age': 'interview_age_t2'})
demo_sample = pd.merge(demo_t0_sample, demo_t2_sample, on='src_subject_id', how='outer')

ksads_ss_pres = ksads_ss_pres[ksads_ss_pres['src_subject_id'].isin(sample_ids)]
cbcl_ss_pres = cbcl_ss_pres[cbcl_ss_pres['src_subject_id'].isin(sample_ids)]

nback_t0_all = nback_t0_all[nback_t0_all['src_subject_id'].isin(sample_ids)]
nback_t0_only = nback_t0_only[nback_t0_only['src_subject_id'].isin(sample_ids)]
nback_t0_rt = nback_t0_rt[nback_t0_rt['src_subject_id'].isin(sample_ids)]
nback_t2_all = nback_t2_all[nback_t2_all['src_subject_id'].isin(sample_ids)]
nback_t2_only = nback_t2_only[nback_t2_only['src_subject_id'].isin(sample_ids)]
nback_t2_rt = nback_t2_rt[nback_t2_rt['src_subject_id'].isin(sample_ids)]

sst_t0_all = sst_t0_all[sst_t0_all['src_subject_id'].isin(sample_ids)]
sst_t0_only = sst_t0_only[sst_t0_only['src_subject_id'].isin(sample_ids)]
sst_t0_rt = sst_t0_rt[sst_t0_rt['src_subject_id'].isin(sample_ids)]
sst_t2_all = sst_t2_all[sst_t2_all['src_subject_id'].isin(sample_ids)]
sst_t2_only = sst_t2_only[sst_t2_only['src_subject_id'].isin(sample_ids)]
sst_t2_rt = sst_t2_rt[sst_t2_rt['src_subject_id'].isin(sample_ids)]

nback_t0_all = nback_t0_all.drop(columns='eventname')
nback_t0_only = nback_t0_only.drop(columns='eventname')
nback_t0_rt = nback_t0_rt.drop(columns='eventname')
nback_t2_all = nback_t2_all.drop(columns='eventname')
nback_t2_only = nback_t2_only.drop(columns='eventname')
nback_t2_rt = nback_t2_rt.drop(columns='eventname')

sst_t0_all = sst_t0_all.drop(columns='eventname')
sst_t0_only = sst_t0_only.drop(columns='eventname')
sst_t0_rt = sst_t0_rt.drop(columns='eventname')
sst_t2_all = sst_t2_all.drop(columns='eventname')
sst_t2_only = sst_t2_only.drop(columns='eventname')
sst_t2_rt = sst_t2_rt.drop(columns='eventname')

sst_t0_all_pca_90.columns = [col.replace('sst_t0_', 'sst_t0_pc') if col.startswith('sst_t0_') else col for col in sst_t0_all_pca_90.columns]
sst_t0_only_pca_90.columns = [col.replace('sst_t0_', 'sst_t0_pc') if col.startswith('sst_t0_') else col for col in sst_t0_only_pca_90.columns]
sst_t0_rt_pca_90.columns = [col.replace('sst_t0_', 'sst_t0_pc') if col.startswith('sst_t0_') else col for col in sst_t0_rt_pca_90.columns]
sst_t2_all_pca_90.columns = [col.replace('sst_t2_', 'sst_t2_pc') if col.startswith('sst_t2_') else col for col in sst_t2_all_pca_90.columns]
sst_t2_only_pca_90.columns = [col.replace('sst_t2_', 'sst_t2_pc') if col.startswith('sst_t2_') else col for col in sst_t2_only_pca_90.columns]
sst_t2_rt_pca_90.columns = [col.replace('sst_t2_', 'sst_t2_pc') if col.startswith('sst_t2_') else col for col in sst_t2_rt_pca_90.columns]

nback_t0_all_pca_90.columns = [col.replace('nback_t0_', 'nback_t0_pc') if col.startswith('nback_t0_') else col for col in nback_t0_all_pca_90.columns]
nback_t0_only_pca_90.columns = [col.replace('nback_t0_', 'nback_t0_pc') if col.startswith('nback_t0_') else col for col in nback_t0_only_pca_90.columns]
nback_t0_rt_pca_90.columns = [col.replace('nback_t0_', 'nback_t0_pc') if col.startswith('nback_t0_') else col for col in nback_t0_rt_pca_90.columns]
nback_t2_all_pca_90.columns = [col.replace('nback_t2_', 'nback_t2_pc') if col.startswith('nback_t2_') else col for col in nback_t2_all_pca_90.columns]
nback_t2_only_pca_90.columns = [col.replace('nback_t2_', 'nback_t2_pc') if col.startswith('nback_t2_') else col for col in nback_t2_only_pca_90.columns]
nback_t2_rt_pca_90.columns = [col.replace('nback_t2_', 'nback_t2_pc') if col.startswith('nback_t2_') else col for col in nback_t2_rt_pca_90.columns]

tasks_t0_all_km_labels.columns = [col.replace("_all_", "_") for col in tasks_t0_all_km_labels.columns]
tasks_t2_all_km_labels.columns = [col.replace("_all_", "_") for col in tasks_t2_all_km_labels.columns]
tasks_t0_only_km_labels.columns = [col.replace("_only_", "_") for col in tasks_t0_only_km_labels.columns]
tasks_t2_only_km_labels.columns = [col.replace("_only_", "_") for col in tasks_t2_only_km_labels.columns]
tasks_t0_rt_km_labels.columns = [col.replace("_rt_", "_") for col in tasks_t0_rt_km_labels.columns]
tasks_t2_rt_km_labels.columns = [col.replace("_rt_", "_") for col in tasks_t2_rt_km_labels.columns]
tasks_t0_singleton_km_labels.columns = [col.replace("_singleton_", "_") for col in tasks_t0_singleton_km_labels.columns]
tasks_t2_singleton_km_labels.columns = [col.replace("_singleton_", "_") for col in tasks_t2_singleton_km_labels.columns]

tasks_t0_all_km_labels = tasks_t0_all_km_labels[['src_subject_id','tasks_t0_km_2']]
tasks_t0_only_km_labels = tasks_t0_only_km_labels[['src_subject_id','tasks_t0_km_2']]
tasks_t0_rt_km_labels = tasks_t0_rt_km_labels[['src_subject_id','tasks_t0_km_2']]
tasks_t0_singleton_km_labels = tasks_t0_singleton_km_labels[['src_subject_id','tasks_t0_km_2']]

tasks_t2_all_km_labels = tasks_t2_all_km_labels[['src_subject_id','tasks_t2_km_2']]
tasks_t2_only_km_labels = tasks_t2_only_km_labels[['src_subject_id','tasks_t2_km_2']]
tasks_t2_rt_km_labels = tasks_t2_rt_km_labels[['src_subject_id','tasks_t2_km_2']]
tasks_t2_singleton_km_labels = tasks_t2_singleton_km_labels[['src_subject_id','tasks_t2_km_2']]

"""
All Impute
"""
tasks_all_km_labels = pd.merge(tasks_t0_all_km_labels, tasks_t2_all_km_labels, on='src_subject_id', how='inner')

all_dfs = [demo_sample, 
           sst_t0_all,
           sst_t2_all, 
           nback_t0_all, 
           nback_t2_all,
           sst_t0_all_pca_90,
           sst_t2_all_pca_90,
           nback_t0_all_pca_90,
           nback_t2_all_pca_90,
           tasks_all_km_labels,
           ksads_ss_pres,
           cbcl_ss_pres]

abcd_all = all_dfs[0]
for df in all_dfs[1:]:
    abcd_all = abcd_all.merge(df, on='src_subject_id', how='inner')
    
"""
Singleton
"""
tasks_singleton_km_labels = pd.merge(tasks_t0_singleton_km_labels, tasks_t2_singleton_km_labels, on='src_subject_id', how='inner')

singleton_dfs = [demo_sample, 
           sst_t0_all,
           sst_t2_all, 
           nback_t0_all, 
           nback_t2_all,
           sst_t0_all_pca_90,
           sst_t2_all_pca_90,
           nback_t0_all_pca_90,
           nback_t2_all_pca_90,
           tasks_singleton_km_labels,
           ksads_ss_pres,
           cbcl_ss_pres]

abcd_singleton = singleton_dfs[0]
for df in singleton_dfs[1:]:
    abcd_singleton = abcd_singleton.merge(df, on='src_subject_id', how='inner')

"""
Only Impute
"""
tasks_only_km_labels = pd.merge(tasks_t0_only_km_labels, tasks_t2_only_km_labels, on='src_subject_id', how='inner')

only_dfs = [demo_sample, 
           sst_t0_only,
           sst_t2_only, 
           nback_t0_only, 
           nback_t2_only,
           sst_t0_only_pca_90,
           sst_t2_only_pca_90,
           nback_t0_only_pca_90,
           nback_t2_only_pca_90,
           tasks_only_km_labels,
           ksads_ss_pres,
           cbcl_ss_pres]

abcd_only = only_dfs[0]
for df in only_dfs[1:]:
    abcd_only = abcd_only.merge(df, on='src_subject_id', how='inner')

"""
RT Impute
"""
tasks_rt_km_labels = pd.merge(tasks_t0_rt_km_labels, tasks_t2_rt_km_labels, on='src_subject_id', how='inner')

rt_dfs = [demo_sample, 
           sst_t0_rt,
           sst_t2_rt, 
           nback_t0_rt, 
           nback_t2_rt,
           sst_t0_rt_pca_90,
           sst_t2_rt_pca_90,
           nback_t0_rt_pca_90,
           nback_t2_rt_pca_90,
           tasks_rt_km_labels,
           ksads_ss_pres,
           cbcl_ss_pres]

abcd_rt = rt_dfs[0]
for df in rt_dfs[1:]:
    abcd_rt = abcd_rt.merge(df, on='src_subject_id', how='inner')

"""
Export Dataframes
"""
abcd_all.to_csv(export_directory/'abcd_all.csv', index=False)
abcd_singleton.to_csv(export_directory/'abcd_singleton.csv', index=False)
abcd_only.to_csv(export_directory/'abcd_only.csv', index=False)
abcd_rt.to_csv(export_directory/'abcd_rt.csv', index=False)
