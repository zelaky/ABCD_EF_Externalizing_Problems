#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Cluster Stability Sub-Groups (Main Analyses)
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Packages: 
- Spyder version: 6.07
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform 
import os
import pingouin as pg
from statsmodels.stats.multitest import multipletests

#load data
abcd_all = pd.read_csv(import_directory/'abcd_all.csv')
high_high_ids = pd.read_csv(import_directory/'Temporal_Stability'/'IDs'/'all_tasks_t0_km_2_t2_t0_0_t2_0_ids.csv')
high_low_ids = pd.read_csv(import_directory/'Temporal_Stability'/'IDs'/'all_tasks_t0_km_2_t2_t0_0_t2_1_ids.csv')
low_high_ids = pd.read_csv(import_directory/'Temporal_Stability'/'IDs'/'all_tasks_t0_km_2_t2_t0_1_t2_0_ids.csv')
low_low_ids = pd.read_csv(import_directory/'Temporal_Stability'/'IDs'/'all_tasks_t0_km_2_t2_t0_1_t2_1_ids.csv')

"""
Functions
"""
def oneway_welch_anova(df, dv, group, posthoc=True):
    data = df[[dv, group]].dropna()
    aov = pg.welch_anova(dv=dv, between=group, data=data)
    posthoc_test = None
    if posthoc:
        posthoc_test = pg.pairwise_gameshowell(dv=dv, between=group, data=data)
    return aov, posthoc_test

def fdr_correction(df, p_col='p_value', model_col='model', method='fdr_bh'):
    corrected_df = df.copy()
    reject, p_corrected, _, _ = multipletests(corrected_df[p_col], alpha=0.05, method=method)
    corrected_df['p_fdr'] = p_corrected
    corrected_df['reject'] = reject  
    return corrected_df

"""
Prepare Dataframe
"""
high_high_ids = high_high_ids['src_subject_id'].tolist()
high_low_ids = high_low_ids['src_subject_id'].tolist()
low_high_ids = low_high_ids['src_subject_id'].tolist()
low_low_ids = low_low_ids['src_subject_id'].tolist()

conditions = [
    abcd_all['src_subject_id'].isin(high_high_ids),
    abcd_all['src_subject_id'].isin(high_low_ids),
    abcd_all['src_subject_id'].isin(low_high_ids),
    abcd_all['src_subject_id'].isin(low_low_ids)]
values = [1, 2, 3, 4]
abcd_all['stability_subgroup'] = np.select(conditions, values, default=pd.NA)

abcd_train = abcd_all.loc[abcd_all['train_ids'] == 1].copy()
abcd_test = abcd_all.loc[abcd_all['test_ids'] == 1].copy()

"""
Train: CBCL 
"""
#baseline
aggressive_t0_train_anova, aggressive_t0_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_aggressive_t_t0', group='stability_subgroup', posthoc=True)
attention_t0_train_anova, attention_t0_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_attention_t_t0', group='stability_subgroup', posthoc=True)
rulebreak_t0_train_anova, rulebreak_t0_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_rulebreak_t_t0', group='stability_subgroup', posthoc=True)

# external_t0_train_anova, external_t0_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_external_t_t0', group='stability_subgroup', posthoc=True)
# total_t0_train_anova, total_t0_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_totprob_t_t0', group='stability_subgroup', posthoc=True)

#1-year follow-up
aggressive_t1_train_anova, aggressive_t1_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_aggressive_t_t1', group='stability_subgroup', posthoc=True)
attention_t1_train_anova, attention_t1_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_attention_t_t1', group='stability_subgroup', posthoc=True)
rulebreak_t1_train_anova, rulebreak_t1_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_rulebreak_t_t1', group='stability_subgroup', posthoc=True)

# external_t1_train_anova, external_t1_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_external_t_t1', group='stability_subgroup', posthoc=True)
# total_t1_train_anova, total_t1_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_totprob_t_t1', group='stability_subgroup', posthoc=True)

#2-year follow-up
aggressive_t2_train_anova, aggressive_t2_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_aggressive_t_t2', group='stability_subgroup', posthoc=True)
attention_t2_train_anova, attention_t2_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_attention_t_t2', group='stability_subgroup', posthoc=True)
rulebreak_t2_train_anova, rulebreak_t2_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_rulebreak_t_t2', group='stability_subgroup', posthoc=True)

# external_t2_train_anova, external_t2_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_external_t_t2', group='stability_subgroup', posthoc=True)
# total_t2_train_anova, total_t2_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_totprob_t_t2', group='stability_subgroup', posthoc=True)

#3-year follow-up
aggressive_t3_train_anova, aggressive_t3_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_aggressive_t_t3', group='stability_subgroup', posthoc=True)
attention_t3_train_anova, attention_t3_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_attention_t_t3', group='stability_subgroup', posthoc=True)
rulebreak_t3_train_anova, rulebreak_t3_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_rulebreak_t_t3', group='stability_subgroup', posthoc=True)

# external_t3_train_anova, external_t3_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_external_t_t3', group='stability_subgroup', posthoc=True)
# total_t3_train_anova, total_t3_train_posthoc = oneway_welch_anova(abcd_train, dv='cbcl_totprob_t_t3', group='stability_subgroup', posthoc=True)

"""
Test: CBCL 
"""
attention_t0_test_anova, attention_t0_test_posthoc = oneway_welch_anova(abcd_test, dv='cbcl_attention_t_t0', group='stability_subgroup', posthoc=True)
attention_t1_test_anova, attention_t1_test_posthoc = oneway_welch_anova(abcd_test, dv='cbcl_attention_t_t1', group='stability_subgroup', posthoc=True)
rulebreak_t1_test_anova, rulebreak_t1_test_posthoc = oneway_welch_anova(abcd_test, dv='cbcl_rulebreak_t_t1', group='stability_subgroup', posthoc=True)
attention_t2_test_anova, attention_t2_test_posthoc = oneway_welch_anova(abcd_test, dv='cbcl_attention_t_t2', group='stability_subgroup', posthoc=True)
attention_t3_test_anova, attention_t3_test_posthoc = oneway_welch_anova(abcd_test, dv='cbcl_attention_t_t3', group='stability_subgroup', posthoc=True)

"""
FDR Correction
""" 
results = {
    'model': ['attention_t0_test_anova', 'attention_t1_test_anova', 'rulebreak_t1_test_anova', 
              'attention_t2_test_anova', 'attention_t3_test_anova'],
    'p_value': [7.78386e-05, 2.04059e-07, 0.000253419, 
                5.22007e-07, 1.27487e-06]}

all_model_anova_test = fdr_correction(results)
all_model_anova_test = pd.DataFrame(all_model_anova_test)

"""
Export Files
""" 
save = [
    (all_model_anova_test, 'all_model_anova_test'), 
    #baseline train
    (aggressive_t0_train_anova, 'aggressive_t0_train_anova'), (aggressive_t0_train_posthoc, 'aggressive_t0_train_posthoc'), 
    (attention_t0_train_anova, 'attention_t0_train_anova'), (attention_t0_train_posthoc, 'attention_t0_train_posthoc'),
    (rulebreak_t0_train_anova, 'rulebreak_t0_train_anova'), (rulebreak_t0_train_posthoc, 'rulebreak_t0_train_posthoc'),  
    # (external_t0_train_anova, 'external_t0_train_anova'), (external_t0_train_posthoc, 'external_t0_train_posthoc'),  
    #1-year follow-up train
    (aggressive_t1_train_anova, 'aggressive_t1_train_anova'), (aggressive_t1_train_posthoc, 'aggressive_t1_train_posthoc'), 
    (attention_t1_train_anova, 'attention_t1_train_anova'), (attention_t1_train_posthoc, 'attention_t1_train_posthoc'),
    (rulebreak_t1_train_anova, 'rulebreak_t1_train_anova'), (rulebreak_t1_train_posthoc, 'rulebreak_t1_train_posthoc'), 
    # (external_t1_train_anova, 'external_t1_train_anova'), (external_t1_train_posthoc, 'external_t1_train_posthoc'),  
    #2-year follow-up train
    (aggressive_t2_train_anova, 'aggressive_t2_train_anova'), (aggressive_t2_train_posthoc, 'aggressive_t2_train_posthoc'), 
    (attention_t2_train_anova, 'attention_t2_train_anova'), (attention_t2_train_posthoc, 'attention_t2_train_posthoc'),
    (rulebreak_t2_train_anova, 'rulebreak_t2_train_anova'), (rulebreak_t2_train_posthoc, 'rulebreak_t2_train_posthoc'), 
    # (external_t2_train_anova, 'external_t2_train_anova'), (external_t2_train_posthoc, 'external_t2_train_posthoc'),  
    #3-year follow-up train
    (aggressive_t3_train_anova, 'aggressive_t3_train_anova'), (aggressive_t3_train_posthoc, 'aggressive_t3_train_posthoc'), 
    (attention_t3_train_anova, 'attention_t3_train_anova'), (attention_t3_train_posthoc, 'attention_t3_train_posthoc'),
    (rulebreak_t3_train_anova, 'rulebreak_t3_train_anova'), (rulebreak_t3_train_posthoc, 'rulebreak_t3_train_posthoc'),
    # (external_t3_train_anova, 'external_t3_train_anova'), (external_t3_train_posthoc, 'external_t3_train_posthoc'),   
    #test
    (attention_t0_test_anova, 'attention_t0_test_anova'), (attention_t0_test_posthoc, 'attention_t0_test_posthoc'),
    (attention_t1_test_anova, 'attention_t1_test_anova'), (attention_t1_test_posthoc, 'attention_t1_test_posthoc'),
    (rulebreak_t1_test_anova, 'rulebreak_t1_test_anova'), (rulebreak_t1_test_posthoc, 'rulebreak_t1_test_posthoc'), 
    (attention_t2_test_anova, 'attention_t2_test_anova'), (attention_t2_test_posthoc, 'attention_t2_test_posthoc'),
    (attention_t3_test_anova, 'attention_t3_test_anova'), (attention_t3_test_posthoc, 'attention_t3_test_posthoc')]

for df, name in save:
    df.to_csv(export_directory/f'{name}.csv', index=False)
    
    
    
    
    
    
    
    
