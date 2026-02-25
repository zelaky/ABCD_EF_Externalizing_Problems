#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Univariate Clinical Analyses Across the Train-Test Split 
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Exploring univariate associations between EF cluster membership and clinical diagnoses and symptoms in the training and testing set for the baseline and 2-year follow-up.

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
import scipy
from scipy.stats import ttest_ind, levene, fisher_exact, skew, kurtosis, mannwhitneyu
import statsmodels
from statsmodels.stats.multitest import multipletests
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import os

#load data
abcd_all = pd.read_csv(import_directory/'10_Analyses'/'abcd_all.csv')
abcd_singleton = pd.read_csv(import_directory/'10_Analyses'/'abcd_singleton.csv')
abcd_only = pd.read_csv(import_directory/'10_Analyses'/'abcd_only.csv')
abcd_rt = pd.read_csv(import_directory/'10_Analyses'/'abcd_rt.csv')

"""
Functions
"""
def skew_kurtosis(dictionary, numeric_vars, group_col='tasks_t0_km_2', skew_threshold=2, kurt_threshold=2):
    distribution_results = []
    for key, df in dictionary.items():
        for var in numeric_vars:
            if var not in df.columns:
                continue
            group_0 = df[df[group_col] == 0][var].dropna()
            group_1 = df[df[group_col] == 1][var].dropna()
            g0_skew_val = skew(group_0) if len(group_0) > 2 else np.nan
            g0_kurt_val = kurtosis(group_0) if len(group_0) > 2 else np.nan
            g1_skew_val = skew(group_1) if len(group_1) > 2 else np.nan
            g1_kurt_val = kurtosis(group_1) if len(group_1) > 2 else np.nan
            distribution_results.append({
                'dataframe': key,
                'variable': var,
                'group_0_n': len(group_0),
                'group_0_skew_value': g0_skew_val,
                'group_0_skew': abs(g0_skew_val) > skew_threshold if not np.isnan(g0_skew_val) else np.nan,
                'group_0_kurtosis_value': g0_kurt_val,
                'group_0_kurtosis': abs(g0_kurt_val) > kurt_threshold if not np.isnan(g0_kurt_val) else np.nan,
                'group_1_n': len(group_1),
                'group_1_skew_value': g1_skew_val,
                'group_1_skew': abs(g1_skew_val) > skew_threshold if not np.isnan(g1_skew_val) else np.nan,
                'group_1_kurtosis_value': g1_kurt_val,
                'group_1_kurtosis': abs(g1_kurt_val) > kurt_threshold if not np.isnan(g1_kurt_val) else np.nan
            })
    distribution_df = pd.DataFrame(distribution_results)
    mask = (
        (distribution_df['group_0_skew'] == True) |
        (distribution_df['group_1_skew'] == True) |
        (distribution_df['group_0_kurtosis'] == True) |
        (distribution_df['group_1_kurtosis'] == True)
    )
    non_normal_df = distribution_df.loc[mask, ['dataframe', 'variable']].reset_index(drop=True)
    return distribution_df, non_normal_df

def train_test(df, test_func, variables, train_col='train_ids', test_col='test_ids', group_col='tasks_t0_km_2', alpha=0.05):
    train_df = df[df[train_col] == 1].copy()
    test_df = df[df[test_col] == 1].copy()
    train_results = test_func(train_df, variables, group_col=group_col)
    sig_vars = train_results.loc[train_results['p_value'] < alpha, 'variable'].tolist()
    if not sig_vars:
        test_results = pd.DataFrame(columns=train_results.columns)
    else:
        test_results = test_func(test_df, sig_vars, group_col=group_col)
    return train_results, test_results

def mann_whitney_u(df, non_normal_vars, group_col='tasks_t0_km_2'):
    results = []
    for var in non_normal_vars:
        if var not in df.columns:
            continue
        group_0 = df[df[group_col] == 0][var].dropna()
        group_1 = df[df[group_col] == 1][var].dropna()
        if len(group_0) > 1 and len(group_1) > 1:
            u_stat, u_p = mannwhitneyu(group_0, group_1, alternative='two-sided')
            g0_n, g1_n = len(group_0), len(group_1)
            g0_sd, g1_sd = group_0.std(ddof=1), group_1.std(ddof=1)
            r_rb = 1 - (2 * u_stat) / (g0_n * g1_n)
            results.append({
                'variable': var,
                'group_0_n': g0_n,
                'group_0_mean': group_0.mean(),
                'group_0_sd': g0_sd,
                'group_1_n': g1_n,
                'group_1_mean': group_1.mean(),
                'group_1_sd': g1_sd,
                'u_value': u_stat,
                'p_value': u_p,
                'rank_biserial_r': r_rb
            })
    return pd.DataFrame(results)

def independent_t(df, normal_vars, group_col='tasks_t0_km_2'):
    results = []
    for var in normal_vars:
        if var not in df.columns:
            continue
        group_0 = df[df[group_col] == 0][var].dropna()
        group_1 = df[df[group_col] == 1][var].dropna()
        if len(group_0) < 2 or len(group_1) < 2:
            continue
        levene_stat, levene_p = levene(group_0, group_1)
        equal_var = levene_p >= 0.05
        t_stat, p_val = ttest_ind(group_0, group_1, equal_var=equal_var)
        results.append({
            'variable': var,
            'p_value': p_val,
            't_value': t_stat,
            'levene_val': levene_stat,
            'levene_pval': levene_p,
            'equal_var': equal_var,
            'group_0_n': len(group_0),
            'group_0_mean': group_0.mean(),
            'group_0_sd': group_0.std(ddof=1),
            'group_1_n': len(group_1),
            'group_1_mean': group_1.mean(),
            'group_1_sd': group_1.std(ddof=1)
        })
    return pd.DataFrame(results)

def fishers_exact(df, categorical_vars, group_col='tasks_t0_km_2'):
    results = []
    for var in categorical_vars:
        if var not in df.columns:
            continue
        contingency = pd.crosstab(df[group_col], df[var])
        if contingency.shape != (2, 2):
            continue
        oddsratio, p_val = fisher_exact(contingency)
        results.append({
            'variable': var,
            'group00': contingency.iloc[0, 0],
            'group01': contingency.iloc[0, 1],
            'group10': contingency.iloc[1, 0],
            'group11': contingency.iloc[1, 1],
            'p_value': p_val,
            'odds_ratio': oddsratio
        })
    return pd.DataFrame(results)

def fdr_correction(dfs, p_value_cols, alpha=0.05, method='fdr_bh', fdr_col_name='fdr_p_value', reject_col_name='fdr_reject'):
    p_values = []
    lengths = []
    for df, col in zip(dfs, p_value_cols):
        p_values.extend(df[col].values)
        lengths.append(len(df))
    p_values = pd.Series(p_values)
    reject, fdr_p_values, _, _ = multipletests(p_values, alpha=alpha, method=method)
    fdr_p_values = pd.Series(fdr_p_values)
    start = 0
    for i, df in enumerate(dfs):
        end = start + lengths[i]
        df[fdr_col_name] = fdr_p_values[start:end].values
        df[reject_col_name] = df[fdr_col_name] < alpha
        start = end

def align_labels(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize agreement
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned = [mapping[label] for label in pred_labels]
    return aligned

"""
Prepare Dataframes
"""
abcd_all_t0_check = abcd_all["tasks_t0_km_2"].value_counts()
abcd_all_t2_check = abcd_all["tasks_t2_km_2"].value_counts()
abcd_rt_t0_check = abcd_rt["tasks_t0_km_2"].value_counts()
abcd_rt_t2_check = abcd_rt["tasks_t2_km_2"].value_counts()
abcd_rt["tasks_t0_km_2"] = 1 - abcd_rt["tasks_t0_km_2"]

abcd_singleton_t0_check = abcd_singleton["tasks_t0_km_2"].value_counts()
abcd_singleton_t2_check = abcd_singleton["tasks_t2_km_2"].value_counts()
abcd_singleton["tasks_t0_km_2"] = 1 - abcd_singleton["tasks_t0_km_2"]
abcd_singleton["tasks_t2_km_2"] = 1 - abcd_singleton["tasks_t2_km_2"]

#similarity score
full_dfs = {
    'all': abcd_all,
    'singleton': abcd_singleton,
    'only': abcd_only,
    'rt': abcd_rt
}

column_pairs = [('tasks_t0_km_2', 'tasks_t2_km_2')]

for key, df in full_dfs.items():
    for col1, col2 in column_pairs:
        aligned = align_labels(df[col1], df[col2])
        
variable_groups = {
    'ksads_cd_present_rc': ['ksads_cd_present_child', 'ksads_cd_present_adolescent'],
    'ksads_adhd_present_rc': ['ksads_adhd_other', 'ksads_adhd_present']
}

timepoints = ['t0', 't1', 't2', 't3']

for df_name, df in full_dfs.items():
    for new_var, components in variable_groups.items():
        for tp in timepoints:
            df[f'{new_var}_{tp}'] = (
                df[[f'{comp}_{tp}' for comp in components]] == 1
            ).any(axis=1).astype(int)

"""
Frequency and Descriptive Dtatistics
"""
#baseline
higher_ef_t0 = abcd_all[abcd_all['tasks_t0_km_2']==0]
higher_ef_t0['cbcl_aggressive_t_t0'].mean() #52.2476938845234
higher_ef_t0['cbcl_attention_t_t0'].mean() #52.921762897164335
higher_ef_t0['cbcl_rulebreak_t_t0'].mean() #52.06047147249744
higher_ef_t0['cbcl_external_t_t0'].mean() #44.563717116501536
higher_ef_t0['cbcl_aggressive_t_t0'].std() #4.680255568357412
higher_ef_t0['cbcl_attention_t_t0'].std() #4.983536933087662
higher_ef_t0['cbcl_rulebreak_t_t0'].std() #3.9171310203268046
higher_ef_t0['cbcl_external_t_t0'].std() #9.583675934359603
(higher_ef_t0['ksads_dmdd_t0'] == 1).sum() #1
(higher_ef_t0['ksads_cd_present_rc_t0'] == 1).sum() #64
(higher_ef_t0['ksads_odd_present_t0'] == 1).sum() #140
(higher_ef_t0['ksads_adhd_present_rc_t0'] == 1).sum() #166

lower_ef_t0 = abcd_all[abcd_all['tasks_t0_km_2']==1]
lower_ef_t0['cbcl_aggressive_t_t0'].mean() #52.481351981351985
lower_ef_t0['cbcl_attention_t_t0'].mean() #53.695027195027194
lower_ef_t0['cbcl_rulebreak_t_t0'].mean() #52.38500388500388
lower_ef_t0['cbcl_external_t_t0'].mean() #44.934731934731936
lower_ef_t0['cbcl_aggressive_t_t0'].std() #5.093322823693801
lower_ef_t0['cbcl_attention_t_t0'].std() #5.979937839766291
lower_ef_t0['cbcl_rulebreak_t_t0'].std() #4.413149500678262
lower_ef_t0['cbcl_external_t_t0'].std() #9.932282081549507
(lower_ef_t0['ksads_dmdd_t0'] == 1).sum() #2
(lower_ef_t0['ksads_cd_present_rc_t0'] == 1).sum() #46
(lower_ef_t0['ksads_odd_present_t0'] == 1).sum() #124
(lower_ef_t0['ksads_adhd_present_rc_t0'] == 1).sum() #183

# 2-year follow-up
higher_ef_t2 = abcd_all[abcd_all['tasks_t2_km_2']==0]
higher_ef_t2['cbcl_aggressive_t_t2'].mean() #51.87001620745543
higher_ef_t2['cbcl_attention_t_t2'].mean() #52.63500810372771
higher_ef_t2['cbcl_rulebreak_t_t2'].mean() #51.555267423014584
higher_ef_t2['cbcl_external_t_t2'].mean() #43.366288492706644
higher_ef_t2['cbcl_aggressive_t_t2'].std() #4.192857110076871
higher_ef_t2['cbcl_attention_t_t2'].std() #4.7036364193842175
higher_ef_t2['cbcl_rulebreak_t_t2'].std() #3.346496072194158
higher_ef_t2['cbcl_external_t_t2'].std() #9.057642436753367
(higher_ef_t2['ksads_dmdd_t2'] == 1).sum() #0
(higher_ef_t2['ksads_cd_present_rc_t2'] == 1).sum() #33
(higher_ef_t2['ksads_odd_present_t2'] == 1).sum() #131
(higher_ef_t2['ksads_adhd_present_rc_t2'] == 1).sum() #131

lower_ef_t2 = abcd_all[abcd_all['tasks_t2_km_2']==1]
lower_ef_t2['cbcl_aggressive_t_t2'].mean() #52.347717842323654
lower_ef_t2['cbcl_attention_t_t2'].mean() #53.53278008298755
lower_ef_t2['cbcl_rulebreak_t_t2'].mean() #51.90871369294606
lower_ef_t2['cbcl_external_t_t2'].mean() #44.30497925311203
lower_ef_t2['cbcl_aggressive_t_t2'].std() #5.0660482972424274
lower_ef_t2['cbcl_attention_t_t2'].std() #5.650169269532021
lower_ef_t2['cbcl_rulebreak_t_t2'].std() #3.905160289096984
lower_ef_t2['cbcl_external_t_t2'].std() #9.699487095539986
(lower_ef_t2['ksads_dmdd_t2'] == 1).sum() #1
(lower_ef_t2['ksads_cd_present_rc_t2'] == 1).sum() #36
(lower_ef_t2['ksads_odd_present_t2'] == 1).sum() #118
(lower_ef_t2['ksads_adhd_present_rc_t2'] == 1).sum() #140

"""
Check Distributions
"""
numeric_vars = [col for col in full_dfs['all'] if col.startswith('cbcl_')]
distribution_cluster_t0, non_normal_t0 = skew_kurtosis(dictionary=full_dfs, numeric_vars=numeric_vars, group_col='tasks_t0_km_2')
distribution_cluster_t2, non_normal_t2 = skew_kurtosis(dictionary=full_dfs, numeric_vars=numeric_vars, group_col='tasks_t2_km_2')

"""
Baseline Clusters
"""
categorical_t0_vars = ['ksads_dmdd_t0', 'ksads_adhd_present_rc_t0', 'ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 
                    'ksads_dmdd_t1', 'ksads_adhd_present_rc_t1', 'ksads_odd_present_t1', 'ksads_cd_present_rc_t1',
                    'ksads_dmdd_t2', 'ksads_adhd_present_rc_t2', 'ksads_odd_present_t2', 'ksads_cd_present_rc_t2',
                    'ksads_dmdd_t3', 'ksads_adhd_present_rc_t3', 'ksads_odd_present_t3', 'ksads_cd_present_rc_t3'
                    ]  

non_normal_t0_vars = ['cbcl_aggressive_t_t0', 'cbcl_rulebreak_t_t0', 'cbcl_attention_t_t0',
                   'cbcl_aggressive_t_t1', 'cbcl_rulebreak_t_t1', 'cbcl_attention_t_t1',
                   'cbcl_aggressive_t_t2', 'cbcl_rulebreak_t_t2', 'cbcl_attention_t_t2',
                   'cbcl_aggressive_t_t3', 'cbcl_rulebreak_t_t3', 'cbcl_attention_t_t3'
                   ] 
 
normal_t0_vars = ['cbcl_external_t_t0', 'cbcl_external_t_t1','cbcl_external_t_t2', 'cbcl_external_t_t3']


results_t0 = {}

for df_name, df in full_dfs.items():
    mann_whitney_t0_train, mann_whitney_t0_test = train_test(df, mann_whitney_u, non_normal_t0_vars, 
                                                             train_col='train_ids', test_col='test_ids',
                                                             group_col='tasks_t0_km_2')
    
    t_t0_train, t_t0_test = train_test(df, independent_t, normal_t0_vars, 
                                       train_col='train_ids', test_col='test_ids',
                                       group_col='tasks_t0_km_2')
    
    fishers_exact_t0_train, fishers_exact_t0_test = train_test(df, fishers_exact, categorical_t0_vars, 
                                                               train_col='train_ids', test_col='test_ids',
                                                               group_col='tasks_t0_km_2')
    
    results_t0[df_name] = {
        'mann_whitney_t0_train': mann_whitney_t0_train,
        'mann_whitney_t0_test': mann_whitney_t0_test,
        't_t0_train': t_t0_train,
        't_t0_test': t_t0_test,
        'fishers_exact_t0_train': fishers_exact_t0_train,
        'fishers_exact_t0_test': fishers_exact_t0_test
    }
    
#all
all_t0_results = results_t0['all']
all_t0_results = {
    key.replace("t0", "all_t0"): df
    for key, df in all_t0_results.items()
}

dfs = []
p_value_cols = []
for name, df in all_t0_results.items():
    if name.endswith('_test') and not df.empty:
        dfs.append(df)
        if 'p_value' in df.columns:
            p_value_cols.append('p_value')
        else:
            raise ValueError(f"No p-value column found in {name}")
fdr_correction(dfs, p_value_cols)

#singleton
singleton_t0_results = results_t0['singleton']
singleton_t0_results = {
    key.replace("t0", "singleton_t0"): df
    for key, df in singleton_t0_results.items()
}

dfs = []
p_value_cols = []
for name, df in singleton_t0_results.items():
    if name.endswith('_test') and not df.empty:
        dfs.append(df)
        if 'p_value' in df.columns:
            p_value_cols.append('p_value')
        else:
            raise ValueError(f"No p-value column found in {name}")
fdr_correction(dfs, p_value_cols)

#only
only_t0_results = results_t0['only']
only_t0_results = {
    key.replace("t0", "only_t0"): df
    for key, df in only_t0_results.items()
}

dfs = []
p_value_cols = []
for name, df in only_t0_results.items():
    if name.endswith('_test') and not df.empty:
        dfs.append(df)
        if 'p_value' in df.columns:
            p_value_cols.append('p_value')
        else:
            raise ValueError(f"No p-value column found in {name}")
fdr_correction(dfs, p_value_cols)

#rt
rt_t0_results = results_t0['rt']
rt_t0_results = {
    key.replace("t0", "rt_t0"): df
    for key, df in rt_t0_results.items()
}

dfs = []
p_value_cols = []
for name, df in rt_t0_results.items():
    if name.endswith('_test') and not df.empty:
        dfs.append(df)
        if 'p_value' in df.columns:
            p_value_cols.append('p_value')
        else:
            raise ValueError(f"No p-value column found in {name}")
fdr_correction(dfs, p_value_cols)

"""
2-year Follow-up Clusters
"""
categorical_t2_vars = ['ksads_dmdd_t0', 'ksads_adhd_present_rc_t0', 'ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 
                       'ksads_dmdd_t1', 'ksads_adhd_present_rc_t1', 'ksads_odd_present_t1', 'ksads_cd_present_rc_t1', 
                       'ksads_dmdd_t2', 'ksads_adhd_present_rc_t2', 'ksads_odd_present_t2', 'ksads_cd_present_rc_t2',
                       'ksads_dmdd_t3', 'ksads_adhd_present_rc_t3', 'ksads_odd_present_t3', 'ksads_cd_present_rc_t3']  

non_normal_t2_vars = ['cbcl_aggressive_t_t0', 'cbcl_rulebreak_t_t0', 'cbcl_attention_t_t0',
                      'cbcl_aggressive_t_t1', 'cbcl_rulebreak_t_t1', 'cbcl_attention_t_t1',
                      'cbcl_aggressive_t_t2', 'cbcl_rulebreak_t_t2', 'cbcl_attention_t_t2',
                      'cbcl_aggressive_t_t3', 'cbcl_rulebreak_t_t3', 'cbcl_attention_t_t3']  

normal_t2_vars = ['cbcl_external_t_t0', 'cbcl_external_t_t1','cbcl_external_t_t2', 'cbcl_external_t_t3']

results_t2 = {}

for df_name, df in full_dfs.items():
    mann_whitney_t2_train, mann_whitney_t2_test = train_test(df, mann_whitney_u, non_normal_t2_vars, 
                                                             train_col='train_ids', test_col='test_ids',
                                                             group_col='tasks_t2_km_2')
    
    t_t2_train, t_t2_test = train_test(df, independent_t, normal_t2_vars, 
                                       train_col='train_ids', test_col='test_ids',
                                       group_col='tasks_t2_km_2')
    
    fishers_exact_t2_train, fishers_exact_t2_test = train_test(df, fishers_exact, categorical_t2_vars, 
                                                               train_col='train_ids', test_col='test_ids',
                                                               group_col='tasks_t2_km_2')
    
    results_t2[df_name] = {
        'mann_whitney_t2_train': mann_whitney_t2_train,
        'mann_whitney_t2_test': mann_whitney_t2_test,
        't_t2_train': t_t2_train,
        't_t2_test': t_t2_test,
        'fishers_exact_t2_train': fishers_exact_t2_train,
        'fishers_exact_t2_test': fishers_exact_t2_test
    }

#all
all_t2_results = results_t2['all']
all_t2_results = {
    key.replace("t2", "all_t2"): df
    for key, df in all_t2_results.items()
}

dfs = []
p_value_cols = []
for name, df in all_t2_results.items():
    if name.endswith('_test') and not df.empty:
        dfs.append(df)
        if 'p_value' in df.columns:
            p_value_cols.append('p_value')
        else:
            raise ValueError(f"No p-value column found in {name}")
fdr_correction(dfs, p_value_cols)

#singleton
singleton_t2_results = results_t2['singleton']
singleton_t2_results = {
    key.replace("t2", "singleton_t2"): df
    for key, df in singleton_t2_results.items()
}

dfs = []
p_value_cols = []
for name, df in singleton_t2_results.items():
    if name.endswith('_test') and not df.empty:
        dfs.append(df)
        if 'p_value' in df.columns:
            p_value_cols.append('p_value')
        else:
            raise ValueError(f"No p-value column found in {name}")
fdr_correction(dfs, p_value_cols)

#only
only_t2_results = results_t2['only']
only_t2_results = {
    key.replace("t2", "only_t2"): df
    for key, df in only_t2_results.items()
}

dfs = []
p_value_cols = []
for name, df in only_t2_results.items():
    if name.endswith('_test') and not df.empty:
        dfs.append(df)
        if 'p_value' in df.columns:
            p_value_cols.append('p_value')
        else:
            raise ValueError(f"No p-value column found in {name}")
fdr_correction(dfs, p_value_cols)

#rt
rt_t2_results = results_t2['rt']
rt_t2_results = {
    key.replace("t2", "rt_t2"): df
    for key, df in rt_t2_results.items()
}

dfs = []
p_value_cols = []
for name, df in rt_t2_results.items():
    if name.endswith('_test') and not df.empty:
        dfs.append(df)
        if 'p_value' in df.columns:
            p_value_cols.append('p_value')
        else:
            raise ValueError(f"No p-value column found in {name}")
fdr_correction(dfs, p_value_cols)

"""
Export Files
"""
#all
for name, df in all_t0_results.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
    
for name, df in all_t2_results.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)

#singleton
for name, df in singleton_t0_results.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
    
for name, df in singleton_t2_results.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
    
#only
for name, df in only_t0_results.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
    
for name, df in only_t2_results.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
    
#rt
for name, df in rt_t0_results.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
    
for name, df in rt_t2_results.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)