#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Cluster Characterization 
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Exploring and comparing frequencies and descriptives of task, clinical, and demographic variables for cluster solutions. 

Packages: 
- Spyder version: 6.07
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
    
Baseline Best Subset
- all: sst_t0_crgo_stdrt, sst_t0_crlg_rate, nback_t0_c0b_rate, nback_t0_c0b_mrt, nback_t0_c2b_rate, nback_t0_c2b_stdrt
- singleton: sst_t0_nrgo_rate, sst_t0_crlg_rate, nback_t0_c0b_rate, nback_t0_c0b_mrt, nback_t0_c2b_rate, nback_t0_c2b_stdrt, 
- only: sst_t0_crgo_stdrt, sst_t0_crlg_rate, nback_t0_c0b_rate, nback_t0_c0b_mrt, nback_t0_c2b_rate, nback_t0_c2b_stdrt
- rt: sst_t0_crgo_stdrt, sst_t0_crlg_rate, nback_t0_c0b_rate, nback_t0_c0b_mrt, nback_t0_c2b_rate, nback_t0_c2b_stdrt

2-year Follow-up Best Subset
- all: sst_t2_crgo_stdrt, nback_t2_c0b_rate, nback_t2_c0b_mrt, nback_t2_c0b_stdrt, nback_t2_c2b_rate, nback_t2_c2b_mrt, nback_t2_c2b_stdrt
- singleton: sst_t2_crgo_stdrt, nback_t2_c0b_rate, nback_t2_c0b_mrt, nback_t2_c0b_stdrt, nback_t2_c2b_rate, nback_t2_c2b_mrt, nback_t2_c2b_stdrt
- only: sst_t2_crgo_stdrt, nback_t2_c0b_rate, nback_t2_c0b_mrt, nback_t2_c0b_stdrt, nback_t2_c2b_rate, nback_t2_c2b_mrt, nback_t2_c2b_stdrt
- rt: sst_t2_crgo_stdrt, nback_t2_c0b_rate, nback_t2_c0b_mrt, nback_t2_c0b_stdrt, nback_t2_c2b_rate, nback_t2_c2b_mrt, nback_t2_c2b_stdrt
"""
#import 
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os

#statistical suite
from scipy.stats import ttest_ind, levene, fisher_exact, chi2_contingency, skew, kurtosis, mannwhitneyu
from statsmodels.stats.multitest import multipletests

#load data
abcd_all = pd.read_csv(import_directory/'10_Analyses'/'abcd_all.csv')
abcd_singleton = pd.read_csv(import_directory/'10_Analyses'/'abcd_singleton.csv')
abcd_only = pd.read_csv(import_directory/'10_Analyses'/'abcd_only.csv')
abcd_rt = pd.read_csv(import_directory/'10_Analyses'/'abcd_rt.csv')
abcd_split = pd.read_csv(import_directory/'5_Split'/'demo_t0_split.csv')
abcd_iq = pd.read_csv(import_directory/'10_Analyses'/'nc_y_wisc.csv')

abcd_all['tasks_t0_km_2'].value_counts()
abcd_all['tasks_t2_km_2'].value_counts()

abcd_singleton['tasks_t0_km_2'].value_counts()
abcd_singleton['tasks_t2_km_2'].value_counts()

abcd_only['tasks_t0_km_2'].value_counts()
abcd_only['tasks_t2_km_2'].value_counts()

abcd_rt['tasks_t0_km_2'].value_counts()
abcd_rt['tasks_t2_km_2'].value_counts()

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
        n0, n1 = len(group_0), len(group_1)
        sd0, sd1 = group_0.std(ddof=1), group_1.std(ddof=1)
        pooled_sd = np.sqrt(((n0 - 1)*sd0**2 + (n1 - 1)*sd1**2) / (n0 + n1 - 2))
        cohens_d = (group_0.mean() - group_1.mean()) / pooled_sd
        
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
            'group_1_sd': group_1.std(ddof=1),
            'cohens_d': cohens_d
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

def chi_square(df, categorical_vars, group_col='tasks_t0_km_2'):
    results = []
    for var in categorical_vars:
        if var not in df.columns:
            continue
        contingency = pd.crosstab(df[var], df[group_col])
        if contingency.shape[0] == 0 or contingency.shape[1] < 2:
            continue
        chi2_stat, p_val, dof, expected = chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        k = min(contingency.shape)
        if contingency.shape == (2, 2):
            effect_size = np.sqrt(chi2_stat / n)  #Phi
            effect_label = 'phi'
        else:
            effect_size = np.sqrt(chi2_stat / (n * (k - 1)))  #Cramér's V
            effect_label = "cramers_v"
        
        results.append({
            'variable': var,
            'chisq_value': chi2_stat,
            'p_value': p_val,
            'dof': dof,
            'effect_size': effect_size,
            'effect_type': effect_label,
            'observed_counts': contingency.values,
            'contingency_table': contingency
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
        
def cluster_category_counts(df, cluster_col, category_col, export_dir=None, prefix=""):
    merged = pd.DataFrame()
    for cluster in sorted(df[cluster_col].dropna().unique()):
        subset = df[df[cluster_col] == cluster]
        train_counts = (subset[subset['train_ids'] == 1][category_col].value_counts().reset_index())
        train_counts.columns = [category_col, f"c{cluster}_train"]
        test_counts = (subset[subset['test_ids'] == 1][category_col].value_counts().reset_index())
        test_counts.columns = [category_col, f"c{cluster}_test"]
        if merged.empty:
            merged = train_counts.merge(test_counts, on=category_col, how="outer")
        else:
            merged = (
                merged.merge(train_counts, on=category_col, how="outer")
                       .merge(test_counts, on=category_col, how="outer"))
    count_cols = [col for col in merged.columns if col != category_col]
    merged[count_cols] = merged[count_cols].fillna(0).astype(int)
    if export_dir:
        filename = f"{prefix}_{category_col}_counts.csv"
        merged.to_csv(export_dir / filename, index=False)
    return merged

"""
Preparing Dataframes
"""
#ids
all_ids = abcd_split[abcd_split['sample_ids'] == 1]
all_ids = all_ids['src_subject_id'].tolist()
singleton_ids = abcd_singleton['src_subject_id'].tolist()

all_iq = abcd_iq[abcd_iq['src_subject_id'].isin(all_ids)]
all_iq = all_iq[['src_subject_id', 'pea_wiscv_tss']]
singleton_iq = abcd_iq[abcd_iq['src_subject_id'].isin(singleton_ids)]
singleton_iq = singleton_iq[['src_subject_id', 'pea_wiscv_tss']]

abcd_all = pd.merge(abcd_all, all_iq, on='src_subject_id', how='outer')
abcd_singleton = pd.merge(abcd_singleton, singleton_iq, on='src_subject_id', how='outer')
abcd_only = pd.merge(abcd_only, all_iq, on='src_subject_id', how='outer')
abcd_rt = pd.merge(abcd_rt, all_iq, on='src_subject_id', how='outer')

del all_ids, singleton_ids, all_iq, singleton_iq

#merge onto baseline dataframe
full_dfs = {
    'all': abcd_all,
    'singleton': abcd_singleton,
    'only': abcd_only,
    'rt': abcd_rt
}

t0_numeric_vars = ['interview_age_t0', 'pea_wiscv_tss',
                   #best subset 
                   'sst_t0_crgo_stdrt', 'sst_t0_crlg_rate','nback_t0_c0b_rate', 
                   'nback_t0_c0b_mrt', 'nback_t0_c2b_rate', 'nback_t0_c2b_stdrt', 
                   'sst_t0_nrgo_rate',
                   #other
                   'sst_t0_crgo_rate', 'sst_t0_crgo_mrt', 'sst_t0_crlg_mrt', 
                   'sst_t0_crlg_stdrt', 'sst_t0_incrgo_rate', 'sst_t0_incrgo_mrt', 
                   'sst_t0_incrgo_stdrt', 'sst_t0_incrlg_rate', 'sst_t0_crs_rate', 
                   'sst_t0_incrs_rate', 'sst_t0_incrs_mrt', 'sst_t0_incrs_stdrt', 
                   'sst_t0_ssds_rate', 'sst_t0_mssd', 'sst_t0_mssrt', 
                   'sst_t0_issrt', 'nback_t0_c2bpf_rate', 'nback_t0_c2bpf_mrt', 
                   'nback_t0_c2bpf_stdrt', 'nback_t0_c2bnf_rate', 'nback_t0_c2bnf_mrt', 
                   'nback_t0_c2bnf_stdrt', 'nback_t0_c2bp_rate', 'nback_t0_c2bp_mrt', 
                   'nback_t0_c2bp_stdrt', 'nback_t0_c0bpf_rate', 'nback_t0_c0bpf_mrt', 
                   'nback_t0_c0bpf_stdrt', 'nback_t0_c0bnf_rate', 'nback_t0_c0bnf_mrt', 
                   'nback_t0_c0bnf_stdrt', 'nback_t0_c0bngf_rate', 'nback_t0_c0bngf_mrt', 
                   'nback_t0_c0bngf_stdrt', 'nback_t0_c0bp_rate', 'nback_t0_c0bp_mrt', 
                   'nback_t0_c0bp_stdrt', 'nback_t0_c2bngf_rate', 'nback_t0_c2bngf_mrt', 
                   'nback_t0_c2bngf_stdrt', 'nback_t0_c0b_stdrt', 'nback_t0_c2b_mrt'
                   ]

t2_numeric_vars = ['interview_age_t2', 'pea_wiscv_tss',
                   #best subset 
                   'sst_t2_crgo_stdrt', 'nback_t2_c0b_rate', 
                   'nback_t2_c0b_mrt', 'nback_t2_c0b_stdrt', 'nback_t2_c2b_rate', 
                   'nback_t2_c2b_mrt', 'nback_t2_c2b_stdrt', 'sst_t2_nrgo_rate',
                   #other
                   'sst_t2_crgo_rate', 'sst_t2_crgo_mrt', 'sst_t2_crlg_mrt', 
                   'sst_t2_crlg_stdrt', 'sst_t2_incrgo_rate', 'sst_t2_incrgo_mrt', 
                   'sst_t2_incrgo_stdrt', 'sst_t2_incrlg_rate', 'sst_t2_crs_rate', 
                   'sst_t2_incrs_rate', 'sst_t2_incrs_mrt', 'sst_t2_incrs_stdrt', 
                   'sst_t2_ssds_rate', 'sst_t2_mssd', 'sst_t2_mssrt', 
                   'sst_t2_issrt', 'nback_t2_c2bpf_rate', 'nback_t2_c2bpf_mrt', 
                   'nback_t2_c2bpf_stdrt', 'nback_t2_c2bnf_rate', 'nback_t2_c2bnf_mrt', 
                   'nback_t2_c2bnf_stdrt', 'nback_t2_c2bp_rate', 'nback_t2_c2bp_mrt', 
                   'nback_t2_c2bp_stdrt', 'nback_t2_c0bpf_rate', 'nback_t2_c0bpf_mrt', 
                   'nback_t2_c0bpf_stdrt', 'nback_t2_c0bnf_rate', 'nback_t2_c0bnf_mrt', 
                   'nback_t2_c0bnf_stdrt', 'nback_t2_c0bngf_rate', 'nback_t2_c0bngf_mrt', 
                   'nback_t2_c0bngf_stdrt', 'nback_t2_c0bp_rate', 'nback_t2_c0bp_mrt', 
                   'nback_t2_c0bp_stdrt', 'nback_t2_c2bngf_rate', 'nback_t2_c2bngf_mrt', 
                   'nback_t2_c2bngf_stdrt'
                   ]
"""
Full Sample Means and SDs
"""
task_waves = {
    't0': {'prefixes': ('nback_t0', 'sst_t0'), 'cluster_col': 'tasks_t0_km_2', 'demo_cols': ['pea_wiscv_tss','interview_age_t0']},
    't2': {'prefixes': ('nback_t2', 'sst_t2'), 'cluster_col': 'tasks_t2_km_2', 'demo_cols': ['interview_age_t0']}
}

means_dict = {}
sd_dict = {}

for df_key, df in full_dfs.items():
    means_dict[df_key] = {}
    sd_dict[df_key] = {}
    
    for wave, info in task_waves.items():
        task_cols = [col for col in df.columns if col.startswith(info['prefixes'])]
        descriptive_cols = task_cols + info['demo_cols']
        
        cluster_col = info['cluster_col']
        means_dict[df_key][wave] = {}
        sd_dict[df_key][wave] = {}
        
        for cluster_val in [0, 1]:
            subset = df.loc[df[cluster_col] == cluster_val, descriptive_cols]
            mean_df = subset.mean().reset_index(drop=False).rename(columns={0: 'mean'})
            sd_df = subset.std().reset_index(drop=False).rename(columns={0: 'sd'})
            means_dict[df_key][wave][f'c{cluster_val}'] = mean_df
            sd_dict[df_key][wave][f'c{cluster_val}'] = sd_df


for df_key, wave_dict in means_dict.items():
        for wave, cluster_dict in wave_dict.items():
            for cluster_val, df_save in cluster_dict.items():
                filename = f"{df_key}_{wave}_{cluster_val}_mean.csv"
                filepath = os.path.join(export_directory, filename)
                df_save.to_csv(filepath, index=False)
                
for df_key, wave_dict in sd_dict.items():
    for wave, cluster_dict in wave_dict.items():
        for cluster_val, df_save in cluster_dict.items():
            filename = f"{df_key}_{wave}_{cluster_val}_sd.csv"
            filepath = os.path.join(export_directory, filename)
            df_save.to_csv(filepath, index=False)

"""
Cluster Distributions 
"""
#check variable normality
distribution_cluster_t0, non_normal_t0 = skew_kurtosis(dictionary=full_dfs, numeric_vars=t0_numeric_vars, group_col='tasks_t0_km_2')
distribution_cluster_t2, non_normal_t2 = skew_kurtosis(dictionary=full_dfs, numeric_vars=t2_numeric_vars, group_col='tasks_t2_km_2')

#check categorical levels
#site
site_all_counts = cluster_category_counts(df=abcd_all, cluster_col='tasks_t0_km_2', 
                                          category_col='site_id_l', export_dir=export_directory,
                                          prefix='site_all')
site_singleton_counts = cluster_category_counts(df=abcd_singleton, cluster_col='tasks_t0_km_2', 
                                                category_col='site_id_l', export_dir=export_directory,
                                                prefix='site_singleton')
#gender
gender_all_counts = cluster_category_counts(df=abcd_all, cluster_col='tasks_t0_km_2', 
                                          category_col='demo_gender_id_v2', export_dir=export_directory,
                                          prefix='gender_all')
gender_singleton_counts = cluster_category_counts(df=abcd_singleton, cluster_col='tasks_t0_km_2', 
                                                category_col='demo_gender_id_v2', export_dir=export_directory,
                                                prefix='gender_singleton')
#education
education_all_counts = cluster_category_counts(df=abcd_all, cluster_col='tasks_t0_km_2', 
                                          category_col='demo_ed_v2', export_dir=export_directory,
                                          prefix='education_all')
education_singleton_counts = cluster_category_counts(df=abcd_singleton, cluster_col='tasks_t0_km_2', 
                                                category_col='demo_ed_v2', export_dir=export_directory,
                                                prefix='education_singleton')

#income
income_all_counts = cluster_category_counts(df=abcd_all, cluster_col='tasks_t0_km_2', 
                                          category_col='demo_comb_income_v2', export_dir=export_directory,
                                          prefix='income_all')
income_singleton_counts = cluster_category_counts(df=abcd_singleton, cluster_col='tasks_t0_km_2', 
                                                category_col='demo_comb_income_v2', export_dir=export_directory,
                                                prefix='income_singleton')

#race
race_all_counts = cluster_category_counts(df=abcd_all, cluster_col='tasks_t0_km_2', 
                                          category_col='race_ethnicity', export_dir=export_directory,
                                          prefix='race_all')
race_singleton_counts = cluster_category_counts(df=abcd_singleton, cluster_col='tasks_t0_km_2', 
                                                category_col='race_ethnicity', export_dir=export_directory,
                                                prefix='race_singleton')

#included: 2, 3, 4, 5, 6 
#excluded: 1, 7
recode_map = {1: np.nan, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: np.nan}
for key, df in full_dfs.items():
    if 'demo_ed_v2' in df.columns:
        df['education_rc'] = df['demo_ed_v2'].replace(recode_map)
        
categorical_vars = ['demo_sex_rc', 'education_rc', 
                    'demo_comb_income_v2',
                    'race_ethnicity','site_id_l']

"""
Tests
"""
#baseline
non_normal_t0_vars = non_normal_t0[non_normal_t0['dataframe'] == 'all']['variable'].tolist()
normal_t0_vars = list(set(t0_numeric_vars) - set(non_normal_t0_vars))

results_t0 = {}

for df_name, df in full_dfs.items():
    mann_whitney_t0_train, mann_whitney_t0_test = train_test(df, mann_whitney_u, non_normal_t0_vars, 
                                                             train_col='train_ids', test_col='test_ids',
                                                             group_col='tasks_t0_km_2')
    
    t_t0_train, t_t0_test = train_test(df, independent_t, normal_t0_vars, 
                                       train_col='train_ids', test_col='test_ids',
                                       group_col='tasks_t0_km_2')
    
    chi_square_t0_train, chi_square_t0_test = train_test(df, chi_square, categorical_vars, 
                                                               train_col='train_ids', test_col='test_ids',
                                                               group_col='tasks_t0_km_2')
    
    results_t0[df_name] = {
        'mann_whitney_t0_train': mann_whitney_t0_train,
        'mann_whitney_t0_test': mann_whitney_t0_test,
        't_t0_train': t_t0_train,
        't_t0_test': t_t0_test,
        'chi_square_t0_train': chi_square_t0_train,
        'chi_square_t0_test': chi_square_t0_test
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

# 2-year follow-up
non_normal_t2_vars = non_normal_t2[non_normal_t2['dataframe'] == 'all']['variable'].tolist()
normal_t2_vars = list(set(t2_numeric_vars) - set(non_normal_t2_vars))

results_t2 = {}

for df_name, df in full_dfs.items():
    mann_whitney_t2_train, mann_whitney_t2_test = train_test(df, mann_whitney_u, non_normal_t2_vars, 
                                                             train_col='train_ids', test_col='test_ids',
                                                             group_col='tasks_t2_km_2')
    
    t_t2_train, t_t2_test = train_test(df, independent_t, normal_t2_vars, 
                                       train_col='train_ids', test_col='test_ids',
                                       group_col='tasks_t2_km_2')
    
    chi_square_t2_train, chi_square_t2_test = train_test(df, chi_square, categorical_vars, 
                                                               train_col='train_ids', test_col='test_ids',
                                                               group_col='tasks_t2_km_2')
    
    results_t2[df_name] = {
        'mann_whitney_t2_train': mann_whitney_t2_train,
        'mann_whitney_t2_test': mann_whitney_t2_test,
        't_t2_train': t_t2_train,
        't_t2_test': t_t2_test,
        'chi_square_t2_train': chi_square_t2_train,
        'chi_square_t2_test': chi_square_t2_test
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
distribution_cluster_t0.to_csv(export_directory/'distribution_cluster_t0.csv', index=False)
distribution_cluster_t2.to_csv(export_directory/'distribution_cluster_t2.csv', index=False)

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
