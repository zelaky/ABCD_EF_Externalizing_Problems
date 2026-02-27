#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Baseline Participant Comparisons (Part 2)
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Comparing training to testing participants on all baseline variables of interest.

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4
- scipy version: 1.15.3

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
import scipy 
from scipy.stats import ttest_ind, levene, fisher_exact, skew, kurtosis, mannwhitneyu
from statsmodels.stats.multitest import multipletests

#load data
sst_t0 = pd.read_csv(import_directory/'1_Task_Cleaning'/'sst_bissett_garavan_t0.csv')
nback_t0 = pd.read_csv(import_directory/'1_Task_Cleaning'/'nback_manual_t0.csv')
ksads_ss = pd.read_csv(import_directory/'2_Clinical_Cleaning'/'ksads_ss.csv')
cbcl_ss = pd.read_csv(import_directory/'2_Clinical_Cleaning'/'cbcl_ss.csv')
demo_t0_split = pd.read_csv(import_directory/'5_Split'/'demo_t0_split.csv')

"""
Preparing Dataframes
"""
#select baseline wave for clinical assessments
ksads_t0 = ksads_ss[ksads_ss["eventname"] == "baseline_year_1_arm_1"]
cbcl_t0 = cbcl_ss[cbcl_ss["eventname"] == "baseline_year_1_arm_1"]

#get participant lists
demo_t0_ids = demo_t0_split[['src_subject_id', 'split_ids', 'train_ids', 'test_ids']]
demo_t0 = demo_t0_split[['src_subject_id', 'eventname','interview_age']]

del ksads_ss, cbcl_ss, demo_t0_split

#merge onto baseline dataframe
source_dfs = {
    'demo_t0': demo_t0,
    'sst_t0': sst_t0,
    'nback_t0': nback_t0,
    'ksads_t0': ksads_t0,
    'cbcl_t0': cbcl_t0
}

baseline_dfs = {}

for name, df in source_dfs.items():
    baseline_df = df.merge(demo_t0_ids, on='src_subject_id', how='left')
    baseline_dfs[f"{name}"] = baseline_df

#variables
print(demo_t0.columns.tolist()) 
print(sst_t0.columns.tolist()) 
print(nback_t0.columns.tolist()) 
print(ksads_t0.columns.tolist()) 
print(cbcl_t0.columns.tolist()) 

numeric_vars = ['interview_age', 'sst_crgo_rate', 
                'sst_crgo_mrt', 'sst_crgo_stdrt', 
                'sst_crlg_rate', 'sst_crlg_mrt', 
                'sst_crlg_stdrt', 'sst_incrgo_rate', 
                'sst_incrgo_mrt', 'sst_incrgo_stdrt', 
                'sst_incrlg_rate', 'sst_nrgo_rate', 
                'sst_crs_rate', 'sst_incrs_rate', 
                'sst_incrs_mrt', 'sst_incrs_stdrt', 
                'sst_ssds_rate', 'sst_mssd', 
                'sst_mssrt', 'sst_issrt', 
                'nback_c2bpf_rate', 'nback_c2bpf_mrt', 
                'nback_c2bpf_stdrt', 'nback_c2bnf_rate', 
                'nback_c2bnf_mrt', 'nback_c2bnf_stdrt', 
                'nback_c2bp_rate', 'nback_c2bp_mrt', 
                'nback_c2bp_stdrt', 'nback_c0bpf_rate', 
                'nback_c0bpf_mrt', 'nback_c0bpf_stdrt', 
                'nback_c0bnf_rate', 'nback_c0bnf_mrt', 
                'nback_c0bnf_stdrt', 'nback_c0bngf_rate', 
                'nback_c0bngf_mrt', 'nback_c0bngf_stdrt', 
                'nback_c0bp_rate', 'nback_c0bp_mrt', 
                'nback_c0bp_stdrt', 'nback_c2bngf_rate', 
                'nback_c2bngf_mrt', 'nback_c2bngf_stdrt', 
                'nback_c0b_rate', 'nback_c0b_mrt', 
                'nback_c0b_stdrt', 'nback_c2b_rate', 
                'nback_c2b_mrt', 'nback_c2b_stdrt',
                # 'cbcl_internal_t', 'cbcl_totprob_t', 
                'cbcl_external_t', 'cbcl_aggressive_t', 
                'cbcl_rulebreak_t', 'cbcl_attention_t']

categorical_vars = ['ksads_dmdd', 'ksads_adhd_other', 
                'ksads_adhd_present', 'ksads_adhd_past', 
                'ksads_odd_present', 'ksads_odd_past', 
                'ksads_cd_present_child', 'ksads_cd_present_adolescent', 
                'ksads_cd_past_child', 'ksads_cd_past_adolescent']

"""
Split Comparisons
"""
#check variable normality
distribution_results = []

for key, df in baseline_dfs.items():
    for var in numeric_vars:
        if var in df.columns:
            train_split = df[df['split_ids'] == 1][var].dropna()
            test_split = df[df['split_ids'] == 0][var].dropna()

            train_skew_val = skew(train_split) if len(train_split) > 2 else np.nan
            train_kurt_val = kurtosis(train_split) if len(train_split) > 2 else np.nan
            test_skew_val = skew(test_split) if len(test_split) > 2 else np.nan
            test_kurt_val = kurtosis(test_split) if len(test_split) > 2 else np.nan

            distribution_results.append({
                'dataframe': key,
                'variable': var,
                'train_n': len(train_split),
                'train_skew_value': train_skew_val,
                'train_skew': abs(train_skew_val) > 2 if not np.isnan(train_skew_val) else np.nan,
                'train_kurtosis_value': train_kurt_val,
                'train_kurtosis': abs(train_kurt_val) > 2 if not np.isnan(train_kurt_val) else np.nan,
                'test_n': len(test_split),
                'test_skew_value': test_skew_val,
                'test_skew': abs(test_skew_val) > 2 if not np.isnan(test_skew_val) else np.nan,
                'test_kurtosis_value': test_kurt_val,
                'test_kurtosis': abs(test_kurt_val) > 2 if not np.isnan(test_kurt_val) else np.nan
            })

distribution_split = pd.DataFrame(distribution_results)

mask = (
    (distribution_split['train_skew'] == True) |
    (distribution_split['test_skew'] == True) |
    (distribution_split['train_kurtosis'] == True) |
    (distribution_split['test_kurtosis'] == True)
)

non_normal_split = distribution_split.loc[mask, ['dataframe', 'variable']].reset_index(drop=True)

#updated lists
non_normal_split_vars = ['sst_crlg_rate', 'sst_crlg_mrt', 'sst_crlg_stdrt', 'sst_incrlg_rate',
                          'sst_nrgo_rate', 'sst_crs_rate', 'sst_incrs_rate', 'sst_ssds_rate', 
                          'sst_issrt', 'cbcl_aggressive_t', 'cbcl_rulebreak_t', 'cbcl_attention_t']

normal_split_vars = ['interview_age', 'sst_crgo_rate', 'sst_crgo_mrt', 'sst_crgo_stdrt', 
                'sst_incrgo_rate', 'sst_incrgo_mrt', 'sst_incrgo_stdrt', 'sst_incrs_mrt', 
                'sst_incrs_stdrt', 'sst_mssd', 'sst_mssrt', 'nback_c2bpf_rate', 
                'nback_c2bpf_mrt', 'nback_c2bpf_stdrt', 'nback_c2bnf_rate', 'nback_c2bnf_mrt', 
                'nback_c2bnf_stdrt', 'nback_c2bp_rate', 'nback_c2bp_mrt', 'nback_c2bp_stdrt', 
                'nback_c0bpf_rate', 'nback_c0bpf_mrt', 'nback_c0bpf_stdrt', 'nback_c0bnf_rate', 
                'nback_c0bnf_mrt', 'nback_c0bnf_stdrt', 'nback_c0bngf_rate', 'nback_c0bngf_mrt', 
                'nback_c0bngf_stdrt', 'nback_c0bp_rate', 'nback_c0bp_mrt', 'nback_c0bp_stdrt', 
                'nback_c2bngf_rate', 'nback_c2bngf_mrt', 'nback_c2bngf_stdrt', 'nback_c0b_rate', 
                'nback_c0b_mrt', 'nback_c0b_stdrt', 'nback_c2b_rate', 'nback_c2b_mrt', 
                'nback_c2b_stdrt', 'cbcl_external_t']

del non_normal_split, mask

#Mann-Whitney U Test (i.e., Wilcoxon Rank Sum Test)
u_results = []

for key, df in baseline_dfs.items():
    for var in non_normal_split_vars:
        if var in df.columns:
            train_split = df[df['split_ids'] == 1][var].dropna()
            test_split = df[df['split_ids'] == 0][var].dropna()

            if len(train_split) > 1 and len(test_split) > 1:
                u_stat, u_p = mannwhitneyu(train_split, test_split, alternative='two-sided')

                train_n, test_n = len(train_split), len(test_split)
                train_sd, test_sd = train_split.std(ddof=1), test_split.std(ddof=1)
                r_rb = 1 - (2 * u_stat) / (train_n * test_n) 

                u_results.append({
                    'dataframe': key,
                    'variable': var,
                    'train_n': train_n,
                    'train_mean': train_split.mean(),
                    'train_sd': train_sd,
                    'test_n': test_n,
                    'test_mean': test_split.mean(),
                    'test_sd': test_sd,
                    'u_value': u_stat,
                    'u_p_value': u_p,
                    'rank_biserial_r': r_rb
                })

mann_whitney_u_split = pd.DataFrame(u_results)

#Independent Samples T-Test
t_results = []

for key, df in baseline_dfs.items():
    for var in normal_split_vars:
        if var in df.columns:
            train_split = df[df['split_ids'] == 1][var].dropna()
            test_split = df[df['split_ids'] == 0][var].dropna()

            if len(train_split) > 1 and len(test_split) > 1:
                levene_stat, levene_p = levene(train_split, test_split)
                equal_var = levene_p > 0.05

                t_stat, t_p = ttest_ind(train_split, test_split, equal_var=equal_var)

                in_n, ex_n = len(train_split), len(test_split)
                in_sd, ex_sd = train_split.std(ddof=1), test_split.std(ddof=1)
                pooled_sd = np.sqrt(((in_n - 1)*in_sd**2 + (ex_n - 1)*ex_sd**2) / (in_n + ex_n - 2))
                cohen_d = (train_split.mean() - test_split.mean()) / pooled_sd if pooled_sd != 0 else np.nan

                t_results.append({
                    'dataframe': key,
                    'variable': var,
                    'train_n': in_n,
                    'train_mean': train_split.mean(),
                    'train_sd': in_sd,
                    'test_n': ex_n,
                    'test_mean': test_split.mean(),
                    'test_sd': ex_sd,
                    't_value': t_stat,
                    'levene_value': levene_stat,
                    'levene_p_value': levene_p,
                    'equal_var': equal_var,
                    't_p_value': t_p,
                    'cohen_d': cohen_d
                })

independent_t_split = pd.DataFrame(t_results)

#Fisher's Exact Test
fishers_results = []

for key, df in baseline_dfs.items():
    for var in categorical_vars:
        if var in df.columns:
            sub_df = df[['split_ids', var]].dropna()
            contingency = pd.crosstab(sub_df['split_ids'], sub_df[var])
            
            if contingency.shape == (2, 2):
                oddsratio, p_value = fisher_exact(contingency)
                values = contingency.values
                
                fishers_results.append({
                    'dataframe': key,
                    'variable': var,
                    'train_n': values[1].sum(),  
                    'test_n': values[0].sum(),
                    'count_00': values[0, 0],
                    'count_01': values[0, 1],
                    'count_10': values[1, 0],
                    'count_11': values[1, 1],
                    'odds_ratio': oddsratio,
                    'fisher_p_value': p_value
                })
            else:
                print(f"Skipping {key} - {var}: not a 2x2 table (shape {contingency.shape})")

fishers_exact_split = pd.DataFrame(fishers_results)

#FDR Corrections
dfs = [mann_whitney_u_split, independent_t_split, fishers_exact_split]
cols = ['u_p_value', 't_p_value', 'fisher_p_value']

all_p_value = []
lengths = []

for df, col in zip(dfs, cols):
    all_p_value.extend(df[col].values)
    lengths.append(len(df))

all_p_value = pd.Series(all_p_value)

reject, fdr_p_values, _, _ = multipletests(all_p_value, alpha=0.05, method='fdr_bh')
fdr_p_values = pd.Series(fdr_p_values)

start = 0
for i, (df, col) in enumerate(zip(dfs, cols)):
    end = start + lengths[i]
    df['fdr_p'] = fdr_p_values[start:end].values  
    df['fdr_p_reject'] = df['fdr_p'] < 0.05         
    start = end

"""
Export Files
"""
distribution_split.to_csv(export_directory/'distribution_split.csv', index=False)
mann_whitney_u_split.to_csv(export_directory/'mann_whitney_u_split.csv', index=False)
independent_t_split.to_csv(export_directory/'independent_t_split.csv', index=False)
fishers_exact_split.to_csv(export_directory/'fishers_exact_split.csv', index=False)






