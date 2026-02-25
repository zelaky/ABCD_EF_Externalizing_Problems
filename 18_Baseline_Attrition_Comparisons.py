#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Baseline Participant Comparisons (Part 1)
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Comparing sample to non-sample participants on all baseline variables of interest.
- Comparing sample participants who completed ABCD Study waves baseline and 1-year, 2-year, and 3-year follow-up waves 
  to those who drop out on all baseline variables of interest. 

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
demo_t0_ids = demo_t0_split[['src_subject_id', 'sample_ids', 'complete_ids']]
demo_t0_ids.loc[demo_t0_ids['sample_ids'] == 0, 'complete_ids'] = np.nan

demo_t0 = demo_t0_split[['src_subject_id', 'eventname', 'demo_sex_rc', 'interview_age']]

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

categorical_vars = ['demo_sex_rc', 'ksads_dmdd', 'ksads_adhd_other', 
                'ksads_adhd_present', 'ksads_adhd_past', 
                'ksads_odd_present', 'ksads_odd_past', 
                'ksads_cd_present_child', 'ksads_cd_present_adolescent', 
                'ksads_cd_past_child', 'ksads_cd_past_adolescent']

"""
Sample Comparisons
"""
#check variable normality
distribution_results = []

for key, df in baseline_dfs.items():
    for var in numeric_vars:
        if var in df.columns:
            include_sample = df[df['sample_ids'] == 1][var].dropna()
            exclude_sample = df[df['sample_ids'] == 0][var].dropna()

            include_skew_val = skew(include_sample) if len(include_sample) > 2 else np.nan
            include_kurt_val = kurtosis(include_sample) if len(include_sample) > 2 else np.nan
            exclude_skew_val = skew(exclude_sample) if len(exclude_sample) > 2 else np.nan
            exclude_kurt_val = kurtosis(exclude_sample) if len(exclude_sample) > 2 else np.nan

            distribution_results.append({
                'dataframe': key,
                'variable': var,
                'include_n': len(include_sample),
                'include_skew_value': include_skew_val,
                'include_skew': abs(include_skew_val) > 2 if not np.isnan(include_skew_val) else np.nan,
                'include_kurtosis_value': include_kurt_val,
                'include_kurtosis': abs(include_kurt_val) > 2 if not np.isnan(include_kurt_val) else np.nan,
                'exclude_n': len(exclude_sample),
                'exclude_skew_value': exclude_skew_val,
                'exclude_skew': abs(exclude_skew_val) > 2 if not np.isnan(exclude_skew_val) else np.nan,
                'exclude_kurtosis_value': exclude_kurt_val,
                'exclude_kurtosis': abs(exclude_kurt_val) > 2 if not np.isnan(exclude_kurt_val) else np.nan
            })

distribution_sample = pd.DataFrame(distribution_results)

mask = (
    (distribution_sample['include_skew'] == True) |
    (distribution_sample['exclude_skew'] == True) |
    (distribution_sample['include_kurtosis'] == True) |
    (distribution_sample['exclude_kurtosis'] == True)
)

non_normal_sample = distribution_sample.loc[mask, ['dataframe', 'variable']].reset_index(drop=True)

#updated lists
non_normal_sample_vars = ['sst_crlg_rate', 'sst_crlg_mrt', 'sst_crlg_stdrt', 'sst_incrlg_rate',
                          'sst_nrgo_rate', 'sst_crs_rate', 'sst_incrs_rate', 'sst_ssds_rate', 
                          'sst_issrt', 'cbcl_aggressive_t', 'cbcl_rulebreak_t', 'cbcl_attention_t']

normal_sample_vars = ['interview_age', 'sst_crgo_rate', 'sst_crgo_mrt', 'sst_crgo_stdrt', 
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

del non_normal_sample, mask

#Mann-Whitney U Test (i.e., Wilcoxon Rank Sum Test)
u_results = []

for key, df in baseline_dfs.items():
    for var in non_normal_sample_vars:
        if var in df.columns:
            include_sample = df[df['sample_ids'] == 1][var].dropna()
            exclude_sample = df[df['sample_ids'] == 0][var].dropna()

            if len(include_sample) > 1 and len(exclude_sample) > 1:
                u_stat, u_p = mannwhitneyu(include_sample, exclude_sample, alternative='two-sided')

                in_n, ex_n = len(include_sample), len(exclude_sample)
                in_sd, ex_sd = include_sample.std(ddof=1), exclude_sample.std(ddof=1)
                r_rb = 1 - (2 * u_stat) / (in_n * ex_n) 

                u_results.append({
                    'dataframe': key,
                    'variable': var,
                    'include_n': in_n,
                    'include_mean': include_sample.mean(),
                    'include_sd': in_sd,
                    'exclude_n': ex_n,
                    'exclude_mean': exclude_sample.mean(),
                    'exclude_sd': ex_sd,
                    'u_value': u_stat,
                    'u_p_value': u_p,
                    'rank_biserial_r': r_rb
                })

mann_whitney_u_sample = pd.DataFrame(u_results)

#Independent Samples T-Test
t_results = []

for key, df in baseline_dfs.items():
    for var in normal_sample_vars:
        if var in df.columns:
            include_sample = df[df['sample_ids'] == 1][var].dropna()
            exclude_sample = df[df['sample_ids'] == 0][var].dropna()

            if len(include_sample) > 1 and len(exclude_sample) > 1:
                levene_stat, levene_p = levene(include_sample, exclude_sample)
                equal_var = levene_p > 0.05

                t_stat, t_p = ttest_ind(include_sample, exclude_sample, equal_var=equal_var)

                in_n, ex_n = len(include_sample), len(exclude_sample)
                in_sd, ex_sd = include_sample.std(ddof=1), exclude_sample.std(ddof=1)
                pooled_sd = np.sqrt(((in_n - 1)*in_sd**2 + (ex_n - 1)*ex_sd**2) / (in_n + ex_n - 2))
                cohen_d = (include_sample.mean() - exclude_sample.mean()) / pooled_sd if pooled_sd != 0 else np.nan

                t_results.append({
                    'dataframe': key,
                    'variable': var,
                    'include_n': in_n,
                    'include_mean': include_sample.mean(),
                    'include_sd': in_sd,
                    'exclude_n': ex_n,
                    'exclude_mean': exclude_sample.mean(),
                    'exclude_sd': ex_sd,
                    't_value': t_stat,
                    'levene_value': levene_stat,
                    'levene_p_value': levene_p,
                    'equal_var': equal_var,
                    't_p_value': t_p,
                    'cohen_d': cohen_d
                })

independent_t_sample = pd.DataFrame(t_results)

#Fisher's Exact Test
fishers_results = []

for key, df in baseline_dfs.items():
    for var in categorical_vars:
        if var in df.columns:
            sub_df = df[['sample_ids', var]].dropna()
            contingency = pd.crosstab(sub_df['sample_ids'], sub_df[var])
            
            if contingency.shape == (2, 2):
                oddsratio, p_value = fisher_exact(contingency)
                values = contingency.values
                
                fishers_results.append({
                    'dataframe': key,
                    'variable': var,
                    'include_n': values[1].sum(),  
                    'exclude_n': values[0].sum(),
                    'count_00': values[0, 0], #row 0, col 0
                    'count_01': values[0, 1], #row 0, col 1
                    'count_10': values[1, 0], #row 1, col 0
                    'count_11': values[1, 1], #row 1, col 1
                    'odds_ratio': oddsratio,
                    'fisher_p_value': p_value
                })
            else:
                print(f"Skipping {key} - {var}: not a 2x2 table (shape {contingency.shape})")

fishers_exact_sample = pd.DataFrame(fishers_results)

##Contingency Tables Structure:
    #         KSADS=0     KSADS=1
    # Exclude    X          X
    # Include    X          X
    
#FDR Corrections
dfs = [mann_whitney_u_sample, independent_t_sample, fishers_exact_sample]
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
Attrition Comparisons
"""
#check variable normality
distribution_results = []

for key, df in baseline_dfs.items():
    for var in numeric_vars:
        if var in df.columns:
            retain_sample = df[df['complete_ids'] == 1][var].dropna()
            drop_sample = df[df['complete_ids'] == 0][var].dropna()

            retain_skew_val = skew(retain_sample) if len(retain_sample) > 2 else np.nan
            retain_kurt_val = kurtosis(retain_sample) if len(retain_sample) > 2 else np.nan
            drop_skew_val = skew(drop_sample) if len(drop_sample) > 2 else np.nan
            drop_kurt_val = kurtosis(drop_sample) if len(drop_sample) > 2 else np.nan

            distribution_results.append({
                'dataframe': key,
                'variable': var,
                'retain_n': len(retain_sample),
                'retain_skew_value': retain_skew_val,
                'retain_skew': abs(retain_skew_val) > 2 if not np.isnan(retain_skew_val) else np.nan,
                'retain_kurtosis_value': retain_kurt_val,
                'retain_kurtosis': abs(retain_kurt_val) > 2 if not np.isnan(retain_kurt_val) else np.nan,
                'drop_n': len(drop_sample),
                'drop_skew_value': drop_skew_val,
                'drop_skew': abs(drop_skew_val) > 2 if not np.isnan(drop_skew_val) else np.nan,
                'drop_kurtosis_value': drop_kurt_val,
                'drop_kurtosis': abs(drop_kurt_val) > 2 if not np.isnan(drop_kurt_val) else np.nan
            })

distribution_attrition = pd.DataFrame(distribution_results)

mask = (
    (distribution_attrition['retain_skew'] == True) |
    (distribution_attrition['drop_skew'] == True) |
    (distribution_attrition['retain_kurtosis'] == True) |
    (distribution_attrition['drop_kurtosis'] == True)
)

non_normal_attrition = distribution_attrition.loc[mask, ['dataframe', 'variable']].reset_index(drop=True)

#updated lists
non_normal_attrition_vars = ['sst_crlg_rate', 'sst_crlg_mrt', 'sst_crlg_stdrt', 'sst_incrgo_mrt',
                             'sst_incrlg_rate', 'sst_nrgo_rate', 'sst_crs_rate', 'sst_incrs_rate',
                             'sst_ssds_rate', 'sst_issrt', 'cbcl_aggressive_t', 'cbcl_rulebreak_t',
                             'cbcl_attention_t']

normal_attrition_vars = ['interview_age', 'sst_crgo_rate', 'sst_crgo_mrt', 'sst_crgo_stdrt', 
                         'sst_incrgo_rate', 'sst_incrgo_stdrt', 'sst_incrs_mrt', 'sst_incrs_stdrt', 
                         'sst_mssd', 'sst_mssrt', 'nback_c2bpf_rate', 'nback_c2bpf_mrt', 
                         'nback_c2bpf_stdrt', 'nback_c2bnf_rate', 'nback_c2bnf_mrt', 'nback_c2bnf_stdrt', 
                         'nback_c2bp_rate', 'nback_c2bp_mrt', 'nback_c2bp_stdrt', 'nback_c0bpf_rate',
                         'nback_c0bpf_mrt', 'nback_c0bpf_stdrt',  'nback_c0bnf_rate', 'nback_c0bnf_mrt', 
                         'nback_c0bnf_stdrt', 'nback_c0bngf_rate', 'nback_c0bngf_mrt', 'nback_c0bngf_stdrt', 
                         'nback_c0bp_rate', 'nback_c0bp_mrt', 'nback_c0bp_stdrt', 'nback_c2bngf_rate', 
                         'nback_c2bngf_mrt', 'nback_c2bngf_stdrt', 'nback_c0b_rate', 'nback_c0b_mrt', 
                         'nback_c0b_stdrt', 'nback_c2b_rate', 'nback_c2b_mrt', 'nback_c2b_stdrt',
                         'cbcl_external_t']

del non_normal_attrition, mask

#Mann-Whitney U Test (i.e., Wilcoxon Rank Sum Test)
u_results = []

for key, df in baseline_dfs.items():
    for var in non_normal_sample_vars:
        if var in df.columns:
            retain_sample = df[df['complete_ids'] == 1][var].dropna()
            drop_sample = df[df['complete_ids'] == 0][var].dropna()

            if len(retain_sample) > 1 and len(drop_sample) > 1:
                u_stat, u_p = mannwhitneyu(retain_sample, drop_sample, alternative='two-sided')

                retain_n, drop_n = len(retain_sample), len(drop_sample)
                retain_sd, drop_sd = retain_sample.std(ddof=1), drop_sample.std(ddof=1)
                r_rb = 1 - (2 * u_stat) / (retain_n * drop_n)

                u_results.append({
                    'dataframe': key,
                    'variable': var,
                    'retain_n': retain_n,
                    'retain_mean': retain_sample.mean(),
                    'retain_sd': retain_sd,
                    'drop_n': drop_n,
                    'drop_mean': drop_sample.mean(),
                    'drop_sd': drop_sd,
                    'u_value': u_stat,
                    'u_p_value': u_p,
                    'rank_biserial_r': r_rb
                })

mann_whitney_u_attrition = pd.DataFrame(u_results)


#Independent Samples T-Test
t_results = []

for key, df in baseline_dfs.items():
    for var in normal_sample_vars:
        if var in df.columns:
            retain_sample = df[df['complete_ids'] == 1][var].dropna()
            drop_sample = df[df['complete_ids'] == 0][var].dropna()

            if len(retain_sample) > 1 and len(drop_sample) > 1:
                levene_stat, levene_p = levene(retain_sample, drop_sample)
                equal_var = levene_p > 0.05

                t_stat, t_p = ttest_ind(retain_sample, drop_sample, equal_var=equal_var)

                retain_n, drop_n = len(retain_sample), len(drop_sample)
                retain_sd, drop_sd = retain_sample.std(ddof=1), drop_sample.std(ddof=1)
                pooled_sd = np.sqrt(((retain_n - 1)*in_sd**2 + (drop_n - 1)*ex_sd**2) / (retain_n + drop_n - 2))
                cohen_d = (retain_sample.mean() - drop_sample.mean()) / pooled_sd if pooled_sd != 0 else np.nan

                t_results.append({
                    'dataframe': key,
                    'variable': var,
                    'retain_n': retain_n,
                    'retain_mean': retain_sample.mean(),
                    'retain_sd': retain_sd,
                    'drop_n': drop_n,
                    'drop_mean': drop_sample.mean(),
                    'drop_sd': drop_sd,
                    't_value': t_stat,
                    'levene_value': levene_stat,
                    'levene_p_value': levene_p,
                    'equal_var': equal_var,
                    't_p_value': t_p,
                    'cohen_d': cohen_d
                })

independent_t_attrition = pd.DataFrame(t_results)

#Fisher's Exact Test
fishers_results = []

for key, df in baseline_dfs.items():
    for var in categorical_vars:
        if var in df.columns:
            sub_df = df[['complete_ids', var]].dropna()
            contingency = pd.crosstab(sub_df['complete_ids'], sub_df[var])
            
            if contingency.shape == (2, 2):
                values = contingency.values.astype(float)

                #continuity correction for OR
                continuity_correct = values + 0.5
                oddsratio = (continuity_correct[0,0] * continuity_correct[1,1]) / \
                            (continuity_correct[0,1] * continuity_correct[1,0])
                _, p_value = fisher_exact(values)

                fishers_results.append({
                    'dataframe': key,
                    'variable': var,
                    'retain_n': values[1].sum(),  
                    'drop_n': values[0].sum(),
                    'count_00': values[0, 0],
                    'count_01': values[0, 1],
                    'count_10': values[1, 0],
                    'count_11': values[1, 1],
                    'odds_ratio_correct': oddsratio,
                    'fisher_p_value': p_value
                })
            else:
                print(f"Skipping {key} - {var}: not a 2x2 table (shape {contingency.shape})")

fishers_exact_attrition = pd.DataFrame(fishers_results)

#FDR Corrections
dfs = [mann_whitney_u_attrition, independent_t_attrition, fishers_exact_attrition]
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
distribution_sample.to_csv(export_directory/'distribution_sample.csv', index=False)
distribution_attrition.to_csv(export_directory/'distribution_attrition.csv', index=False)
mann_whitney_u_sample.to_csv(export_directory/'mann_whitney_u_sample.csv', index=False)
mann_whitney_u_attrition.to_csv(export_directory/'mann_whitney_u_attrition.csv', index=False)
independent_t_sample.to_csv(export_directory/'independent_t_sample.csv', index=False)
independent_t_attrition.to_csv(export_directory/'independent_t_attrition.csv', index=False)
fishers_exact_sample.to_csv(export_directory/'fishers_exact_sample.csv', index=False)
fishers_exact_attrition.to_csv(export_directory/'fishers_exact_attrition.csv', index=False)






