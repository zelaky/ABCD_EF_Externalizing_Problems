#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Cluster Temporal Stability 
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Exploring and comparing the proportion of participants in the same relative cluster across timepoints (i.e., baseline and 2-year follow-up).

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
import scipy
import statsmodels
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from scipy.stats import skew, kurtosis
import pingouin as pg

#load data
abcd_all = pd.read_csv(import_directory/'abcd_all.csv')
abcd_singleton = pd.read_csv(import_directory/'abcd_singleton.csv')
abcd_only = pd.read_csv(import_directory/'abcd_only.csv')
abcd_rt = pd.read_csv(import_directory/'abcd_rt.csv') 

"""
Functions
"""
def similarity_value(i_val, j_val):
    if np.isnan(i_val) or np.isnan(j_val):
        return np.nan
    return (i_val == j_val) and (i_val != -1)

def temporal_stability(dataframe, col1, col2):
    n = len(dataframe)
    pairs = list(combinations(range(n), 2))
    similarity = []
    for i, j in pairs:
        val1_i = dataframe[col1].iloc[i]
        val1_j = dataframe[col1].iloc[j]
        val2_i = dataframe[col2].iloc[i]
        val2_j = dataframe[col2].iloc[j]
        sim_t0 = similarity_value(val1_i, val1_j)
        sim_t2 = similarity_value(val2_i, val2_j)
        if np.isnan(sim_t0) or np.isnan(sim_t2):
            similarity.append(np.nan)
        else:
            similarity.append(sim_t0 == sim_t2)  
    consistency_vect = np.array(similarity, dtype=np.float64)
    consistency_score = np.nanmean(consistency_vect)
    return {
        'temporal_stability': consistency_score,
        'consistency_vect': consistency_vect
    }

def align_labels(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize agreement
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned = [mapping[label] for label in pred_labels]
    return aligned
        
def mixed_anova(df, paired_vars, id_groups, skew_threshold=2, kurt_threshold=2, padjust='bonferroni'):
    group_col = 'group' 
    df[group_col] = pd.Series(dtype='object')
    for group_name, ids in id_groups.items(): 
        df.loc[df['src_subject_id'].isin(ids), group_col] = str(group_name)
    results = []
    posthocs_dict = {}
    for t0_var, t2_var in paired_vars:
        data = df[['src_subject_id', group_col, t0_var, t2_var]].dropna()
        if data.empty:
            continue
        diff = data[t2_var] - data[t0_var]
        diff_skew = skew(diff)
        diff_kurt = kurtosis(diff)
        transform = abs(diff_skew) > skew_threshold or abs(diff_kurt) > kurt_threshold
        long_df = pd.melt(
            data,
            id_vars=['src_subject_id', group_col],
            value_vars=[t0_var, t2_var],
            var_name='timepoint',
            value_name='value'
        )
        if transform:
            long_df['value'] = long_df['value'].apply(lambda x: np.log1p(x) if x > 0 else x)
        try:
            aov = pg.mixed_anova(
                dv='value',
                within='timepoint',
                between=group_col,
                subject='src_subject_id',
                data=long_df
            )
            aov['variable'] = f'{t0_var}_t2'
            aov['log_transform'] = transform
            results.append(aov)
        except Exception as e:
            print(f"ANOVA error for {t0_var}_t2: {e}")
            continue
        try:
            posthoc = pg.pairwise_tests(
                dv='value',
                within='timepoint',
                between=group_col,
                subject='src_subject_id',
                data=long_df,
                padjust=padjust,
                effsize='cohen'
            )
            posthocs_dict[f'{t0_var}_t2'] = posthoc
        except Exception as e:
            print(f"Post-hoc error for {t0_var}_t2: {e}")
            posthocs_dict[f'{t0_var}_t2'] = pd.DataFrame()
    results_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return results_df, posthocs_dict

def group_descriptives(df, paired_vars, id_groups):
    group_col = 'group'
    df[group_col] = pd.Series(dtype='object')
    for group_name, ids in id_groups.items():
        df.loc[df['src_subject_id'].isin(ids), group_col] = str(group_name)
    all_stats = []
    for t0_var, t2_var in paired_vars:
        data = df[['src_subject_id', group_col, t0_var, t2_var]].dropna()
        if data.empty:
            continue
        long_df = pd.melt(
            data,
            id_vars=['src_subject_id', group_col],
            value_vars=[t0_var, t2_var],
            var_name='timepoint',
            value_name='value'
        )
        stats = long_df.groupby([group_col, 'timepoint'])['value'].agg(
            mean='mean',
            sd='std',
            n='count'
        ).reset_index()
        stats.rename(columns={group_col: 'group'}, inplace=True)
        all_stats.append(stats)
    desc_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    return desc_df

"""
Impute Stability
"""
all_clusters = abcd_all[['src_subject_id', 'tasks_t0_km_2', 'tasks_t2_km_2']]
all_clusters = all_clusters.rename(
    columns={col: col + "_all" for col in all_clusters.columns if col.startswith("tasks_")}
)

only_clusters = abcd_only[['src_subject_id', 'tasks_t0_km_2', 'tasks_t2_km_2']]
only_clusters = only_clusters.rename(
    columns={col: col + "_only" for col in only_clusters.columns if col.startswith("tasks_")}
)

rt_clusters = abcd_rt[['src_subject_id', 'tasks_t0_km_2', 'tasks_t2_km_2']]
rt_clusters = rt_clusters.rename(
    columns={col: col + "_rt" for col in rt_clusters.columns if col.startswith("tasks_")}
)

cluster_assignments = pd.merge(all_clusters, only_clusters, on='src_subject_id', how='inner')
cluster_assignments = pd.merge(cluster_assignments, rt_clusters, on='src_subject_id', how='inner')

column_pairs = [
    ('tasks_t0_km_2_all', 'tasks_t0_km_2_only'),
    ('tasks_t0_km_2_all', 'tasks_t0_km_2_rt'),
    ('tasks_t2_km_2_all', 'tasks_t2_km_2_only'),
    ('tasks_t2_km_2_all', 'tasks_t2_km_2_rt'),
]

df = cluster_assignments
results = []

for col1, col2 in column_pairs:
    output = temporal_stability(df, col1, col2)
    results.append({
        'comparison': f'{col1}_t2',
        'stability': output['temporal_stability']
    })

impute_similarity_df = pd.DataFrame(results)
impute_similarity_df.to_csv(export_directory/'impute_similarity.csv', index=False)

"""
Temporal Stability
"""
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
        new_col = f"{col2}_align"
        df.loc[df.index, new_col] = aligned

column_pairs = [('tasks_t0_km_2', 'tasks_t2_km_2_align')]
results = []

for key, df in full_dfs.items():
    for col1, col2 in column_pairs: 
        output = temporal_stability(df, col1, col2) 
        results.append({
            'dataframe': key,
            'comparison': f'{col1}_t2',
            'stability': output['temporal_stability']
        })
temporal_similarity_df = pd.DataFrame(results)
temporal_similarity_df.to_csv(export_directory / 'temporal_similarity.csv', index=False)

results = []

#Cohen's Kappa 
for key, df in full_dfs.items():
    for col1, col2 in column_pairs: 
        output = cohen_kappa_score(df[col1], df[col2])
        confusion_mat = pd.crosstab(df[col1], df[col2], rownames=['T0'], colnames=['T2'])
        print(confusion_mat)
        results.append({
            'dataframe': key,
            'comparison': f'{col1}_t2',
            'stability': output,
            'confusion_matrix': confusion_mat
            
        })

#all
#    T2     0.0   1.0
#T0            
#    0      2199   728
#    1      890    1684

#singleton
#    T2     0.0   1.0
#T0            
#    0      1524   809
#    1      609    1920

#only
#    T2     0.0   1.0
#T0            
#    0      2199   728
#    1      890    1684

#rt
#    T2     0.0   1.0
#T0            
#    0      1684   888
#    1      729    2200
 
temporal_kappa_df = pd.DataFrame(results)
temporal_kappa_df.to_csv(export_directory / 'temporal_kappa.csv', index=False)

"""
Task Performance
"""

id_lookup = {}

for key, df in full_dfs.items():
    for col1, col2 in column_pairs:
        confusion_mat = pd.crosstab(df[col1], df[col2], rownames=['T0'], colnames=['T2'])
        for t0_val in confusion_mat.index:
            for t2_val in confusion_mat.columns:
                ids = df.loc[(df[col1] == t0_val) & (df[col2] == t2_val), 'src_subject_id'].tolist()
                key_str = f't0_{int(t0_val)}_t2_{int(t2_val)}'
                id_lookup[(key, f'{col1}_t2', key_str)] = ids

full_dfs['all']['tasks_t0_km_2'].value_counts()
# tasks_t0_km_2
# 0    2927
# 1    2574

full_dfs['all']['tasks_t2_km_2_align'].value_counts()
# tasks_t2_km_2_align
# 0.0    3089
# 1.0    2412

all_t0_0_t2_1_high_low = id_lookup[('all', 'tasks_t0_km_2_t2', 't0_0_t2_1')]
all_t0_0_t2_0_high = id_lookup[('all', 'tasks_t0_km_2_t2', 't0_0_t2_0')]
all_t0_1_t2_0_low_high = id_lookup[('all', 'tasks_t0_km_2_t2', 't0_1_t2_0')]
all_t0_1_t2_1_low = id_lookup[('all', 'tasks_t0_km_2_t2', 't0_1_t2_1')]

singleton_t0_0_t2_1_high_low = id_lookup[('singleton', 'tasks_t0_km_2_t2', 't0_0_t2_1')]
singleton_t0_0_t2_0_high = id_lookup[('singleton', 'tasks_t0_km_2_t2', 't0_0_t2_0')]
singleton_t0_1_t2_0_low_high = id_lookup[('singleton', 'tasks_t0_km_2_t2', 't0_1_t2_0')]
singleton_t0_1_t2_1_low = id_lookup[('singleton', 'tasks_t0_km_2_t2', 't0_1_t2_1')]

only_t0_0_t2_1_high_low = id_lookup[('only', 'tasks_t0_km_2_t2', 't0_0_t2_1')]
only_t0_0_t2_0_high = id_lookup[('only', 'tasks_t0_km_2_t2', 't0_0_t2_0')]
only_t0_1_t2_0_low_high = id_lookup[('only', 'tasks_t0_km_2_t2', 't0_1_t2_0')]
only_t0_1_t2_1_low = id_lookup[('only', 'tasks_t0_km_2_t2', 't0_1_t2_1')]

rt_t0_0_t2_1_high_low = id_lookup[('rt', 'tasks_t0_km_2_t2', 't0_0_t2_1')]
rt_t0_0_t2_0_high = id_lookup[('rt', 'tasks_t0_km_2_t2', 't0_0_t2_0')]
rt_t0_1_t2_0_low_high = id_lookup[('rt', 'tasks_t0_km_2_t2', 't0_1_t2_0')]
rt_t0_1_t2_1_low = id_lookup[('rt', 'tasks_t0_km_2_t2', 't0_1_t2_1')]

all_id_groups = {
    'high_low': all_t0_0_t2_1_high_low,
    'high': all_t0_0_t2_0_high,
    'low_high': all_t0_1_t2_0_low_high,
    'low': all_t0_1_t2_1_low
}

singleton_id_groups = {
    'high_low': singleton_t0_0_t2_1_high_low,
    'high': singleton_t0_0_t2_0_high,
    'low_high': singleton_t0_1_t2_0_low_high,
    'low': singleton_t0_1_t2_1_low
}

only_id_groups = {
    'high_low': only_t0_0_t2_1_high_low,
    'high': only_t0_0_t2_0_high,
    'low_high': only_t0_1_t2_0_low_high,
    'low': only_t0_1_t2_1_low
}

rt_id_groups = {
    'high_low': rt_t0_0_t2_1_high_low,
    'high': rt_t0_0_t2_0_high,
    'low_high': rt_t0_1_t2_0_low_high,
    'low': rt_t0_1_t2_1_low
}

column_pairs = [
    ('sst_t0_crgo_stdrt', 'sst_t2_crgo_stdrt'),
    ('sst_t0_crlg_rate', 'sst_t2_crlg_rate'),
    ('nback_t0_c0b_rate', 'nback_t2_c0b_rate'),
    ('nback_t0_c0b_mrt', 'nback_t2_c0b_mrt'),
    ('nback_t0_c2b_rate', 'nback_t2_c2b_rate'),
    ('nback_t0_c2b_stdrt', 'nback_t2_c2b_stdrt'),
    ('nback_t0_c0b_stdrt', 'nback_t2_c0b_stdrt'),
    ('nback_t0_c2b_mrt', 'nback_t2_c2b_mrt')
  ]


#all
all_anova = mixed_anova(df=full_dfs['all'], paired_vars=column_pairs, id_groups=all_id_groups)
all_tasks_main = all_anova[0]
all_tasks_posthocs = all_anova[1]
for key in list(all_tasks_posthocs.keys()):
    all_tasks_posthocs[f"{key}_all_tasks_posthoc"] = all_tasks_posthocs.pop(key)
all_tasks_descriptives = group_descriptives(df=full_dfs['all'], paired_vars=column_pairs, id_groups=all_id_groups)

#singleton
singleton_anova = mixed_anova(df=full_dfs['singleton'], paired_vars=column_pairs, id_groups=singleton_id_groups)
singleton_tasks_main = singleton_anova[0]
singleton_tasks_posthocs = singleton_anova[1]
for key in list(singleton_tasks_posthocs.keys()):
    singleton_tasks_posthocs[f"{key}_singleton_tasks_posthoc"] = singleton_tasks_posthocs.pop(key)
singleton_tasks_descriptives = group_descriptives(df=full_dfs['singleton'], paired_vars=column_pairs, id_groups=singleton_id_groups)

#only
only_anova = mixed_anova(df=full_dfs['only'], paired_vars=column_pairs, id_groups=only_id_groups)
only_tasks_main = only_anova[0]
only_tasks_posthocs = only_anova[1]
for key in list(only_tasks_posthocs.keys()):
    only_tasks_posthocs[f"{key}_only_tasks_posthoc"] = only_tasks_posthocs.pop(key)
only_tasks_descriptives = group_descriptives(df=full_dfs['only'], paired_vars=column_pairs, id_groups=only_id_groups)

#rt
rt_anova = mixed_anova(df=full_dfs['rt'], paired_vars=column_pairs, id_groups=rt_id_groups)
rt_tasks_main = rt_anova[0]
rt_tasks_posthocs = rt_anova[1]
for key in list(rt_tasks_posthocs.keys()):
    rt_tasks_posthocs[f"{key}_rt_tasks_posthoc"] = rt_tasks_posthocs.pop(key)
rt_tasks_descriptives = group_descriptives(df=full_dfs['rt'], paired_vars=column_pairs, id_groups=rt_id_groups)

"""
Clinical Symptoms
"""
column_pairs = [
    ('cbcl_aggressive_t_t0', 'cbcl_aggressive_t_t2'),
    ('cbcl_attention_t_t0', 'cbcl_attention_t_t2'),
    ('cbcl_rulebreak_t_t0', 'cbcl_rulebreak_t_t2'),
    ('cbcl_external_t_t0', 'cbcl_external_t_t2'),
    ('cbcl_internal_t_t0', 'cbcl_internal_t_t2'),
    ('cbcl_totprob_t_t0', 'cbcl_totprob_t_t2')
  ]

#all
all_anova = mixed_anova(df=full_dfs['all'], paired_vars=column_pairs, id_groups=all_id_groups)
all_clinical_main = all_anova[0]
all_clinical_posthocs = all_anova[1]
for key in list(all_clinical_posthocs.keys()):
    all_clinical_posthocs[f"{key}_all_clinical_posthoc"] = all_clinical_posthocs.pop(key)
all_clinical_descriptives = group_descriptives(df=full_dfs['all'], paired_vars=column_pairs, id_groups=all_id_groups)

#singleton
singleton_anova = mixed_anova(df=full_dfs['singleton'], paired_vars=column_pairs, id_groups=singleton_id_groups)
singleton_clinical_main = singleton_anova[0]
singleton_clinical_posthocs = singleton_anova[1]
for key in list(singleton_clinical_posthocs.keys()):
    singleton_clinical_posthocs[f"{key}_singleton_clinical_posthoc"] = singleton_clinical_posthocs.pop(key)
singleton_clinical_descriptives = group_descriptives(df=full_dfs['singleton'], paired_vars=column_pairs, id_groups=singleton_id_groups)

#only
only_anova = mixed_anova(df=full_dfs['only'], paired_vars=column_pairs, id_groups=only_id_groups)
only_clinical_main = only_anova[0]
only_clinical_posthocs = only_anova[1]
for key in list(only_clinical_posthocs.keys()):
    only_clinical_posthocs[f"{key}_only_clinical_posthoc"] = only_clinical_posthocs.pop(key)
only_clinical_descriptives = group_descriptives(df=full_dfs['only'], paired_vars=column_pairs, id_groups=only_id_groups)

#rt
rt_anova = mixed_anova(df=full_dfs['rt'], paired_vars=column_pairs, id_groups=rt_id_groups)
rt_clinical_main = rt_anova[0]
rt_clinical_posthocs = rt_anova[1]
for key in list(rt_clinical_posthocs.keys()):
    rt_clinical_posthocs[f"{key}_rt_clinical_posthoc"] = rt_clinical_posthocs.pop(key)
rt_clinical_descriptives = group_descriptives(df=full_dfs['rt'], paired_vars=column_pairs, id_groups=rt_id_groups)

"""
Export Files
"""
for name, df in all_tasks_posthocs.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
all_tasks_main.to_csv(export_directory/'all_tasks_main.csv', index=False)
all_tasks_descriptives.to_csv(export_directory/'all_tasks_descriptives.csv', index=False)

for name, df in singleton_tasks_posthocs.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
singleton_tasks_main.to_csv(export_directory/'singleton_tasks_main.csv', index=False)
singleton_tasks_descriptives.to_csv(export_directory/'singleton_tasks_descriptives.csv', index=False)

for name, df in only_tasks_posthocs.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
only_tasks_main.to_csv(export_directory/'only_tasks_main.csv', index=False)
only_tasks_descriptives.to_csv(export_directory/'only_tasks_descriptives.csv', index=False)

for name, df in rt_tasks_posthocs.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
rt_tasks_main.to_csv(export_directory/'rt_tasks_main.csv', index=False)
rt_tasks_descriptives.to_csv(export_directory/'rt_tasks_descriptives.csv', index=False)
    
for name, df in all_clinical_posthocs.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
all_clinical_main.to_csv(export_directory/'all_clinical_main.csv', index=False)
all_clinical_descriptives.to_csv(export_directory/'all_clinical_descriptives.csv', index=False)

for name, df in singleton_clinical_posthocs.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
singleton_clinical_main.to_csv(export_directory/'singleton_clinical_main.csv', index=False)
singleton_clinical_descriptives.to_csv(export_directory/'singleton_clinical_descriptives.csv', index=False)
    
for name, df in only_clinical_posthocs.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
only_clinical_main.to_csv(export_directory/'only_clinical_main.csv', index=False)
only_clinical_descriptives.to_csv(export_directory/'only_clinical_descriptives.csv', index=False)

for name, df in rt_clinical_posthocs.items():
    file_path = os.path.join(export_directory, f"{name}.csv")
    df.to_csv(file_path, index=False)
rt_clinical_main.to_csv(export_directory/'rt_clinical_main.csv', index=False)
rt_clinical_descriptives.to_csv(export_directory/'rt_clinical_descriptives.csv', index=False)

for (key1, key2, key3), ids in id_lookup.items():
    filename = f"{key1}_{key2}_{key3}_ids.csv"
    filename = filename.replace('/', '-')
    filepath = os.path.join(export_directory, filename)
    pd.DataFrame({'src_subject_id': ids}).to_csv(filepath, index=False)
    print(f"Saved {filepath} with {len(ids)} IDs")