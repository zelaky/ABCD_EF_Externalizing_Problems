#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Train-Test Split
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Splitting the sample into training and testing sets, stratefied by missing pattern, with families grouped together, and balanced for sex at birth. 

Input(s):
- demo_t0_pattern.csv

Output(s):
- demo_t0_split.csv

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
from scipy.stats import chi2_contingency

#load data
demo_t0_pattern = pd.read_csv(import_directory/'4_Attrition'/'demo_t0_pattern.csv')

"""
Group IDs by Family
- 8 participants removed with unique missing profiles (i.e., <2 people with identical profile 
and cannot be split across 2 groups) from split opperation
"""
#get final sample to splt
sample_ids = demo_t0_pattern[demo_t0_pattern['sample_ids' ] == 1]

rare_categories = sample_ids['miss_category'].value_counts()
rare_categories = rare_categories[rare_categories < 2].index.tolist()
sample_ids = sample_ids[~sample_ids['miss_category'].isin(rare_categories)]

#check if there is a pattern in missing by family
family_multiple = sample_ids[sample_ids.duplicated('rel_family_id', keep=False)]
unique_counts = sample_ids['rel_family_id'].value_counts()
unique_counts = unique_counts.value_counts()

groups = family_multiple.groupby('rel_family_id')
same_miss = groups['miss_category'].transform('nunique') == 1
family_same_miss = family_multiple[same_miss]
family_diff_miss = family_multiple[~same_miss]

del same_miss, family_same_miss, family_diff_miss

#identify families where there are multiple essential members and select one participant per family
miss_category_count = sample_ids.groupby('miss_category').size()
mask = (miss_category_count == 2)
miss_category_filter = sample_ids['miss_category'].map(mask).astype(int)
sample_ids.loc[:, 'miss_category_flag'] = miss_category_filter
sample_ids.loc[:, 'miss_category_count'] = sample_ids.groupby('rel_family_id')['miss_category_flag'].transform('sum')

family_split_random = sample_ids[sample_ids['miss_category_count'] == 0]
family_split_manual = sample_ids[(sample_ids['miss_category_count'] == 1) & (sample_ids['miss_category_flag'] == 1)]

seed = 348
family_select = family_split_random.groupby('rel_family_id').sample(n=1, random_state=seed)
family_select = pd.concat([family_select, family_split_manual], ignore_index=True)

#sort data
family_select = family_select.sort_values(by='miss_category').reset_index(drop=True)

n_group = 2
family_select['split_ids'] = family_select.index % n_group

del mask, miss_category_filter, miss_category_count, n_group, family_split_manual, family_split_random

sex_split_family = family_select.groupby(['split_ids', 'demo_sex_rc'])['src_subject_id'].count()
sex_split_family = sex_split_family.reset_index()

miss_split_family = family_select.groupby(['split_ids', 'miss_category'])['src_subject_id'].count()
miss_split_family = miss_split_family.reset_index()

#merge in siblings to split groups
family_groups = family_select.loc[:, ['rel_family_id', 'split_ids']]
split_ids = sample_ids.merge(family_groups, how='outer', on='rel_family_id')

groups = split_ids.groupby('rel_family_id')
same_split = groups['split_ids'].transform('nunique') == 1
family_same_split = split_ids[same_split]
family_diff_split = split_ids[~same_split]

del groups, family_groups, same_split, family_same_split, family_diff_split

sex_split_all = split_ids.groupby(['split_ids', 'demo_sex_rc'])['src_subject_id'].count()
sex_split_all = sex_split_all.reset_index()

miss_split_all = split_ids.groupby(['split_ids', 'miss_category'])['src_subject_id'].count()
miss_split_all = miss_split_all.reset_index()

del miss_split_all
del sex_split_family
del miss_split_family

#chi-square test on sex
test_freq = [1452, 1291]
train_freq = [1461, 1297]
table = pd.DataFrame({'Train': train_freq, 'Test': test_freq}, index=['Male', 'Female'])
stat, p_value, _, _ = chi2_contingency(table)
#stat = 2.3623568570284194e-06; p-value = 0.9987736550647125

del sex_split_all, test_freq, train_freq, stat, p_value, seed, table

"""
Merge with Demographics
"""
split_ids = split_ids[['src_subject_id', 'split_ids']]
demo_t0_split = demo_t0_pattern.merge(split_ids, how='outer', on=['src_subject_id'])
unique_counts = demo_t0_split['split_ids'].value_counts()

demo_t0_split['train_ids'] = np.where(demo_t0_split['split_ids'] == 1, 1, np.nan)
demo_t0_split['test_ids'] = np.where(demo_t0_split['split_ids'] == 0, 1, np.nan)
demo_t0_split.loc[(demo_t0_split['sample_ids'] == 1) & (demo_t0_split['split_ids'].isna()), 'sample_ids'] = 0

"""
Export Files
"""
demo_t0_split.to_csv(export_directory/'demo_t0_split.csv', index=False)


