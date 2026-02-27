#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Kiddie Schedule for Affective Disorders and Schizophrenia (KSADS) Pre-processing
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Cleaning parent-report diagnostic data from the ABCD Study baseline and 1-year, 2-year, and 3-year follow-up waves. 

Modification History: 
- Script written by Zoë E. Laky (NOV2023 - present)

Input(s):
- mh_p_ksads_ss.csv

Output(s):
- ksads_raw.csv
- ksads_ss.csv

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
- https://wiki.abcdstudy.org/release-notes/non-imaging/mental-health.html
- Coding: 1 = present, 0 = absent, 888 = Question not asked due to primary question response (branching logic), 555 = Not administered in the assessment
"""
#core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os  

#load data
ksads_raw  = pd.read_csv(import_directory/'mh_p_ksads_ss.csv')

"""
Dimensions
"""
#11747 participants completed at baseline
ksads_raw_t0 = ksads_raw[ksads_raw['eventname'] == 'baseline_year_1_arm_1']

#11103 participants completed at 1-year follow-up
ksads_raw_t1 = ksads_raw[ksads_raw['eventname'] == '1_year_follow_up_y_arm_1']

#10756 participants completed at 2-year follow-up
ksads_raw_t2 = ksads_raw[ksads_raw['eventname'] == '2_year_follow_up_y_arm_1']

#10330 participants completed at 3-year follow-up
ksads_raw_t3 = ksads_raw[ksads_raw['eventname'] == '3_year_follow_up_y_arm_1']

del ksads_raw_t0, ksads_raw_t1, ksads_raw_t2, ksads_raw_t3

"""
Reduce and Rename Datasets
"""
#reduce dataset
ksads_vars = ['src_subject_id', 
              'eventname',
              'ksads_import_id_p',
              'ksads2_import_id_p',
              'ksads_3_848_p',
              'ksads_14_856_p',
              'ksads_14_853_p',
              'ksads_14_854_p',
              'ksads_15_901_p',
              'ksads_15_902_p',
              'ksads_16_900_p',
              'ksads_16_897_p',
              'ksads_16_899_p',
              'ksads_16_898_p',
              'ksads2_3_804_p',
              'ksads2_14_809_p',
              'ksads2_14_810_p',
              'ksads2_14_813_p',
              'ksads2_15_859_p',
              'ksads2_15_860_p',
              'ksads2_16_855_p',
              'ksads2_16_856_p',
              'ksads2_16_857_p',
              'ksads2_16_858_p'
              ]

ksads_externalize = ksads_raw[ksads_vars]

#rename features
rename_dict = {
    #Disruptive Mood Dysregulation Disorder (DMDD) Current (F34.8)
    'ksads_3_848_p': 'ksads1_dmdd',
    'ksads2_3_804_p': 'ksads2_dmdd',
    
    #Unspecified Attention-Deficit/Hyperactivity Disorder (F90.9)
    'ksads_14_856_p': 'ksads1_adhd_unspecified',
    
    #Other Specified Attention-Deficit/Hyperactivity Disorder (F90.8)
    'ksads2_14_813_p': 'ksads2_adhd_other',
    
    #Attention-Deficit/Hyperactivity Disorder Present
    'ksads_14_853_p': 'ksads1_adhd_present',
    'ksads2_14_809_p': 'ksads2_adhd_present',
    
    #Attention-Deficit/Hyperactivity Disorder Past	
    'ksads_14_854_p': 'ksads1_adhd_past',
    'ksads2_14_810_p': 'ksads2_adhd_past',
    
    #Oppositional Defiant Disorder Present (F91.3)
    'ksads_15_901_p': 'ksads1_odd_present',
    'ksads2_15_859_p': 'ksads2_odd_present',
    
    #Oppositional Defiant Disorder Past (F91.3)
    'ksads_15_902_p': 'ksads1_odd_past',
    'ksads2_15_860_p': 'ksads2_odd_past',
    
    #Conduct Disorder present childhood onset (F91.1)
    'ksads_16_897_p': 'ksads1_cd_present_child',
    'ksads2_16_855_p': 'ksads2_cd_present_child',
    
    #Conduct Disorder present adolescent onset (F91.2)
    'ksads_16_898_p': 'ksads1_cd_present_adolescent',
    'ksads2_16_856_p': 'ksads2_cd_present_adolescent',
    
    #Conduct Disorder past childhood onset (F91.1)
    'ksads_16_899_p': 'ksads1_cd_past_child',
    'ksads2_16_857_p': 'ksads2_cd_past_child',
    
    #Conduct Disorder past adolescent onset (F91.2)
    'ksads_16_900_p': 'ksads1_cd_past_adolescent',
    'ksads2_16_858_p': 'ksads2_cd_past_adolescent',
    
    #ids
    'ksads_import_id_p': 'ksads1_import_id',
    'ksads2_import_id_p': 'ksads2_import_id',
}

ksads_externalize = ksads_externalize.rename(columns=rename_dict)

del ksads_vars, rename_dict

"""
Collapsing KSADS Versions
- Columns 'ksads1_import_id' and 'ksads2_import_id' hold information split between 
  first two years and last two years of data collection, due to a KSADS version change
"""
ksads_ids_flag = ksads_externalize[ksads_externalize['ksads1_import_id'].notna() & ksads_externalize['ksads2_import_id'].notna()]

indices = [39313, 40069]
ksads1_cols = [col for col in ksads_externalize.columns if col.startswith('ksads1_')]

ksads_externalize.loc[indices, ksads1_cols] = np.nan

del ksads_ids_flag, indices, ksads1_cols

#reducing columns
merge_cols = {
    'ksads_dmdd': ('ksads1_dmdd', 'ksads2_dmdd'),
    'ksads_adhd_other': ('ksads1_adhd_unspecified', 'ksads2_adhd_other'),
    'ksads_adhd_present': ('ksads1_adhd_present', 'ksads2_adhd_present'),
    'ksads_adhd_past': ('ksads1_adhd_past', 'ksads2_adhd_past'),
    'ksads_odd_present': ('ksads1_odd_present', 'ksads2_odd_present'),
    'ksads_odd_past': ('ksads1_odd_past', 'ksads2_odd_past'),
    'ksads_cd_present_child': ('ksads1_cd_present_child', 'ksads2_cd_present_child'),
    'ksads_cd_present_adolescent': ('ksads1_cd_present_adolescent', 'ksads2_cd_present_adolescent'),
    'ksads_cd_past_child': ('ksads1_cd_past_child', 'ksads2_cd_past_child'),
    'ksads_cd_past_adolescent': ('ksads1_cd_past_adolescent', 'ksads2_cd_past_adolescent'),
}

for combined_col, (col1, col2) in merge_cols.items():
    ksads_externalize[combined_col] = ksads_externalize[col1].combine_first(ksads_externalize[col2])

ksads_externalize = ksads_externalize.drop(
    columns=[col for col in ksads_externalize.columns if col.startswith('ksads1_') or col.startswith('ksads2_')]
)

#check missing counts by column and row
cols_miss = ksads_externalize.isnull().sum()

del cols_miss, merge_cols, combined_col

"""
Assessment Timepoints 
- DMDD: baseline, 2-year follow-up, 4-year follow-up
- ODD: baseline, 1-year follow-up, 2-year follow-up, 4-year follow-up
- CD: baseline, 1-year follow-up, 2-year follow-up, 3-year follow-up, 4-year follow-up
- ADHD: baseline, 1-year follow-up, 2-year follow-up, 3-year follow-up, 4-year follow-up
"""
#value counts by column
diagnosis_cols = ['ksads_dmdd', #n = 42
             'ksads_adhd_other', #n = 182
             'ksads_adhd_present', #n = 3175
             'ksads_adhd_past', #n = 2624
             'ksads_odd_present', #n = 2068
             'ksads_odd_past', #n = 3278
             'ksads_cd_present_child', #n = 758
             'ksads_cd_present_adolescent', #n = 431
             'ksads_cd_past_child', #n = 252
             'ksads_cd_past_adolescent' #n = 65
             ]

diagnosis_counts = pd.DataFrame({
    'non_missing': ksads_externalize[diagnosis_cols].notna().sum(),
    'positive_diagnosis': (ksads_externalize[diagnosis_cols] == 1).sum()
})

#recode values
ksads_externalize = ksads_externalize.replace(888, 0)

del diagnosis_cols, diagnosis_counts

#datasets by wave
waves = {
    't0': 'baseline_year_1_arm_1',
    't1': '1_year_follow_up_y_arm_1',
    't2': '2_year_follow_up_y_arm_1',
    't3': '3_year_follow_up_y_arm_1',
    't4': '4_year_follow_up_y_arm_1'
}

for wave, label in waves.items():
    globals()[f'ksads_ss_{wave}'] = ksads_externalize[ksads_externalize['eventname'] == label].copy()

del wave, waves, label

#recode columns not administered as '555'
ksads_ss_t1['ksads_dmdd'] = 555.0
ksads_ss_t3['ksads_dmdd'] = 555.0
ksads_ss_t3['ksads_odd_present'] = 555.0
ksads_ss_t3['ksads_odd_past'] = 555.0

#bind rows across waves and drop data with missing 
ksads_ss = pd.concat([ksads_ss_t0, ksads_ss_t1, ksads_ss_t2, ksads_ss_t3, ksads_ss_t4], axis=0, ignore_index=True)
ksads_ss = ksads_ss.dropna(how='any')

del ksads_ss_t0, ksads_ss_t1, ksads_ss_t2, ksads_ss_t3, ksads_ss_t4

"""
Export Files
"""
ksads_raw.to_csv(export_directory /'ksads_raw.csv', index=False)
ksads_ss.to_csv(export_directory /'ksads_ss.csv', index=False)
