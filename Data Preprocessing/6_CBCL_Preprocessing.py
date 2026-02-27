#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Child Behavior Checklist (CBCL) Pre-processing 
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Cleaning parent-report symptom data from the ABCD Study baseline and 1-year, 2-year, and 3-year follow-up waves. 

Input(s):
- mh_p_cbcls.csv

Output(s):
- cbcl_ss.csv

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
- https://wiki.abcdstudy.org/release-notes/non-imaging/mental-health.html
"""
#core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os  

#load data
cbcl_raw = pd.read_csv(import_directory/'mh_p_cbcl.csv')

"""
Dimensions
"""
#11862 participants completed at baseline
cbcl_raw_t0 = cbcl_raw[cbcl_raw['eventname'] == 'baseline_year_1_arm_1']

#11201 participants completed at 1-year follow-up
cbcl_raw_t1 = cbcl_raw[cbcl_raw['eventname'] == '1_year_follow_up_y_arm_1']

#10896 participants completed at 2-year follow-up
cbcl_raw_t2 = cbcl_raw[cbcl_raw['eventname'] == '2_year_follow_up_y_arm_1']

#10098 participants completed at 3-year follow-up
cbcl_raw_t3 = cbcl_raw[cbcl_raw['eventname'] == '3_year_follow_up_y_arm_1']

"""
Reduce and Rename Columns
"""
#reduce dataset
cbcl_vars = ['src_subject_id', 
             'eventname', 
             'cbcl_scr_syn_internal_r', 
             'cbcl_scr_syn_internal_t', 
             'cbcl_scr_syn_external_r', 
             'cbcl_scr_syn_external_t', 
             'cbcl_scr_syn_totprob_r', 
             'cbcl_scr_syn_totprob_t', 
             'cbcl_scr_syn_aggressive_r', 
             'cbcl_scr_syn_aggressive_t', 
             'cbcl_scr_syn_rulebreak_r', 
             'cbcl_scr_syn_rulebreak_t', 
             'cbcl_scr_syn_attention_r', 
             'cbcl_scr_syn_attention_t' 
             ]

cbcl_ss = cbcl_raw[cbcl_vars]

del cbcl_raw, cbcl_vars

column_miss = cbcl_ss.isnull().sum()
exclude_cols = ['src_subject_id', 'eventname']
check_miss_cols = [col for col in cbcl_ss.columns if col not in exclude_cols]
cbcl_ss = cbcl_ss.dropna(subset=check_miss_cols, how='all')

del column_miss, exclude_cols, check_miss_cols

#rename columns
cbcl_ss.columns = cbcl_ss.columns.str.replace('scr_syn_', '')

"""
Export Files
"""
cbcl_ss.to_csv(export_directory/'cbcl_ss.csv', index=False)


