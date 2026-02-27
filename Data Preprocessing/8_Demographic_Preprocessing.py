#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: ABCD Demographics
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Cleaning demographic information from ABCD Study baseline wave. 

Inputs(s):
- abcd_p_demo.csv
- abcd_y_lt.csv

Output(s):
- demo_full.csv 
- demo_t0.csv

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
- https://nda.nih.gov/data_structure.html?short_name=pdem02
"""
#core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os  

#load data
demo_raw = pd.read_csv(import_directory / 'abcd_p_demo.csv')
family_raw = pd.read_csv(import_directory / 'abcd_y_lt.csv')

"""
Demographics
"""
demo_vars = demo_raw.columns.tolist()
family_vars = family_raw.columns.tolist()

#reduce dataset
keep_cols = ['src_subject_id', 
             'eventname', 
             'demo_prim', 
             'demo_brthdat_v2', 
             'demo_ed_v2', 
             'demo_sex_v2', 
             'demo_gender_id_v2', 
             'demo_comb_income_v2', 
             'race_ethnicity'
             ]
demo_raw = demo_raw[keep_cols]

drop_cols = ['school_id', 'district_id', 'visit_type']
family_raw = family_raw.drop(columns=drop_cols)

del demo_vars, family_vars, keep_cols, drop_cols

#merge datasets
demo_full = pd.merge(demo_raw, family_raw, on=['src_subject_id', 'eventname'], how='outer')

#full sample = 11868 
demo_t0 = demo_full[demo_full['eventname'] == 'baseline_year_1_arm_1']
demo_t0 = demo_t0.copy()
demo_t0['demo_sex_rc'] = demo_t0['demo_sex_v2'].replace({1: 1, 2: 2, 3: 1})

del demo_raw, family_raw

"""
Export Files
"""
demo_full.to_csv(export_directory/'demo_full.csv', index=False)
demo_t0.to_csv(export_directory/'demo_t0.csv', index=False)

