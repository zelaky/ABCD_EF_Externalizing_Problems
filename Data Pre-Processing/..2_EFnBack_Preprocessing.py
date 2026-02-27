"""
Title: Emotional Face n-Back (EFnBack) Pre-processing 
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Cleaning working memory behavioral task from the ABCD Study baseline and 2-year follow-up waves. 

Inputs(s):
- mri_y_tfmr_nback_beh.csv

Output(s):
- nback_manual_t0.csv
- nback_manual_t2.csv

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
- ABCD Study 'tfmri_nback_beh_performflag' exclusion criteria is based on: 
    (1) neuroimaging event related criteria, and 
    (2) average accuracy < 60% on 0-back and 2-back trials.
- https://wiki.abcdstudy.org/release-notes/non-imaging/neurocognition.html
"""
#core
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os  

#load data
nback_raw = pd.read_csv(import_directory/'mri_y_tfmr_nback_beh.csv')

"""
Completion
"""
#11463 participants completed at baseline
nback_raw_t0 = nback_raw[nback_raw['eventname'] == 'baseline_year_1_arm_1']
#7974 participants completed at 2-year follow-up
nback_raw_t2 = nback_raw[nback_raw['eventname'] == '2_year_follow_up_y_arm_1']

"""
Emotional Face n-Back Task (Nback)

Ladouceur, C. D., Silk, J. S., Dahl, R. E., Ostapenko, L., Kronhaus, D. M., & Phillips, M. L. (2009). 
Fearful faces influence attentional control processes in anxious youth and adults. 
Emotion, 9(6), 855-864.

Exclusion Criteria: Manually Calculated
- Remove particpants with average accuracy < 60% on 0-back and 2-back trials.
"""
#participants with: (1) 0-back accuracy >= 0.60, (2) 
nback_manual = nback_raw[(nback_raw['tfmri_nb_all_beh_c0b_rate'] >= 0.60) & #0-back accuracy >= 0.60
                         (nback_raw['tfmri_nb_all_beh_c2b_rate'] >= 0.60)] #2-back accuracy >= 0.60

#at baseline 9751 participants meet manual criteria 
nback_manual_t0 = nback_manual[nback_manual['eventname'] == 'baseline_year_1_arm_1']
nback_manual_t0.shape[0]

#at 2-year follow-up 7464 participants meet manual criteria 
nback_manual_t2 = nback_manual[nback_manual['eventname'] == '2_year_follow_up_y_arm_1']
nback_manual_t2.shape[0]

"""
Reduce and Rename Columns
"""
#reduce dataset
nback_vars = [ 
    'src_subject_id',
    'eventname',
    'tfmri_nb_all_beh_c2bpf_rate', 
    'tfmri_nb_all_beh_c2bpf_mrt', 
    'tfmri_nb_all_beh_c2bpf_stdrt', 
    'tfmri_nb_all_beh_c2bnf_rate', 
    'tfmri_nb_all_beh_c2bnf_mrt', 
    'tfmri_nb_all_beh_c2bnf_stdrt', 
    'tfmri_nb_all_beh_c2bp_rate', 
    'tfmri_nb_all_beh_c2bp_mrt', 
    'tfmri_nb_all_beh_c2bp_stdrt', 
    'tfmri_nb_all_beh_c0bpf_rate', 
    'tfmri_nb_all_beh_c0bpf_mrt', 
    'tfmri_nb_all_beh_c0bpf_stdrt', 
    'tfmri_nb_all_beh_c0bnf_rate', 
    'tfmri_nb_all_beh_c0bnf_mrt', 
    'tfmri_nb_all_beh_c0bnf_stdrt', 
    'tfmri_nb_all_beh_c0bngf_rate', 
    'tfmri_nb_all_beh_c0bngf_mrt', 
    'tfmri_nb_all_beh_c0bngf_stdrt', 
    'tfmri_nb_all_beh_c0bp_rate', 
    'tfmri_nb_all_beh_c0bp_mrt', 
    'tfmri_nb_all_beh_c0bp_stdrt', 
    'tfmri_nb_all_beh_c2bngf_rate', 
    'tfmri_nb_all_beh_c2bngf_mrt', 
    'tfmri_nb_all_beh_c2bngf_stdrt',
    'tfmri_nb_all_beh_c0b_rate', 
    'tfmri_nb_all_beh_c0b_mrt',
    'tfmri_nb_all_beh_c0b_stdrt',
    'tfmri_nb_all_beh_c2b_rate',
    'tfmri_nb_all_beh_c2b_mrt',
    'tfmri_nb_all_beh_c2b_stdrt'
    ]

nback_manual_t0 = nback_manual_t0[nback_vars]
nback_manual_t0.shape

nback_manual_t2 = nback_manual_t2[nback_vars]
nback_manual_t2.shape

del nback_raw, nback_vars

#rename baseline features
nback_manual_t0.columns = nback_manual_t0.columns.str.replace('tfmri_nb_all_beh', 'nback')

#rename 2-year follow-up features
nback_manual_t2.columns = nback_manual_t2.columns.str.replace('tfmri_nb_all_beh', 'nback')

"""
Export Files
"""
nback_manual_t0.to_csv(export_directory/'nback_manual_t0.csv', index=False)
nback_manual_t2.to_csv(export_directory/'nback_manual_t2.csv', index=False)


































