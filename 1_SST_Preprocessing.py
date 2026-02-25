#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Stop Signal Task (SST) Pre-processing 
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Cleaning inhibitory control behavioral task data from the Adolescent Brain Cognitive Development (ABCD) Study baseline and 2-year follow-up waves. 

Inputs(s):
- mri_y_tfmr_sst_beh.csv

Output(s):
- sst_bissett_garavan_t0.csv
- sst_bissett_garavan_t2.csv
- sst_bissett_garavan_t0_flags.csv
- sst_bissett_garavan_t2_flags.csv

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
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
sst_raw = pd.read_csv(import_directory/'mri_y_tfmr_sst_beh.csv')

"""
Completion
"""
#11502 participants completed at baseline
sst_raw_t0 = sst_raw[sst_raw['eventname'] == 'baseline_year_1_arm_1']
#7964 participants completed at 2-year follow-up
sst_raw_t2 = sst_raw[sst_raw['eventname'] == '2_year_follow_up_y_arm_1']

"""
Stop Signal Task (SST)

Logan, G. D. (1994). 
On the ability to inhibit thought and action: A user’s guide to the stop signal paradigm. 
In D. Dagenbach & T. H. Carr, Inhibitory processes in attention, memory, and language (pp. 189-239). 
San Diego: Academic Press.

Exclusion Criteria: Bissett et al., 2021 & Garavan et al., 2022

Bissett, P. G., Hagen, M. P., Jones, H. M., & Poldrack, R. A. (2021). 
Design issues and solutions for stop-signal data from the Adolescent Brain Cognitive Development (ABCD) study. 
eLife, 10, e60185. https://doi.org/10.7554/eLife.60185
- Remove participants who lack two complete runs (180 trials per run).

Garavan, H., Chaarani, B., Hahn, S., Allgaier, N., Juliano, A., Yuan, D. K., Orr, C., Watts, R., Wager, T. D., Ruiz de Leon, O., Hagler, D. J., Jr, & Potter, A. (2022). 
The ABCD stop signal data: Response to Bissett et al. 
Developmental cognitive neuroscience, 57, 101144. https://doi.org/10.1016/j.dcn.2022.101144
- Remove: 
    (1) correct go < 60%, 
    (2) incorrect go > 30%, 
    (3) late go > 30%, 
    (4) go ommissions >30%, 
    (5) total number go trials <300, and 
    (6) stop success rate <20% or > 80%.
"""
sst_bissett_garavan = sst_raw[(sst_raw['tfmri_sst_all_beh_crgo_rt'] >= 0.60) & #correct go >= 0.6
                      (sst_raw['tfmri_sst_all_beh_incrgo_rt'] <= 0.30) & #incorrect go <= 0.3
                      (sst_raw['tfmri_sst_all_beh_crlg_rt'] <= 0.30) & #late correct go <= 0.3
                      (sst_raw['tfmri_sst_all_beh_nrgo_rt'] <= 0.30) & #omitted go <= 0.3
                      (sst_raw['tfmri_sst_all_beh_go_nt'] >= 300) & # go trials >= 300 
                      (sst_raw['tfmri_sst_all_beh_crs_rt'] <= 0.80) & # >= 0.2
                      (sst_raw['tfmri_sst_all_beh_crs_rt'] >= 0.20) & # stop success rate <= 0.8
                      (sst_raw['tfmri_sst_nbeh_nruns'] == 2) & #runs = 2
                      (sst_raw['tfmri_sst_all_beh_total_nt'] == 360)] #total trials = 360

#at baseline 10110 participants meet both criteria 
sst_bissett_garavan_t0 = sst_bissett_garavan[sst_bissett_garavan['eventname'] == 'baseline_year_1_arm_1']
sst_bissett_garavan_t0.shape[0]

#at 2-year follow-up 7217 participants meet both criteria 
sst_bissett_garavan_t2 = sst_bissett_garavan[sst_bissett_garavan['eventname'] == '2_year_follow_up_y_arm_1']
sst_bissett_garavan_t2.shape[0]

del sst_bissett_garavan

sst_bissett_garavan_t0_flags = sst_bissett_garavan_t0[['src_subject_id', 'eventname', 'tfmri_sst_beh_violatorflag', 'tfmri_sst_beh_glitchflag', 'tfmri_sst_beh_0ssdcount']]
sst_bissett_garavan_t2_flags = sst_bissett_garavan_t2[['src_subject_id', 'eventname', 'tfmri_sst_beh_violatorflag', 'tfmri_sst_beh_glitchflag', 'tfmri_sst_beh_0ssdcount']]

rename_map = {
    'tfmri_sst_beh_violatorflag': 'violator_flag',
    'tfmri_sst_beh_glitchflag': 'glitch_flag',
    'tfmri_sst_beh_0ssdcount': 'ssd0_count'
}

sst_bissett_garavan_t0_flags = sst_bissett_garavan_t0_flags.rename(columns=rename_map)
sst_bissett_garavan_t2_flags = sst_bissett_garavan_t2_flags.rename(columns=rename_map)

del rename_map

"""
Reduce and Rename Columns
"""
#reduce dataset
sst_vars = [
    'src_subject_id', 
    'eventname',
    'tfmri_sst_all_beh_crgo_rt', 
    'tfmri_sst_all_beh_crgo_mrt', 
    'tfmri_sst_all_beh_crgo_stdrt', 
    'tfmri_sst_all_beh_crlg_rt', 
    'tfmri_sst_all_beh_crlg_mrt', 
    'tfmri_sst_all_beh_crlg_stdrt', 
    'tfmri_sst_all_beh_incrgo_rt', 
    'tfmri_sst_all_beh_incrgo_mrt', 
    'tfmri_sst_all_beh_incrgo_stdrt', 
    'tfmri_sst_all_beh_incrlg_rt', 
    'tfmri_sst_all_beh_incrlg_mrt', 
    'tfmri_sst_all_beh_incrlg_stdrt', 
    'tfmri_sst_all_beh_nrgo_rt',
    'tfmri_sst_all_beh_crs_rt', 
    'tfmri_sst_all_beh_incrs_rt', 
    'tfmri_sst_all_beh_incrs_mrt', 
    'tfmri_sst_all_beh_incrs_stdrt', 
    'tfmri_sst_all_beh_ssds_rt', 
    'tfmri_sst_all_beh_tot_mssd', 
    'tfmri_sst_all_beh_total_mssrt', 
    'tfmri_sst_all_beh_total_issrt'
    ]

sst_bissett_garavan_t0 = sst_bissett_garavan_t0[sst_vars]
sst_bissett_garavan_t0.shape

sst_bissett_garavan_t2 = sst_bissett_garavan_t2[sst_vars]
sst_bissett_garavan_t2.shape

del sst_raw, sst_vars

#rename features
sst_bissett_garavan_t0.columns = sst_bissett_garavan_t0.columns.str.replace('tfmri_', '').str.replace('all_beh_', '').str.replace('_rt', '_rate')
sst_bissett_garavan_t2.columns = sst_bissett_garavan_t2.columns.str.replace('tfmri_', '').str.replace('all_beh_', '').str.replace('_rt', '_rate')

rename_map = {
    'sst_tot_mssd': 'sst_mssd',
    'sst_total_mssrt': 'sst_mssrt',
    'sst_total_issrt': 'sst_issrt'
}
sst_bissett_garavan_t0 = sst_bissett_garavan_t0.rename(columns=rename_map)
sst_bissett_garavan_t2 = sst_bissett_garavan_t2.rename(columns=rename_map)

"""
Export Files
"""
sst_bissett_garavan_t0.to_csv(export_directory/'sst_bissett_garavan_t0.csv', index=False)
sst_bissett_garavan_t2.to_csv(export_directory/'sst_bissett_garavan_t2.csv', index=False)

sst_bissett_garavan_t0_flags.to_csv(export_directory/'sst_bissett_garavan_t0_flags.csv', index=False)
sst_bissett_garavan_t2_flags.to_csv(export_directory/'sst_bissett_garavan_t2_flags.csv', index=False)
