#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Post-Hoc T-Tests (Main Analyses)
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Packages: 
- Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 14:46:42) [Clang 14.0.6 ]
- pandas version: 2.2.1
- numpy version: 1.26.4

Notes:
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import platform 
import scipy
from scipy.stats import ttest_ind, levene, fisher_exact, skew, kurtosis
import statsmodels
import os

#load task data
abcd_all = pd.read_csv(import_directory/'10_Analyses'/'abcd_all.csv')

"""
Function(s)
"""
def independent_t(df, normal_vars, group_col='demo_sex_rc'):
    results = []
    for var in normal_vars:
        if var not in df.columns:
            continue
        group_0 = df.loc[df[group_col] == 0, var].dropna()
        group_1 = df.loc[df[group_col] == 1, var].dropna()
        n0, n1 = len(group_0), len(group_1)
        if n0 < 2 or n1 < 2:
            continue
        mean0, mean1 = group_0.mean(), group_1.mean()
        sd0, sd1 = group_0.std(ddof=1), group_1.std(ddof=1)
        levene_stat, levene_p = levene(group_0, group_1)
        equal_var = levene_p >= 0.05
        t_stat, p_val = ttest_ind(group_0, group_1, equal_var=equal_var)
        if equal_var:
            pooled_sd = np.sqrt(
                ((n0 - 1) * sd0**2 + (n1 - 1) * sd1**2) / (n0 + n1 - 2))
            effect_size = (mean0 - mean1) / pooled_sd
        else:
            effect_size = (mean0 - mean1) / np.sqrt((sd0**2 + sd1**2) / 2)
        results.append({
            'variable': var,
            'p_value': p_val,
            't_value': t_stat,
            'cohens_d': effect_size, #cohen's d or welch's d
            'levene_val': levene_stat,
            'levene_pval': levene_p,
            'equal_var': equal_var,
            'group_0_n': n0,
            'group_0_mean': mean0,
            'group_0_sd': sd0,
            'group_1_n': n1,
            'group_1_mean': mean1,
            'group_1_sd': sd1
        })
    return pd.DataFrame(results)


"""
Frequencies & Descriptives
"""
group_0 = abcd_all.loc[abcd_all['demo_sex_rc'] == 0.0]  # male
group_1 = abcd_all.loc[abcd_all['demo_sex_rc'] == 1.0]  # female

g0_skew = skew(group_0['sst_t0_mssrt'], nan_policy='omit') #np.float64(0.26979534631695823)
g0_kurt = kurtosis(group_0['sst_t0_mssrt'], nan_policy='omit') #np.float64(1.1606231758838277)
g1_skew = skew(group_1['sst_t0_mssrt'], nan_policy='omit') #np.float64(0.4623686185150385)
g1_kurt = kurtosis(group_1['sst_t0_mssrt'], nan_policy='omit') #np.float64(1.0591583516602086)

g0_skew = skew(group_0['sst_t2_mssrt'], nan_policy='omit') #np.float64(0.027745135870015904)
g0_kurt = kurtosis(group_0['sst_t2_mssrt'], nan_policy='omit') #np.float64(0.8173771626131972)
g1_skew = skew(group_1['sst_t2_mssrt'], nan_policy='omit') #np.float64(0.1180126245689722)
g1_kurt = kurtosis(group_1['sst_t2_mssrt'], nan_policy='omit') #np.float64(1.0907069409667498)

g0_skew = skew(group_0['nback_t0_c2b_rate'], nan_policy='omit') #np.float64(-0.2699931880694372)
g0_kurt = kurtosis(group_0['nback_t0_c2b_rate'], nan_policy='omit') #np.float64(-0.6237289645619586)
g1_skew = skew(group_1['nback_t0_c2b_rate'], nan_policy='omit') #np.float64(-0.05057519124034819)
g1_kurt = kurtosis(group_1['nback_t0_c2b_rate'], nan_policy='omit') #np.float64(-0.6398453100696511)

g0_skew = skew(group_0['nback_t2_c2b_rate'], nan_policy='omit') #np.float64(-0.9113115910450934)
g0_kurt = kurtosis(group_0['nback_t2_c2b_rate'], nan_policy='omit') #np.float64(0.602215801789546)
g1_skew = skew(group_1['nback_t2_c2b_rate'], nan_policy='omit') #np.float64(-0.6757009517591643)
g1_kurt = kurtosis(group_1['nback_t2_c2b_rate'], nan_policy='omit') #np.float64(-0.09584123355907304)

sex_t = independent_t(abcd_all, normal_vars=['sst_t0_mssrt', 'sst_t2_mssrt', 'nback_t0_c2b_rate', 'nback_t2_c2b_rate'], group_col='demo_sex_rc')









