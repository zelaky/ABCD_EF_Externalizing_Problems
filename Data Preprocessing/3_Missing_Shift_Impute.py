#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Missing Imputation: Log Normal Shift
Author: Zoë E. Laky, M.A.
Contact: zoe.laky@nih.gov

Project Description: 
- Shift imputation of missing in the inhibitory control and working memory behavioral task data from the Adolescent Brain Cognitive Development (ABCD) Study baseline and 2-year follow-up waves. 

Inputs(s):
- sst_bissett_garavan_t0.csv
- sst_bissett_garavan_t2.csv

Output(s):
- sst_t0_shift_impute.csv
- sst_t2_shift_impute.csv

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
from tqdm import tqdm
import warnings

#statistical suite
import scipy
from scipy.stats import norm, skew, kurtosis  

#load data
sst_t0_shift_impute = pd.read_csv(export_directory/'sst_bissett_garavan_t0.csv')
sst_t2_shift_impute = pd.read_csv(export_directory/'sst_bissett_garavan_t2.csv')

"""
Functions
"""
def stats_conditional(shift, mu, sigma, miss_lb, miss_ub, reso=100000, norm_values=None): 
    if norm_values is None:
        norm_values = norm.ppf(np.linspace(1/reso, 1 - (1/reso), reso-1))
    dist_vals = shift + np.exp(mu + sigma*norm_values)
    cond_vals = dist_vals[np.logical_and(dist_vals >= miss_lb, dist_vals <= miss_ub)]
    if len(cond_vals) == 0:
        print("Not enough values in the sample. Increase resolution.")
        return np.nan, np.nan
    return np.mean(cond_vals), np.std(cond_vals)

def find_pairs_long(shift, emp_mean, emp_std, obs_lb, obs_ub, optim_delt, optim_mult, n_its_outer, n_its_inner, reso=100000, norm_values=None):
    if norm_values is None:
        norm_values = norm.ppf(np.linspace(1/reso, 1 - (1/reso), reso-1))
    if emp_mean < shift:
        print("Shifted log-normal imputation not possible since empirical mean < shift.")
        return (np.nan, np.nan)
    #initial value
    sln_mu = np.log(emp_mean - shift)
    sln_sigma = 0

    #find bisection endpoint for sigma
    sigma_endpoint = optim_delt
    max_sigma_threshold = 1e15

    flag = True
    while flag:
        calc_mean, calc_std = stats_conditional(shift, sln_mu, sigma_endpoint, obs_lb, obs_ub)
        if calc_std > emp_std:
            flag=False
        else:
            sigma_endpoint = optim_mult * sigma_endpoint
            if sigma_endpoint > max_sigma_threshold:  
                print("Sigma endpoint diverged to infinity. Skipping row.")
                return np.nan, np.nan
    #find bisection endpoint for mu
    if calc_mean > emp_mean:
        mu_endpoint_r = sln_mu
        mu_endpoint_l = 0
    else:
        mu_endpoint_l = sln_mu
        mu_endpoint_r = sln_mu
        flag = True
        while flag:
            calc_mean, calc_std = stats_conditional(shift, mu_endpoint_r, sigma_endpoint, obs_lb, obs_ub)
            if calc_mean > emp_mean:
                flag=False
            else:
                mu_endpoint_r = optim_delt + mu_endpoint_r
    #alternating bisection searches
    for outer_it in range(n_its_outer):
        # check if sigma endpoint is sufficient
        flag = True
        while flag:
            calc_mean, calc_std = stats_conditional(shift, sln_mu, sigma_endpoint, obs_lb, obs_ub)
            if calc_std > emp_std:
                flag=False
            else:
                sigma_endpoint = optim_mult * sigma_endpoint
                if sigma_endpoint > max_sigma_threshold:  
                    print("Sigma endpoint diverged to infinity. Skipping row.")
                    return np.nan, np.nan
        #check if mu endpoint is sufficient
        flag = True
        while flag:
            calc_mean, calc_std = stats_conditional(shift, mu_endpoint_r, sigma_endpoint, obs_lb, obs_ub)
            if calc_mean > emp_mean:
                flag=False
            else:
                mu_endpoint_r = optim_delt + mu_endpoint_r
        #search for sigma
        current_sigma_endpoint_l = 0
        current_sigma_endpoint_r = sigma_endpoint
        for inner_it in range(n_its_inner):
            guess_sigma = (current_sigma_endpoint_l + current_sigma_endpoint_r)/2
            calc_mean, calc_std = stats_conditional(shift, sln_mu, guess_sigma, obs_lb, obs_ub)
            if calc_std < emp_std:
                current_sigma_endpoint_l = guess_sigma
            else:
                current_sigma_endpoint_r = guess_sigma
        sln_sigma = guess_sigma
        #search for mu
        current_mu_endpoint_l = mu_endpoint_l
        current_mu_endpoint_r = mu_endpoint_r
        for inner_it in range(n_its_inner):
            guess_mu = (current_mu_endpoint_l + current_mu_endpoint_r)/2
            calc_mean, calc_std = stats_conditional(shift, guess_mu, sln_sigma, obs_lb, obs_ub)
            if calc_mean < emp_mean:
                current_mu_endpoint_l = guess_mu
            else:
                current_mu_endpoint_r = guess_mu
        sln_mu = guess_mu
    return sln_mu, sln_sigma

def find_pairs_short(shift, obs_mean, obs_std):
    sigma = np.sqrt(np.log((obs_std/(obs_mean - shift))**2 + 1))
    mu = np.log(obs_mean - shift) - sigma**2/2
    return mu, sigma

"""
SST: Correct Late Go Imputation
"""
warnings.filterwarnings("ignore")

#check missing
col_miss_t0 = (sst_t0_shift_impute.isnull().mean() * 100).round(2)
col_miss_t2 = (sst_t2_shift_impute.isnull().mean() * 100).round(2)

long_method = True  

#baseline
reso_0 = 10_000_000
norm_values_0 = norm.ppf(np.linspace(1/reso_0, 1 - (1/reso_0), reso_0-1))
crgo_shift_0 = (150 + sst_t0_shift_impute['sst_crgo_mrt'].min()) / 2 #reaction time speeds < 150ms unlikely related to inhibitory processess

missing_rows_t0 = sst_t0_shift_impute[
    sst_t0_shift_impute[['src_subject_id', 'eventname', 'sst_crgo_mrt', 'sst_crgo_stdrt', 'sst_crlg_mrt', 'sst_crlg_stdrt']].isnull().any(axis=1)
]

for index, row in tqdm(missing_rows_t0.iterrows(), total=len(missing_rows_t0)):
    ref_mean, ref_std = row['sst_crgo_mrt'], row['sst_crgo_stdrt']

    sln_mu, sln_sigma = np.nan, np.nan

    if long_method:
        sln_mu, sln_sigma = find_pairs_long(
            shift=crgo_shift_0,
            emp_mean=ref_mean,
            emp_std=ref_std,
            obs_lb=0,
            obs_ub=1000,
            optim_delt=0.01,
            optim_mult=1.1,
            n_its_outer=20,
            n_its_inner=20,
            reso=reso_0,
            norm_values=norm_values_0
        )
        # fallback to short method if long method fails
        if np.isnan(sln_mu) or np.isnan(sln_sigma):
            sln_mu, sln_sigma = find_pairs_short(
                shift=crgo_shift_0,
                obs_mean=ref_mean,
                obs_std=ref_std
            )
    else:
        sln_mu, sln_sigma = find_pairs_short(
            shift=crgo_shift_0,
            obs_mean=ref_mean,
            obs_std=ref_std
        )

    impute_mean, impute_std = stats_conditional(
        shift=crgo_shift_0,
        mu=sln_mu,
        sigma=sln_sigma,
        miss_lb=1000,
        miss_ub=2000,
        reso=reso_0,
        norm_values=norm_values_0
    )

    sst_t0_shift_impute.loc[index, 'sst_crlg_mrt'] = np.where(
        pd.isna(sst_t0_shift_impute.loc[index, 'sst_crlg_mrt']),
        impute_mean,
        sst_t0_shift_impute.loc[index, 'sst_crlg_mrt']
    )
    sst_t0_shift_impute.loc[index, 'sst_crlg_stdrt'] = np.where(
        pd.isna(sst_t0_shift_impute.loc[index, 'sst_crlg_stdrt']),
        impute_std,
        sst_t0_shift_impute.loc[index, 'sst_crlg_stdrt']
    )

col_miss_t0 = (sst_t0_shift_impute.isnull().mean() * 100).round(2)

#2-year follow-up
reso_2 = 10_000_000
norm_values_2 = norm.ppf(np.linspace(1/reso_2, 1 - (1/reso_2), reso_2-1))
crgo_shift_2 = (150 + sst_t2_shift_impute['sst_crgo_mrt'].min()) / 2 

missing_rows_t2 = sst_t2_shift_impute[
    sst_t2_shift_impute[['src_subject_id', 'eventname', 'sst_crgo_mrt', 'sst_crgo_stdrt', 'sst_crlg_mrt', 'sst_crlg_stdrt']].isnull().any(axis=1)
]

for index, row in tqdm(missing_rows_t2.iterrows(), total=len(missing_rows_t2)):
    ref_mean, ref_std = row['sst_crgo_mrt'], row['sst_crgo_stdrt']

    sln_mu, sln_sigma = np.nan, np.nan

    if long_method:
        sln_mu, sln_sigma = find_pairs_long(
            shift=crgo_shift_2,
            emp_mean=ref_mean,
            emp_std=ref_std,
            obs_lb=0,
            obs_ub=1000,
            optim_delt=0.01,
            optim_mult=1.1,
            n_its_outer=20,
            n_its_inner=20,
            reso=reso_2,
            norm_values=norm_values_2
        )
        # fallback to short method if long method fails
        if np.isnan(sln_mu) or np.isnan(sln_sigma):
            sln_mu, sln_sigma = find_pairs_short(
                shift=crgo_shift_2,
                obs_mean=ref_mean,
                obs_std=ref_std
            )
    else:
        sln_mu, sln_sigma = find_pairs_short(
            shift=crgo_shift_2,
            obs_mean=ref_mean,
            obs_std=ref_std
        )

    impute_mean, impute_std = stats_conditional(
        shift=crgo_shift_2,
        mu=sln_mu,
        sigma=sln_sigma,
        miss_lb=1000,
        miss_ub=2000,
        reso=reso_2,
        norm_values=norm_values_2
    )

    sst_t2_shift_impute.loc[index, 'sst_crlg_mrt'] = np.where(
        pd.isna(sst_t2_shift_impute.loc[index, 'sst_crlg_mrt']),
        impute_mean,
        sst_t2_shift_impute.loc[index, 'sst_crlg_mrt']
    )
    sst_t2_shift_impute.loc[index, 'sst_crlg_stdrt'] = np.where(
        pd.isna(sst_t2_shift_impute.loc[index, 'sst_crlg_stdrt']),
        impute_std,
        sst_t2_shift_impute.loc[index, 'sst_crlg_stdrt']
    )

col_miss_t2 = (sst_t2_shift_impute.isnull().mean() * 100).round(2)

warnings.filterwarnings("default")

"""
Export Files
"""
sst_t0_shift_impute.to_csv(export_directory / 'sst_t0_shift_impute.csv', index=False)
sst_t2_shift_impute.to_csv(export_directory / 'sst_t2_shift_impute.csv', index=False)
