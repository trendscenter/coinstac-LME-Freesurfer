#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script includes the local computations for linear mixed effects model
with decentralized statistic calculation
"""

import json
import numpy as np
import sys
import regression as reg
import warnings
from itertools import chain
from lme_utils import *
from data_utils import *
import csv
import os
import pandas as pd
import npMatrix3d
import itertools

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import statsmodels.api as sm

"""
============================================================================
The below function does the following tasks
1. read the dependent variables, fixed and random covariates from csv files
and forms the X,Y and Z matrices
2. calculate the product matrices
3. solves LME model using pseudo Simplified Fisher Scoring algorithm and
outputs the parameter estimates and inference results
----------------------------------------------------------------------------
This function takes in the following inputs in args['input']:
----------------------------------------------------------------------------
- fixed_covariates : csv file containing fixed covariates, each row for an
observation
- dependents : csv file containing list of freesurfer stats file, each row
for an observation
- random_factor : csv file containing random factor levels for each
observation (contains one column as only 1 random factor)
- random_covariates : csv file containing design matrix for the random
factor
- freesurfer_variables : freesurfer regions to be used as response/dependent
variables
- contrasts : list of contrasts. Contrast vectors to be tested.
Each contrast should contain the fields:
name: A name for the contrast. i.e. Contrast1.
vector: A vector for the contrast.
It must be one dimensional for a T test and two dimensional for an F test
For eg., [1, 0, 0] (T contrast) or [[1, 0, 0],[0,1,0]] (F contrast)
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    nlevels : list containing number of levels for each random factor, as we
    have only one random factor, nlevels is a single value
    nobservns : number of observations
    computation_phase : local_0
- cache:
    X : fixed effects design matrix
    Y : response matrix
    fs_vars : freesurfer regions to be used as response/dependent variables
    ranfac : vector defining level for each observation
    raneffs : vector containing random effects
    contrasts : list of contrasts
    paramVec_local : dict containing estimates and inference outputs of LME.
    for detailed specification of paramVec refer compspec 'output'
============================================================================
"""
def local_0(args):

    input_list = args['input']
    state_list = args['state']

    inputdir = state_list['baseDirectory']
    cache_dir = state_list["cacheDirectory"]

    fc = input_list['fixed_covariates']
    fs_vars = input_list['freesurfer_variables']
    dep = input_list['dependents']
    rf = input_list['random_factor']
    rc = input_list['random_covariates']

    [X,Y,Z,ranfac,raneffs] = form_XYZMatrices(inputdir,fc,fs_vars,dep,rf,rc)

    raise Exception(Z)

    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats3D(X,Y,Z)

    n = len(X)
    nlevels = np.array([np.shape(Z)[1]])
    nraneffs = np.array([1])
    tol = 1e-6
    paramVec = reg.pSFS3D(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nlevels, nraneffs,tol,n)

    nfixeffs = np.shape(X)[1]
    ndepvars = np.shape(YtX)[0]
    [beta,sigma2,vechD,D] = get_parameterestimates(paramVec,nfixeffs,ndepvars,nlevels,nraneffs)

    contrasts=input_list['contrasts']
    prod_matrices = [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
    [llh,resms,covB,tstats,fstats] = reg.cal_inference(prod_matrices,n,nfixeffs,ndepvars,nlevels,
                                                        nraneffs,beta,sigma2,D,contrasts)

    dict_list = gen_compoutputdict(beta,sigma2,vechD,llh,resms,covB,tstats,fstats,ndepvars)

    # Writing covariates and dependents to cache as files
    X=np.array(X)
    Y=np.array(Y)
    saveBin(os.path.join(cache_dir, 'X.npy'), X)
    saveBin(os.path.join(cache_dir, 'Y.npy'), Y)

    computation_output_dict = {
        'output':
        {
            'nlevels': nlevels.tolist(),
            'nobservns': n,
            'computation_phase': 'local_0'
        },
        'cache':
        {
            'X': 'X.npy',
            'Y': 'Y.npy',
            'fs_vars': fs_vars,
            'ranfac': ranfac.tolist(),
            'raneffs': raneffs,
            'contrasts': contrasts,
            'paramVec_local': dict_list
        }
    }

    return json.dumps(computation_output_dict)


"""
============================================================================
The below function does the following tasks
1. form Z matrix
2. calculate product matrices
3. send the product matrices to remote_1
----------------------------------------------------------------------------
This function takes in the following inputs in args:
----------------------------------------------------------------------------
- input :
    nlevels_persite : list containing number of levels of random factor for
    each site
    nlevels_global : total levels summed up for all local sites
    nlocalsites : number of local sites
    computation_phase : remote_0
- cache:
    X : fixed effects design matrix
    Y : response variables
    ranfac : vector defining level for each observation
    raneffs : vector containing random effects
    contrasts : list of contrasts
    paramVec_local : dict containing estimates and inference outputs of LME.
    for detailed specification of paramVec refer compspec 'output'
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    XtransposeX_local
    XtransposeY_local
    XtransposeZ_local
    YtransposeX_local
    YtransposeY_local
    YtransposeZ_local
    ZtransposeX_local
    ZtransposeY_local
    ZtransposeZ_local
    contrasts : list of contrasts
    fs_vars : freesurfer regions to be used as response/dependent variables
    paramVec_local : dict containing estimates and inference outputs of LME.
    for detailed specification of paramVec refer compspec 'output'
    computation_phase : local_1
============================================================================
"""
def local_1(args):

    cache_list = args['cache']
    input_list = args['input']
    state_list = args['state']

    cache_dir = state_list["cacheDirectory"]
    transfer_dir = state_list["transferDirectory"]

    X = loadBin(os.path.join(cache_dir, cache_list['X']))
    Y = loadBin(os.path.join(cache_dir, cache_list['Y']))

    ranfac = cache_list['ranfac']
    raneffs = cache_list['raneffs']

    n = len(X)
    nlevels_persite = input_list['nlevels_persite']
    nlevels_global = input_list['nlevels_global']
    nlocalsites = input_list['nlocalsites']
    clientId = args['state']['clientId']

    Z = form_globalZMatrix(nlevels_persite,nlevels_global,nlocalsites,clientId,ranfac,raneffs)

    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats3D(X,Y,Z)

    prod_matrices = [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
    prod_matrices_name = ['XtX','XtY','XtZ','YtX','YtY','YtZ','ZtX','ZtY','ZtZ']

    for p in range(len(prod_matrices)):
        saveBin(os.path.join(transfer_dir, prod_matrices_name[p]+'.npy'),prod_matrices[p])

    computation_output_dict = {
        'output': {
            'XtransposeX_local': 'XtX.npy',
            'XtransposeY_local': 'XtY.npy',
            'XtransposeZ_local': 'XtZ.npy',
            'YtransposeX_local': 'YtX.npy',
            'YtransposeY_local': 'YtY.npy',
            'YtransposeZ_local': 'YtZ.npy',
            'ZtransposeX_local': 'ZtX.npy',
            'ZtransposeY_local': 'ZtY.npy',
            'ZtransposeZ_local': 'ZtZ.npy',
            'contrasts': cache_list['contrasts'],
            'paramVec_local': cache_list['paramVec_local'],
            'fs_vars': cache_list['fs_vars'],
            'computation_phase': 'local_1',
        }
    }
    return json.dumps(computation_output_dict)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())

    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_0(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_0' in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Error occurred at Local')
