#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys
import npMatrix3d
import pandas as pd
import os
import itertools
from data_utils import *
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm

np.set_printoptions(threshold=sys.maxsize)

'''
=============================================================================
The below function forms the matrices X, Y and Z.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- inputdir: base directory containing input csv files.
- fc: csv file containing fixed covariates, each row for an observation
- fs_vars: freesurfer regions to be used as response/dependent variables
- dep : csv file containing list of freesurfer stats file, each row for an 
observation
- rf : csv file containing random factor levels for each observation (contains
one column as only 1 random factor)
- rc : csv file containing design matrix for the random factor
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - X: Fixed effects design matrix of dimension n times p
 - Y: The response matrix of dimension n times v
 - Z: The random effects design matrix of dimension n times q
 - ranfac: Vector containing site numbers of dimension n times 1
 - raneffs: Random effects design matrix of dimension n times 1
=============================================================================
'''
def form_XYZMatrices(inputdir,fc,fs_vars,dep,rf,rc):

    data_f = pd.read_csv(os.path.join(inputdir,fc))
    covariates=[]
    covariates.extend(['const'])
    covariates.extend(list(data_f.columns))
    
    data_f['isControl'] = data_f['isControl']*1
    cols_categorical = [col for col in data_f if data_f[col].dtype == object]
    cols_mono = [col for col in data_f if data_f[col].nunique() == 1]
    
    # Dropping columsn with unique values
    data_f = data_f.drop(columns=cols_mono)
    
    # Creating dummies on non-unique categorical variables
    cols_nodrop = set(cols_categorical) - set(cols_mono)
    data_f = pd.get_dummies(data_f, columns=cols_nodrop, drop_first=True)
    
    data_f = data_f.dropna(axis=0, how='any')
    data_f = data_f.to_numpy()

    X = sm.add_constant(data_f)
    X = X.tolist()
    n=len(X)
    
    files = pd.read_csv(os.path.join(inputdir,dep))
    files = files.to_numpy()
    Y=[]
    for f in files:
        y = pd.read_csv(os.path.join(inputdir,f[0]),delimiter="\t")
        y1=y[y[y.columns[0]].isin(fs_vars)]
        y1=y1[y.columns[1]].to_list()
        Y.append(y1)

    ranfac = pd.read_csv(os.path.join(inputdir,rf))
    ranfac = ranfac.to_numpy()
    nlevels = np.max(ranfac)

    raneffs = pd.read_csv(os.path.join(inputdir,rc))
    raneffs = raneffs[raneffs.columns[0]].tolist()

    Z = np.zeros([n, nlevels], dtype=int)
    for i in range(n):
        Z[i][ranfac[i][0]-1] = raneffs[i]
    
    return(X,Y,Z,ranfac,raneffs)

'''
=============================================================================
The below function generates the product matrices from matrices X, Y and Z.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
 - X: The design matrix of dimension n times p.
 - Y: The response vector of dimension v times n times 1*.
 - Z: The random effects design matrix of dimension n times q.
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - XtX: X transposed multiplied by X.
 - XtY: X transposed multiplied by Y.
 - XtZ: X transposed multiplied by Z.
 - YtX: Y transposed multiplied by X.
 - YtY: Y transposed multiplied by Y.
 - YtZ: Y transposed multiplied by Z.
 - ZtX: Z transposed multiplied by X.
 - ZtY: Z transposed multiplied by Y.
 - ZtZ: Z transposed multiplied by Z.
=============================================================================
'''
def prodMats3D(X,Y,Z):

    X=np.array(X)
    Y=np.transpose(np.array(Y))
    Y1=np.zeros([np.shape(Y)[0],np.shape(Y)[1],1])
    Y1[:,:,0]=Y
    Y=Y1
    Z=np.array(Z)
    
    # Work out the product matrices (non spatially varying)
    XtX = (X.transpose() @ X).reshape(1, X.shape[1], X.shape[1])
    XtY = X.transpose() @ Y
    XtZ = (X.transpose() @ Z).reshape(1, X.shape[1], Z.shape[1])
    YtX = XtY.transpose(0,2,1)
    YtY = Y.transpose(0,2,1) @ Y
    YtZ = Y.transpose(0,2,1) @ Z
    ZtX = XtZ.transpose(0,2,1)
    ZtY = YtZ.transpose(0,2,1)
    ZtZ = (Z.transpose() @ Z).reshape(1, Z.shape[1], Z.shape[1])    

    # Return product matrices
    return(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ)

'''
=============================================================================
The below function extracts the parameters estimated in LME from paramVec and 
reconstructs D .
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
 - paramVec: parameter estimates output from pSFS3D.
 - p: number of fixed effects.
 - v: number of freesurfer variables/regions
 - nlevels: number of levels of random factor
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - beta : The fixed effects parameter estimates for each fs region
 - sigma2 : The fixed effects variance estimate for each fs region
 - Ddict : unique element of random effects covariance matrix
 - D : The random effects covariance matrix estimate for each fs region
=============================================================================
'''
def get_parameterestimates(paramVec,p,v,nlevels,nraneffs):

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

     # Output beta estimate
    beta = paramVec[:, 0:p]
    
    # Output sigma2 estimate
    sigma2 = paramVec[:,p:(p+1),:]

    # Get the indices in the paramvector corresponding to D matrices
    IndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)//2) + p + 1)
    IndsDk = np.insert(IndsDk,0,p+1)

    # Reconstruct D
    Ddict = dict()
    # D as a dictionary

    Ddict[0] = npMatrix3d.vech2mat3D(paramVec[:,IndsDk[0]:IndsDk[1],:])
    
    # Full version of D
    D = npMatrix3d.getDfromDict3D(Ddict, nraneffs, nlevels)

    return(beta,sigma2,Ddict[0],D)

'''
=============================================================================
The below function generates the global Z matrix including random factors and
effects from all local sites.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- nlevels_persite : list containing number of levels of random factor for 
    each site
- nlevels_global : total levels summed up for all local sites
- nlocalsites : number of local sites
- clientId : local site id
- ranfac : vector defining level for each observation
- raneffs : vector containing random effects
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
- Z : global random effects design matrix of dimension n times q.
=============================================================================
'''
def form_globalZMatrix(nlevels_persite,nlevels_global,nlocalsites,clientId,ranfac,raneffs):

    nlevels_persite = np.array(nlevels_persite)
    n = len(raneffs)
    Z = np.zeros([n, nlevels_global], dtype=int)
    col_start = 0

    for s in range(nlocalsites):
        if (clientId=='local'+str(s)):
            if s==0:
                col_start = 0
            else:
                for i in range(s):
                    col_start = col_start+nlevels_persite[i]
    
    
    ranfac = np.array(ranfac)
    for i in range(n):
        Z[i][col_start+ranfac[i][0]-1] = raneffs[i]

    return(Z)

'''
=============================================================================
The below function generates the output dictionary, this includes,
- global stats and local stats
each of the stats contains
- parameter estimates and inference stats
- parameter estimates:
    beta
    sigma2
    vechD
- inference stats
    llh : 
    resms
    covB
    tstats
    fstats
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- nlevels_persite : list containing number of levels of random factor for 
    each site
- nlevels_global : total levels summed up for all local sites
- nlocalsites : number of local sites
- clientId : local site id
- ranfac : vector defining level for each observation
- raneffs : vector containing random effects
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
- Z : global random effects design matrix of dimension n times q.
=============================================================================
'''
def gen_compoutputdict(beta,sigma2,vechD,llh,resms,covB,tstats,fstats,ndepvars):

    param_estimates_keys = ['SigmaSquared', 'CovRandomEffects' ]
    sigma2_output = [list(itertools.chain.from_iterable(sigma2[i])) for i in range(ndepvars)]
    vechD_output = [list(itertools.chain.from_iterable(vechD[i])) for i in range(ndepvars)]
    sigma2_output = [sigma2_output[i][0] for i in range(ndepvars)]
    vechD_output = [vechD_output[i][0] for i in range(ndepvars)]
    
    dict_list1 = get_stats_to_dict(param_estimates_keys, 
                                    sigma2_output,vechD_output)
    
    param_estimates_keys = ['Contrast Name', 'Contrast Vector', 'Beta','StdErrorBeta',
                            'Degrees of Freedom','T-Statistic','P-value']
    tcon_dict_list_fsregs = []
    for r in range(ndepvars):
        tcon_dict_list = []
        for c in range(len(tstats)):
            Lbeta_output = list(itertools.chain.from_iterable(tstats[c][2][0][r]))
            tcon_dict = get_stats_to_dict(param_estimates_keys, 
                                [tstats[c][0]],[tstats[c][1]],
                                Lbeta_output,[tstats[c][2][1][r]],
                                [tstats[c][2][2][r]],[tstats[c][2][3][r]],
                                [tstats[c][2][4][r]])
            tcon_dict_list.append(tcon_dict)
        tcon_dict_list_fsregs.append(tcon_dict_list)

    param_estimates_keys = ['Contrast Name', 'Contrast Vector','Degrees of Freedom',
                            'F-Statistic','P-value','R-Squared']
    fcon_dict_list_fsregs = []
    for r in range(ndepvars):
        fcon_dict_list = []
        for c in range(len(fstats)):
            fcon_dict = get_stats_to_dict(param_estimates_keys, 
                                [fstats[c][0]],[fstats[c][1]],
                                [fstats[c][2][0][r]],[fstats[c][2][1][r]],
                                [fstats[c][2][2][r]],[fstats[c][2][3][r]])
            fcon_dict_list.append(fcon_dict)
        fcon_dict_list_fsregs.append(fcon_dict_list)

    param_estimates_keys = ['Log-likelihood', 'ResidualMeanSquares', 'CovBeta',
                            'T-Contrasts', 'F-Contrasts']
    dict_list2 = get_stats_to_dict(param_estimates_keys,llh,resms,covB, 
                            tcon_dict_list_fsregs,fcon_dict_list_fsregs)

    param_estimates_keys = ['Parameter Estimates', 'Inference Statistics']
    dict_list = get_stats_to_dict(param_estimates_keys,dict_list1,dict_list2)

    return(dict_list)