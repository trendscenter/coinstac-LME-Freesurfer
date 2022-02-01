#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains functions to perform pseudo-Simplified Fisher Scoring for
the Mass Univariate Linear Mixed Model and other relevant functions including
calculation of the inference parameters
'''
import numpy as np
import warnings
from scipy import stats
import scipy
from npMatrix3d import *
import lme_utils

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import statsmodels.api as sm

'''
============================================================================
This below function performs pseudo-Simplified Fisher Scoring for the Mass
Univariate Linear Mixed Model. It is based on the update rules:

                      beta = (X'V^(-1)X)^(-1)(X'V^(-1)Y)

                            sigma2 = e'V^(-1)e/n

                             for k in {1,...,r};
             vec(D_k) = \theta_f + lam*I(vec(D_k))^+ (dl/dvec(D_k))

Where:
 - lam is a scalar stepsize.
 - I(vec(D_k)) is the Fisher Information matrix of vec(D_k).
 - dl/dvec(D_k) is the derivative of the log likelihood of vec(D_k) with
   respect to vec(D_k).
 - e is the residual vector (e=Y-X\beta)
 - V is the matrix (I+ZDZ')

Note that, as vf(D) is written in terms of 'vec', rather than 'vech',
(full  vector, 'f', rather than half-vector, 'h'), the information matrix
will have repeated rows (due to vf(D) having repeated entries). Because
of this, this method is based on the "pseudo-Inverse" (represented by the
+ above), hence the name.

The name "Simplified" here comes from a convention adopted in (Demidenko
2014).
----------------------------------------------------------------------------
This function takes as input;
----------------------------------------------------------------------------
 - XtX: X transpose multiplied by X (can be spatially varying or non
          -spatially varying).
 - XtY: X transpose multiplied by Y (spatially varying).
 - XtZ: X transpose multiplied by Z (can be spatially varying or non
          -spatially varying).
 - YtX: Y transpose multiplied by X (spatially varying).
 - YtY: Y transpose multiplied by Y (spatially varying).
 - YtZ: Y transpose multiplied by Z (spatially varying).
 - ZtX: Z transpose multiplied by X (can be spatially varying or non
          -spatially varying).
 - ZtY: Z transpose multiplied by Y (spatially varying).
 - ZtZ: Z transpose multiplied by Z (can be spatially varying or non
          -spatially varying).
- nlevels: A vector containing the number of levels for each factor,
             e.g. nlevels=[3,4] would mean the first factor has 3 levels
             and the second factor has 4 levels.
- nraneffs: A vector containing the number of random effects for each
             factor, e.g. nraneffs=[2,1] would mean the first factor has
             random effects and the second factor has 1 random effect.
 - tol: A scalar tolerance value. Iteration stops once successive
          log-likelihood values no longer exceed tol.
 - n: The number of observations (can be spatially varying or non
        -spatially varying).

 - reml: This a backdoor option for restricted maximum likelihood
           estimation. As BLMM is aimed at the high n setting it is
           unlikely this option will be useful and therefore isn't
           implemented everywhere or offered to users as an option.
----------------------------------------------------------------------------
And returns:
----------------------------------------------------------------------------
 - savedparams: \theta_h in the previous notation; the vector (beta,
                  sigma2, vech(D1),...vech(Dr)) for every voxel.
============================================================================
'''
def pSFS3D(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nlevels, nraneffs, tol, n, reml=False):

    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[1]

    # Number of voxels, v
    v = XtY.shape[0]

    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------

    # Inital beta
    beta = initBeta3D(XtX, XtY)

    # Work out e'e, X'e and Z'e
    Xte = XtY - (XtX @ beta)
    Zte = ZtY - (ZtX @ beta)
    ete = ssr3D(YtX, YtY, XtX, beta)

    # Initial sigma2
    sigma2 = initSigma23D(ete, n)

    # ------------------------------------------------------------------------------
    # Duplication matrices
    # ------------------------------------------------------------------------------
    dupMatTdict = dict()
    for i in np.arange(len(nraneffs)):

        dupMatTdict[i] = np.asarray(dupMat2D(nraneffs[i]).todense()).transpose()

    # ------------------------------------------------------------------------------
    # Inital D
    # ------------------------------------------------------------------------------
    # Dictionary version
    Ddict = dict()
    for k in np.arange(len(nraneffs)):

        Ddict[k] = makeDnnd3D(initDk3D(k, ZtZ, Zte, sigma2, nlevels, nraneffs, dupMatTdict))

    # Full version of D
    D = getDfromDict3D(Ddict, nraneffs, nlevels)

    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------
    # Work out the total number of paramateres
    tnp = np.int32(p + 1 + np.sum(nraneffs*(nraneffs+1)/2))

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)

    # ------------------------------------------------------------------------------
    # Obtain D(I+Z'ZD)^(-1)
    # ------------------------------------------------------------------------------
    DinvIplusZtZD =  forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

    # ------------------------------------------------------------------------------
    # Step size and log likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = np.ones(v)

    # Initial log likelihoods
    llhprev = -10*np.ones(XtY.shape[0])
    llhcurr = 10*np.ones(XtY.shape[0])

    # ------------------------------------------------------------------------------
    # Dicts to save repeated computation.
    # ------------------------------------------------------------------------------
    # This will hold the matrices: Sum_j^{l_k} Z_{i,j}'Z_{i,j}
    ZtZmatdict = dict()
    for k in np.arange(len(nraneffs)):
        ZtZmatdict[k] = None

    # This will hold the permutations needed for the covariance between the
    # derivatives with respect to k
    permdict = dict()
    for k in np.arange(len(nraneffs)):
        permdict[str(k)] = None

    # ------------------------------------------------------------------------------
    # Converged voxels and parameter saving
    # ------------------------------------------------------------------------------
    # Vector checking if all voxels converged
    converged_global = np.zeros(v)

    # Vector of saved parameters which have converged
    savedparams = np.zeros((v, np.int32(np.sum(nraneffs*(nraneffs+1)/2) + p + 1),1))

    # ------------------------------------------------------------------------------
    # Work out D indices (there is one block of D per level)
    # ------------------------------------------------------------------------------
    Dinds = np.zeros(np.sum(nlevels)+1)
    counter = 0

    # Loop through and add an index for each block of D.
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):
            Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
            counter = counter + 1

    # Last index will be missing so add it
    Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nraneffs[-1]

    # Make sure indices are ints
    Dinds = np.int64(Dinds)

    # ------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------
    nit=0
    while np.any(np.abs(llhprev-llhcurr)>tol):

        # Update number of iterations
        nit = nit + 1

        # --------------------------------------------------------------------------
        # Update loglikelihood and number of voxels
        # --------------------------------------------------------------------------
        # Change current likelihood to previous
        llhprev = llhcurr

        # Work out how many voxels are left
        v_iter = XtY.shape[0]

        # --------------------------------------------------------------------------
        # Update beta
        # --------------------------------------------------------------------------
        beta = np.linalg.solve(XtX - XtZ @ DinvIplusZtZD @ ZtX, XtY - XtZ @ DinvIplusZtZD @ ZtY)

        # Update sigma^2
        ete = ssr3D(YtX, YtY, XtX, beta)
        Zte = ZtY - (ZtX @ beta)

        # Make sure n is correct shape
        if hasattr(n, 'ndim'):
            if np.prod(n.shape) > 1:
                n = n.reshape(ete.shape)

        # --------------------------------------------------------------------------
        # REML update to sigma2
        # --------------------------------------------------------------------------
        if reml == False:
            sigma2 = (1/n*(ete - Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte)).reshape(v_iter)
        else:
            sigma2 = (1/(n-p)*(ete - Zte.transpose((0,2,1)) @ DinvIplusZtZD @ Zte)).reshape(v_iter)

        # --------------------------------------------------------------------------
        # Update D
        # --------------------------------------------------------------------------
        counter = 0
        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Calculate derivative with respect to D_k
            #-----------------------------------------------------------------------
            # Work out derivative
            if ZtZmatdict[k] is None:
                dldDk,ZtZmatdict[k] = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=None)
            else:
                dldDk,_ = get_dldDk3D(k, nlevels, nraneffs, ZtZ, Zte, sigma2, DinvIplusZtZD,ZtZmat=ZtZmatdict[k])

            #-----------------------------------------------------------------------
            # Calculate covariance of derivative with respect to D_k
            #-----------------------------------------------------------------------
            if permdict[str(k)] is None:
                covdldDk1dDk2,permdict[str(k)] = get_covdldDk1Dk23D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, perm=None)
            else:
                covdldDk1dDk2,_ = get_covdldDk1Dk23D(k, k, nlevels, nraneffs, ZtZ, DinvIplusZtZD, dupMatTdict, vec=True, perm=permdict[str(k)])

            #-----------------------------------------------------------------------
            # Work out update amount
            #-----------------------------------------------------------------------
            update_p = np.linalg.solve(forceSym3D(covdldDk1dDk2), mat2vec3D(dldDk))

            # Multiply by stepsize
            update_p = np.einsum('i,ijk->ijk',lam, update_p)

            # Update D_k
            Ddict[k] = makeDnnd3D(vec2mat3D(mat2vec3D(Ddict[k]) + update_p))

            # Add D_k back into D and recompute DinvIplusZtZD
            for j in np.arange(nlevels[k]):

                D[:, Dinds[counter]:Dinds[counter+1], Dinds[counter]:Dinds[counter+1]] = Ddict[k]
                counter = counter + 1

        # --------------------------------------------------------------------------
        # Obtain D(I+Z'ZD)^(-1)
        # --------------------------------------------------------------------------
        DinvIplusZtZD = forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

        # --------------------------------------------------------------------------
        # Recalculate matrices
        # --------------------------------------------------------------------------
        ete = ssr3D(YtX, YtY, XtX, beta)
        Zte = ZtY - (ZtX @ beta)

        # Check sigma2 hasn't hit a boundary
        sigma2[sigma2<0]=1e-10

        # --------------------------------------------------------------------------
        # Update the step size and log likelihood
        # --------------------------------------------------------------------------
        llhcurr = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D,reml, XtX, XtZ, ZtX)
        lam[llhprev>llhcurr] = lam[llhprev>llhcurr]/2

        # --------------------------------------------------------------------------
        # Work out which voxels converged
        # --------------------------------------------------------------------------
        # Obatin indices of converged voxels
        indices_ConAfterIt, indices_notConAfterIt, indices_ConDuringIt, localconverged, localnotconverged = getConvergedIndices(converged_global, (np.abs(llhprev-llhcurr)<tol))

        # Record which voxels converged.
        converged_global[indices_ConDuringIt] = 1

        # --------------------------------------------------------------------------
        # Save parameters from this run
        # --------------------------------------------------------------------------
        savedparams[indices_ConDuringIt,0:p,:]=beta[localconverged,:,:]
        savedparams[indices_ConDuringIt,p:(p+1),:]=sigma2[localconverged].reshape(sigma2[localconverged].shape[0],1,1)

        for k in np.arange(len(nraneffs)):

            # Get vech form of D_k
            vech_Dk = mat2vech3D(Ddict[k][localconverged,:,:])

            # Make sure it has correct shape (i.e. shape (num voxels converged, num
            # random effects for factor k squared, 1))
            vech_Dk = vech_Dk.reshape(len(localconverged),nraneffs[k]*(nraneffs[k]+1)//2,1)
            savedparams[indices_ConDuringIt,FishIndsDk[k]:FishIndsDk[k+1],:]=vech_Dk

        # --------------------------------------------------------------------------
        # Update matrices
        # --------------------------------------------------------------------------
        XtY = XtY[localnotconverged, :, :]
        YtX = YtX[localnotconverged, :, :]
        YtY = YtY[localnotconverged, :, :]
        ZtY = ZtY[localnotconverged, :, :]
        YtZ = YtZ[localnotconverged, :, :]
        ete = ete[localnotconverged, :, :]

        # Spatially varying design
        if XtX.shape[0] > 1:

            XtX = XtX[localnotconverged, :, :]
            ZtX = ZtX[localnotconverged, :, :]
            ZtZ = ZtZ[localnotconverged, :, :]
            XtZ = XtZ[localnotconverged, :, :]

            # ----------------------------------------------------------------------
            # Update ZtZmat
            # ----------------------------------------------------------------------
            # ZtZmat
            for k in np.arange(len(nraneffs)):
                if ZtZmatdict[k] is not None:
                    ZtZmatdict[k]=ZtZmatdict[k][localnotconverged, :, :]

        # Update n
        if hasattr(n, 'ndim'):
            # Check if n varies with voxel
            if n.shape[0] > 1:
                if n.ndim == 1:
                    n = n[localnotconverged]
                if n.ndim == 2:
                    n = n[localnotconverged,:]
                if n.ndim == 3:
                    n = n[localnotconverged,:,:]

        DinvIplusZtZD = DinvIplusZtZD[localnotconverged, :, :]

        # --------------------------------------------------------------------------
        # Update step size and log likelihoods
        # --------------------------------------------------------------------------
        lam = lam[localnotconverged]
        llhprev = llhprev[localnotconverged]
        llhcurr = llhcurr[localnotconverged]

        # --------------------------------------------------------------------------
        # Update parameters
        # --------------------------------------------------------------------------
        beta = beta[localnotconverged, :, :]

        sigma2 = sigma2[localnotconverged]

        D = D[localnotconverged, :, :]

        for k in np.arange(len(nraneffs)):
            Ddict[k] = Ddict[k][localnotconverged, :, :]

        # --------------------------------------------------------------------------
        # Matrices needed later by many calculations
        # ----------------------------------------------------------------------------
        # X transpose e and Z transpose e
        Xte = XtY - (XtX @ beta)
        Zte = ZtY - (ZtX @ beta)

    return(savedparams)

'''
============================================================================
This below function takes in parameter estimates and product matrices and
calculates the inference parameters including the stats for the contrasts
specified.
Inference Parameters are,
- loglikelihood
- Residual mean square
- covariance of Beta estimates
- T-stats
- F-stats
T-stats include,
- LBeta : contrasts multiplied by estimate of beta (this is same as COPE in FSL)
- seLB: standard error of the contrasts multiplied by beta (only available for
        T contrasts)
- swdfc: Sattherthwaithe degrees of freedom estimates for the T contrasts
- T-stats
- -log10 of the uncorrected P values for the T-contrasts
F-stats include,
- swdfc: Sattherthwaithe degrees of freedom estimates for the F-contrasts
- F-stat
- -log10 of the uncorrected P values for the F-contrasts
- R2: partial R^2 maps for the contrasts (only available for F contrasts)

----------------------------------------------------------------------------
This function takes as input;
----------------------------------------------------------------------------
- prod_matrices: [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
- nlevels: A vector containing the number of levels for each factor,
             e.g. nlevels=[3,4] would mean the first factor has 3 levels
             and the second factor has 4 levels.
- nraneffs: A vector containing the number of random effects for each
             factor, e.g. nraneffs=[2,1] would mean the first factor has
             random effects and the second factor has 1 random effect.
- n: The number of observations (can be spatially varying or non
        -spatially varying).
- p: Number of Fixed Effects parameters in the design
- v: number of freesurfer variables
- beta: fixed effects parameter estimates
- sigma2: fixed effects variance estimates
- D: random effects variance estimates (covariance matrix in full)
- contrasts: contrasts vectors to be tested for T-stats and F-stats

----------------------------------------------------------------------------
And returns:
----------------------------------------------------------------------------
 - llh: log likelihood
 - resms: residual mean squares
 - covB: Beta covariance
 - tstats: [Lbeta,seLB,swdfc,Tc,pc]
 - fstats: [swdfc,Fc,pc,R2]
============================================================================
'''
def cal_inference(prod_matrices,n,p,v,nlevels,nraneffs,beta,sigma2,D,contrasts):

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    llh = cal_loglikelihood(prod_matrices,n,q,nlevels,nraneffs,beta,sigma2,D)

    resms = cal_resms(prod_matrices,n,p,beta).reshape(v)

    covB = cal_covB(prod_matrices,q,sigma2,D).reshape(v, p**2)

    con_names = [contrasts[i]['name']['value'] for i in range(len(contrasts))]
    L=[contrasts[i]['vector']['value'] for i in range(len(contrasts))]

    tstats=[]
    fstats=[]
    for i in range(len(L)):
        con = L[i]
        if(isinstance(con[0],(list))):
            statType = 'F'
            con = np.array(con)
            fstats.append([con_names[i],L[i],cal_fstat(prod_matrices,n,p,q,v,nlevels,nraneffs,beta,sigma2,D,np.array(con))])
        else:
            statType = 'T'
            con = np.array(con)
            con = con.reshape([1,con.shape[0]])

            tstats.append([con_names[i],L[i],cal_tstat(prod_matrices,n,p,q,v,nlevels,nraneffs,beta,sigma2,D,con)])

    return([llh.tolist(),resms.tolist(),covB.tolist(),tstats,fstats])

'''
============================================================================
This below function takes in parameter estimates and product matrices and
calculates the log likelihood.
----------------------------------------------------------------------------
This function takes as input;
----------------------------------------------------------------------------
- prod_matrices: [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
- nlevels: A vector containing the number of levels for each factor,
             e.g. nlevels=[3,4] would mean the first factor has 3 levels
             and the second factor has 4 levels.
- nraneffs: A vector containing the number of random effects for each
             factor, e.g. nraneffs=[2,1] would mean the first factor has
             random effects and the second factor has 1 random effect.
- n: The number of observations (can be spatially varying or non
        -spatially varying).
- q: Total number of Random Effects (duplicates included), i.e. the second
    dimension of, Z, the random effects design matrix.
- beta: fixed effects parameter estimates
- sigma2: fixed effects variance estimates
- D: random effects variance estimates (covariance matrix in full)
----------------------------------------------------------------------------
And returns:
----------------------------------------------------------------------------
 - llh: log likelihood
============================================================================
'''
def cal_loglikelihood(prod_matrices,n,q,nlevels,nraneffs,beta,sigma2,D):

    [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ] = prod_matrices

    # Miscellaneous matrix variables
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    Zte = ZtY - (ZtX @ beta)
    ete = ssr3D(YtX, YtY, XtX, beta)

    REML=False

    # Output log likelihood
    llh = llh3D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD, D, REML, XtX, XtZ, ZtX) - (0.5*(n)*np.log(2*np.pi))

    return(llh)

'''
============================================================================
This below function takes in parameter estimates and product matrices and
calculates the residual mean squares.
----------------------------------------------------------------------------
This function takes as input;
----------------------------------------------------------------------------
- prod_matrices: [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
- n: The number of observations (can be spatially varying or non
        -spatially varying).
- p: Number of Fixed Effects parameters in the design
- beta: fixed effects parameter estimates
----------------------------------------------------------------------------
And returns:
----------------------------------------------------------------------------
- resms: residual mean squares
============================================================================
'''
def cal_resms(prod_matrices,n,p,beta):

    [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ] = prod_matrices

    resms = get_resms3D(YtX, YtY, XtX, beta,n,p)

    return(resms)

'''
============================================================================
This below function takes in parameter estimates and product matrices and
calculates the Beta covariance.
----------------------------------------------------------------------------
This function takes as input;
----------------------------------------------------------------------------
- prod_matrices: [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
- q: Total number of Random Effects (duplicates included), i.e. the second
    dimension of, Z, the random effects design matrix
- sigma2: fixed effects variance estimates
- D: random effects variance estimates (covariance matrix in full)
----------------------------------------------------------------------------
And returns:
----------------------------------------------------------------------------
- covB: beta covariance
============================================================================
'''
def cal_covB(prod_matrices,q,sigma2,D):

    [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ] = prod_matrices

    # Miscellaneous matrix variables
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    covB = get_covB3D(XtX, XtZ, DinvIplusZtZD, sigma2)

    return(covB)

'''
============================================================================
This below function takes in parameter estimates and product matrices and
calculates the tstats for the contrasts specified.
T-stats include,
- contrasts multiplied by estimate of beta (this is same as COPE in FSL)
- standard error of the contrasts multiplied by beta (only available for
T contrasts)
- Sattherthwaithe degrees of freedom estimates for the T contrasts
- T-stat
- -log10 of the uncorrected P values for the T-contrasts
----------------------------------------------------------------------------
This function takes as input;
----------------------------------------------------------------------------
- prod_matrices: [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
- nlevels: A vector containing the number of levels for each factor,
             e.g. nlevels=[3,4] would mean the first factor has 3 levels
             and the second factor has 4 levels.
- nraneffs: A vector containing the number of random effects for each
             factor, e.g. nraneffs=[2,1] would mean the first factor has
             random effects and the second factor has 1 random effect.
- n: The number of observations (can be spatially varying or non
        -spatially varying).
- p: Number of Fixed Effects parameters in the design
- v: number of freesurfer variables
- beta: fixed effects parameter estimates
- sigma2: fixed effects variance estimates
- D: random effects variance estimates (covariance matrix in full)
- L: contrasts vectors to be tested for T-stats and F-stats
- q: Total number of Random Effects (duplicates included), i.e. the second
    dimension of, Z, the random effects design matrix
----------------------------------------------------------------------------
And returns:
----------------------------------------------------------------------------
- LBeta : contrasts multiplied by estimate of beta (this is same as COPE in FSL)
- seLB: standard error of the contrasts multiplied by beta (only available for
        T contrasts)
- swdfc: Sattherthwaithe degrees of freedom estimates for the T contrasts
- Tc: T statistic for the contrasts
- pc: -log10 of the uncorrected P values for the T-contrasts
============================================================================
'''
def cal_tstat(prod_matrices,n,p,q,v,nlevels,nraneffs,beta,sigma2,D,L):

    minlog=-323.3062153431158

    [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ] = prod_matrices

    # Miscellaneous matrix variables
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    Lbeta = L @ beta

    seLB = np.sqrt(get_varLB3D(L, XtX, XtZ, DinvIplusZtZD, sigma2).reshape(v))

    # Calculate sattherwaite estimate of the degrees of freedom of this statistic
    swdfc = get_swdf_T3D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nraneffs).reshape(v)

    # Obtain and output T statistic
    Tc = get_T3D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2).reshape(v)

    # Obatin and output p-values
    pc = T2P3D(Tc,swdfc,minlog)

    Lbeta = Lbeta.tolist()
    seLB = seLB.tolist()
    swdfc = swdfc.tolist()
    Tc = Tc.tolist()
    pc = pc.tolist()

    return(Lbeta,seLB,swdfc,Tc,pc)

'''
============================================================================
This below function takes in parameter estimates and product matrices and
calculates the f-stats for the contrasts specified.
F-stats include,
- swdfc: Sattherthwaithe degrees of freedom estimates for the F-contrasts
- F statistic for the contrasts
- -log10 of the uncorrected P values for the F-contrasts
- R2: partial R^2 maps for the contrasts (only available for F contrasts)
----------------------------------------------------------------------------
This function takes as input;
----------------------------------------------------------------------------
- prod_matrices: [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
- nlevels: A vector containing the number of levels for each factor,
             e.g. nlevels=[3,4] would mean the first factor has 3 levels
             and the second factor has 4 levels.
- nraneffs: A vector containing the number of random effects for each
             factor, e.g. nraneffs=[2,1] would mean the first factor has
             random effects and the second factor has 1 random effect.
- n: The number of observations (can be spatially varying or non
        -spatially varying).
- p: Number of Fixed Effects parameters in the design
- v: number of freesurfer variables
- beta: fixed effects parameter estimates
- sigma2: fixed effects variance estimates
- D: random effects variance estimates (covariance matrix in full)
- L: contrasts vectors to be tested for T-stats and F-stats
----------------------------------------------------------------------------
And returns:
----------------------------------------------------------------------------
 - swdfc: Sattherthwaithe degrees of freedom estimates for the F-contrasts
- Fc: F statistic for the contrasts
- pc: -log10 of the uncorrected P values for the F-contrasts
- R2: partial R^2 maps for the contrasts (only available for F contrasts)
============================================================================
'''
def cal_fstat(prod_matrices,n,p,q,v,nlevels,nraneffs,beta,sigma2,D,L):

    minlog=-323.3062153431158

    [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ] = prod_matrices

    # Miscellaneous matrix variables
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # Calculate sattherthwaite degrees of freedom for the inner.
    swdfc = get_swdf_F3D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nraneffs).reshape(v)

    # Calculate F statistic.
    Fc=get_F3D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2).reshape(v)

    # Work out p for this contrast
    pc = F2P3D(Fc, L, swdfc, minlog).reshape(v)

    # Calculate partial R2 masked for ring.
    R2 = get_R23D(L, Fc, swdfc).reshape(v)

    swdfc = swdfc.tolist()
    Fc = Fc.tolist()
    pc = pc.tolist()
    R2 = R2.tolist()

    return(swdfc,Fc,pc,R2)
