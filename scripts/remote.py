#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script includes the remote computations for linear mixed effects model
with decentralized statistic calculation
"""

import json
import sys
import numpy as np
import regression as reg
import npMatrix3d
import time
import pandas as pd
from scipy.linalg import block_diag
from lme_utils import *
from data_utils import *
import itertools

"""
============================================================================
The below function does the following tasks
1. read the number of levels of random factor and number of observations, for
each local site
2. calculate the total number of levels and total observations
3. send the values to local_1
----------------------------------------------------------------------------
This function takes in the following inputs in args['input']:
----------------------------------------------------------------------------
- nlevels : list containing number of levels for each random factor, as we
    have only one random factor, nlevels is a single value
- nobservns : number of observations per local site
- computation_phase : local_0
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    nlevels_persite : list containing number of levels of random factor for
    each site
    nlevels_global : total levels summed up for all local sites
    nlocalsites : number of local sites
    computation_phase : remote_0
- cache:
    nlevels_global : total levels summed up for all local sites
    nobservns_global : total observations summed up for all local sites
============================================================================
"""
def remote_0(args):
    input_list = args['input']

    nlevels_persite = [input_list[site]['nlevels'] for site in input_list]
    nlevels_global = np.sum(np.array(nlevels_persite))
    nobservns = [input_list[site]['nobservns'] for site in input_list]
    nobservns_global = np.sum(np.array(nobservns))

    computation_output_dict = {
        'cache':
        {
            'nlevels_global': nlevels_global.tolist(),
            'nobservns_global': nobservns_global.tolist(),
        },
        'output':
        {
            'nlevels_persite': nlevels_persite,
            'nlevels_global': nlevels_global.tolist(),
            'nlocalsites': len(input_list),
            'computation_phase': 'remote_0'
        }
    }

    return json.dumps(computation_output_dict)


"""
============================================================================
The below function does the following tasks
1. solves LME model using pseudo Simplified Fisher Scoring algorithm for
aggregate data to find global parameter estimates
2. calculate global inference parameters
----------------------------------------------------------------------------
This function takes in the following inputs in args:
----------------------------------------------------------------------------
- input :
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
    computation_phase : loacl_1
- cache:
    nlevels_global : total levels summed up for all local sites
    nobservns_global : total observations summed up for all local sites
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    computation_output_dict : dict containing regressions output which include
    estimates and inference parameters of LME for each freesurfer region.
    for detailed specification of regressions refer compspec 'output'
============================================================================
"""
def remote_1(args):
    state_list = args['state']
    input_list = args['input']
    cache_list = args['cache']

    input_dir = state_list["baseDirectory"]

    prod_matrices = ['XtransposeX_local','XtransposeY_local','XtransposeZ_local',
                    'YtransposeX_local','YtransposeY_local','YtransposeZ_local',
                    'ZtransposeX_local','ZtransposeY_local','ZtransposeZ_local']
    prod_matrices_vars=[]
    for p in prod_matrices:
        temp=0
        for site in input_list:
            temp=temp+np.array(loadBin(os.path.join(input_dir,
                                site,input_list[site][p])))

        prod_matrices_vars.append(temp)

    XtX = prod_matrices_vars[0]
    XtY = prod_matrices_vars[1]
    XtZ = prod_matrices_vars[2]
    YtX = prod_matrices_vars[3]
    YtY = prod_matrices_vars[4]
    YtZ = prod_matrices_vars[5]
    ZtX = prod_matrices_vars[6]
    ZtY = prod_matrices_vars[7]
    ZtZ = prod_matrices_vars[8]

    nlevels_global = np.array([cache_list['nlevels_global']])
    nobservns_global = cache_list['nobservns_global']
    tol = 1e-6
    nraneffs = np.array([1])

    # Run Pseudo Simplified Fisher Scoring
    paramVec = reg.pSFS3D(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ,
                            nlevels_global,nraneffs,tol,nobservns_global)

    nfixeffs = np.shape(XtX)[1]
    ndepvars = np.shape(XtY)[0]
    [beta,sigma2,vechD,D] = get_parameterestimates(paramVec,nfixeffs,ndepvars,
                                                                nlevels_global,nraneffs)

    prod_matrices = [XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ]
    contrasts = [input_list[site]['contrasts'] for site in input_list]
    con = contrasts[0]
    [llh,resms,covB,tstats,fstats] = reg.cal_inference(prod_matrices,nobservns_global,
                                                        nfixeffs,ndepvars,nlevels_global,
                                                        nraneffs,beta,sigma2,D,con)

    global_dict_list = gen_compoutputdict(beta,sigma2,vechD,llh,resms,covB,
                                                    tstats,fstats,ndepvars)

    paramVec_local = [input_list[site]['paramVec_local'] for site in input_list]
    paramVec_local = list(map(list, zip(*paramVec_local)))
    sites = [site for site in input_list]
    local_dict = [{key: value for key, value in zip(sites, stats_dict)}
                    for stats_dict in paramVec_local]

    keys = ["ROI", "global_stats", "local_stats"]
    fs_vars = input_list[sites[0]]['fs_vars']
    dict_list = get_stats_to_dict(keys,fs_vars,global_dict_list, local_dict)

    output_dict = {"regressions": dict_list}
    computation_output_dict = {"output": output_dict, "success": True}
    return json.dumps(computation_output_dict)

if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if 'local_0' in phase_key:
        computation_output = remote_0(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_1' in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Errors occurred')
