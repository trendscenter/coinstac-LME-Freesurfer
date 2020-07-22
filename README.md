## Decentralized LME for freesurfer data
This repository contains decentralized regression using linear mixed effects model for freesurfer data.

This is an initial version of decentralized LME implementation with the below assumptions,
1. only site covariates are considered for random factors, the number of levels correspond to the number of sub-sites in each local site
2. one random effect for site covariate is considered, hence the random covariates input values contain all 1s

LME model is implemeted using Pseudo-simplified Fisher Scoring algorithm (PSFS). PSFS implementation and calculation of inference parameters are taken from the original implementation by Tom Maullin [1]. Details about PSFS algorithm can be seen in BLMM notes [2] included in this repo.

## Input
Input parameters include the following to be defined for each local site,
1. fixed_covariates : csv file listing out the fixed covariates
2. random_covariates : csv file listing out the random effects for each ramdom factor
3. random_factor : csv fie listing the sub-site numbers
4. dependents : csv file listing out the freesurfer stats file
5. freesurfer_variables : list of freesurfer variables/regions to be included for response variables
6. contrasts : list of contrasts, each list item contains contrast name and vector

## Steps to run:

1. sudo npm i -g coinstac-simulator@4.2.0
2. git clone https://github.com/trendscenter/coinstac-LME-Freesurfer.git
3. cd coinstac-lme-freesurfer
4. docker build -t dlme_fs
5. coinstac-simulator

## References:

> [1]. https://github.com/TomMaullin/BLMM

> [2]. BLMM notes by Tom Maullin
