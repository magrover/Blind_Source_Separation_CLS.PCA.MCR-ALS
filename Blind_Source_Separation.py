# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:04:28 2023

@author: Steven
"""
import numpy as np
import sklearn
from sklearn.decomposition import FastICA, PCA 
from sklearn.linear_model import ElasticNet

## MCR Algorithm from NIST: https://pages.nist.gov/pyMCR/
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg, ConstraintNorm

## Blind source separation procedure
def BSS_removal(X_test, X_ref_targets, X_ref_nontargets=None, n_sources=5):
    X = X_test
    
    ## Allows for optional input for non-target references (X_ref_nontargets)
    if X_ref_nontargets is not None:
        X_ref = np.vstack((X_ref_targets, X_ref_nontargets))
        n_nontargets = np.size(X_ref_nontargets, axis=0)
    else:
        X_ref = X_ref_targets.copy()
        n_nontargets=0
    
    n_components=np.size(X_ref_targets, axis=0)
    n_removed = n_sources - n_components
    
    ## A least squares fit is done with available references and subtracted from the mixture spectra (X_test)
    Y_pca = X_test@X_ref.T@np.linalg.inv(X_ref@X_ref.T)
    E = X_test - Y_pca@X_ref
    
    ## PCA is performed on residuals after subtraction from the mixture spectra
    pca = sklearn.decomposition.PCA(n_components=n_sources-n_components-n_nontargets)
    pca.fit(E)
    P_pca = pca.components_
    mcr_init = np.vstack((X_ref, P_pca**2))

    ## Initialize MCR-ALS
    mcrals = McrAR(st_regr='NNLS',c_regr=ElasticNet(alpha=1e-4,l1_ratio=0.25),tol_increase=5,tol_n_above_min=500,max_iter=2000,tol_err_change=X.mean()*1e-8,c_constraints=[ConstraintNonneg()])
    
    ## Switch to alpha=1e-5 for adding another species or adding/removing both glycolate sources
    # mcrals = McrAR(st_regr='NNLS',c_regr=ElasticNet(alpha=1e-5,l1_ratio=0.25),tol_increase=5,tol_n_above_min=500,max_iter=2000,tol_err_change=X.mean()*1e-8,c_constraints=[ConstraintNonneg()])
    
    ## Fit MCR-ALS with initial conditions of known sources and predicted sources from PCA; constrain known species to match reference spectra (st_fix=[0,1,2])
    mcrals.fit(X, ST = mcr_init, st_fix=[0,1,2])
    
    ## Decomposed mixture spectra
    S0mcr = mcrals.ST_opt_;
    Amcr  = mcrals.C_opt_; 
    Sources = S0mcr.copy()
    
    ## Non-target and newly identified sources are denoted for subtraction
    sources_removed = np.arange((n_components), n_sources)
    
    ## Non-target and newly identified sources are subtracted from mixture spectra using the calculated bilinear mixture model
    Xdeflation = Amcr[:,sources_removed]@S0mcr[sources_removed,:]
    X_preprocessed = X[:,:]-Xdeflation[:,:]

    return X_preprocessed, Sources



## Regularization Parameter (alpha) has been decreased slightly, leading to improved performance on larger datasets in our testing.
def BSS_removal_large_data(X_test, X_ref_targets, X_ref_nontargets=None, n_sources=6):
    X = X_test
    
    ## Allows for optional input for non-target references (X_ref_nontargets)
    if X_ref_nontargets is not None:
        X_ref = np.vstack((X_ref_targets, X_ref_nontargets))
        n_nontargets = np.size(X_ref_nontargets, axis=0)
    else:
        X_ref = X_ref_targets.copy()
        n_nontargets=0
        
    n_components=np.size(X_ref_targets, axis=0)
    n_removed = n_sources - n_components
    
    ## A least squares fit is done with available references and subtracted from the mixture spectra (X_test)
    Y_pca = X_test@X_ref.T@np.linalg.inv(X_ref@X_ref.T)
    E = X_test - Y_pca@X_ref
    
    ## PCA is performed on residuals after subtraction from the mixture spectra
    pca = sklearn.decomposition.PCA(n_components=n_sources-n_components-n_nontargets)
    pca.fit(E)
    P_pca = pca.components_
    mcr_init = np.vstack((X_ref, P_pca**2))
    
    ## Initialize MCR-ALS
    mcrals = McrAR(st_regr='NNLS',c_regr=ElasticNet(alpha=1e-6,l1_ratio=0.75),tol_increase=5,tol_n_above_min=500,max_iter=2000,tol_err_change=X.mean()*1e-8,c_constraints=[ConstraintNonneg()])
    
    ## Fit MCR-ALS with initial conditions of known sources and predicted sources from PCA; constrain known species to match reference spectra (st_fix=[0,1,2])
    mcrals.fit(X, ST = mcr_init, st_fix=[0,1,2])
    
    ## Decomposed mixture spectra
    S0mcr = mcrals.ST_opt_;
    Amcr  = mcrals.C_opt_; 
    Sources = S0mcr.copy()

    ## Non-target and newly identified sources are denoted for subtraction
    sources_removed = np.arange((n_components), (n_sources))
    
    ## Non-target and newly identified sources are subtracted from mixture spectra using the calculated bilinear mixture model
    Xdeflation = Amcr[:,sources_removed]@S0mcr[sources_removed,:]
    X_preprocessed = X[:,:]-Xdeflation[:,:]
    return X_preprocessed , Sources