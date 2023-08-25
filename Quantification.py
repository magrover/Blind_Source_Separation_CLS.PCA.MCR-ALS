# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 18:50:48 2023

@author: Steven
"""

import numpy as np
import pandas as pd
import sklearn
import scipy
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter



"""
Loading Files
"""

## Change this to the directory that you downloaded the associated CSV files and Blind_Source_Separation.py function
directory = 'C:/Users/scrouse6/Dropbox (GaTech)/Python/SRNL/Github/'

## Loads Blind Source Separation functions
sys.path.insert(1, directory)
from Blind_Source_Separation import *

X_train = pd.read_csv(directory + 'X_train.csv', header=None).values
X_run1 = pd.read_csv(directory + 'X_run1.csv', header=None).values
X_run2 = pd.read_csv(directory + 'X_run2.csv', header=None).values
X_ref = pd.read_csv(directory + 'X_ref.csv', header=None).values
wavenumber = pd.read_csv(directory + 'wavenumber.csv', header=None).values
Y_train = pd.read_csv(directory + 'Y_train.csv', header=None).values
Y_run1 = pd.read_csv(directory + 'Y_run1.csv', header=None).values
Y_run2 = pd.read_csv(directory + 'Y_run2.csv', header=None).values

"""
Blind Source Separation Preprocessing
"""

X_run1_preprocessed, Sources_run1 = BSS_removal(X_run1, X_ref[:3,:], X_ref[4:5,:], n_sources=5)
X_run2_preprocessed, Sources_run2 = BSS_removal_large_data(X_run2, X_ref[:3,:], X_ref[4:5,:], n_sources=6)

"""
Savitzky-Golay Filtering
"""

## Parameters for the Savitzky-Golay Filter
filter_points = 7
filter_order = 2
filter_deriv = 1

X_train_SV = savgol_filter(X_train.copy(), filter_points, filter_order, filter_deriv)
X_run1_SV = savgol_filter(X_run1.copy(), filter_points, filter_order, filter_deriv)
X_run2_SV = savgol_filter(X_run2.copy(), filter_points, filter_order, filter_deriv)
X_run1_preprocessed_SV = savgol_filter(X_run1_preprocessed.copy(), filter_points, filter_order, filter_deriv)
X_run2_preprocessed_SV = savgol_filter(X_run2_preprocessed.copy(), filter_points, filter_order, filter_deriv)

"""
Quantification
"""

model_plsr = PLSRegression(n_components = 4, scale = True)
model_plsr.fit(X_train_SV, Y_train)

y_hat_run1 = model_plsr.predict(X_run1_SV)
y_hat_run1_preprocessed = model_plsr.predict(X_run1_preprocessed_SV)
y_hat_run2 = model_plsr.predict(X_run2_SV)
y_hat_run2_preprocessed = model_plsr.predict(X_run2_preprocessed_SV)

y_hat_run1[y_hat_run1<0] = 0
y_hat_run1_preprocessed[y_hat_run1_preprocessed<0] = 0
y_hat_run2[y_hat_run2<0] = 0
y_hat_run2_preprocessed[y_hat_run2_preprocessed<0] = 0

"""
Plotting Parameters
"""

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})
Species = ['Nitrate','Nitrite','Glycolate','Glycolic Acid']
color=['orangered','royalblue','limegreen','goldenrod','darkviolet','slategray','chocolate','turquoise','dodgerblue','deeppink','seagreen']
marker=['o','^','s','*','D','X']

"""
Plotting Raw Data
"""

plt.figure(dpi = 300)
plt.plot(wavenumber, X_train.T)
plt.xlim(np.max(wavenumber), np.min(wavenumber))
plt.title('Training Data')
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Absorbance')

plt.figure(dpi = 300)
plt.plot(wavenumber, X_run1.T)
plt.xlim(np.max(wavenumber), np.min(wavenumber))
plt.title('Run 1 Data')
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Absorbance')

plt.figure(dpi = 300)
plt.plot(wavenumber, X_run2[::10,:].T)
plt.xlim(np.max(wavenumber), np.min(wavenumber))
plt.title('Run 2 Data')
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Absorbance')

plt.figure(dpi = 300)
plt.plot(wavenumber, X_ref.T)
plt.xlim(np.max(wavenumber), np.min(wavenumber))
plt.title('Reference Data')
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Absorbance')

"""
Plotting Run 1
"""

plt.figure(dpi=300)
plt.plot(wavenumber, X_run1.T, color=color[1])
plt.plot(wavenumber, X_run1_preprocessed.T, color=color[0])
plt.xlim(np.max(wavenumber), np.min(wavenumber))
plt.xlabel('Wavenumber (cm$^{-1})$')
plt.ylabel('Absorbance')
legend_elements1= [Line2D([0], [0], color=color[1], lw=2, label='Original'),
                    Line2D([0], [0], color=color[0], lw=2, label='Preprocessed')]
plt.legend(handles=legend_elements1, frameon=False,loc='upper right')

plt.figure(dpi=300)
ax = plt.axes()
x1=np.linspace(0,np.max(Y_run1),101) 
plt.plot(x1,x1,'k-')
plt.xlabel(r'Concentration (Ion Chromatography) ($\frac{mol}{L}$)')
plt.ylabel(r'Concentration (ATR-FTIR) ($\frac{mol}{L}$)')
for i in range(2):
    plt.scatter(Y_run1[:,i],y_hat_run1[:,i], color=color[1],marker=marker[i])
    plt.scatter(Y_run1[:,i],y_hat_run1_preprocessed[:,i], color=color[0],marker=marker[i])
legend_elements1= [Line2D([0], [0], color='w', marker=marker[0], markerfacecolor=color[1], markersize=9, label='Original Nitrate'),
                   Line2D([0], [0], color='w', marker=marker[1], markerfacecolor=color[1], markersize=10, label='Original Nitrite'),
                    Line2D([0], [0], color='w', marker=marker[0], markerfacecolor=color[0], markersize=9, label='Preprocessed Nitrate'),
                    Line2D([0], [0], color='w', marker=marker[1], markerfacecolor=color[0], markersize=10, label='Preprocessed Nitrite')]
plt.legend(handles=legend_elements1, frameon=False,loc='upper left', fontsize=14)

"""
Plotting Run 2
"""

plt.figure(dpi=300)
plt.plot(wavenumber, X_run2[::10, :].T, color=color[1])
plt.plot(wavenumber, X_run2_preprocessed[::10, :].T, color=color[0])
plt.xlim(np.max(wavenumber), np.min(wavenumber))
plt.xlabel('Wavenumber (cm$^{-1})$')
plt.ylabel('Absorbance')
legend_elements1= [Line2D([0], [0], color=color[1], lw=2, label='Original'),
                    Line2D([0], [0], color=color[0], lw=2, label='Preprocessed')]
plt.legend(handles=legend_elements1, frameon=False,loc='upper right')

timepoints = np.array([0,36,63])*60
plt.figure(dpi=300)
ax = plt.axes()
x1=np.linspace(0,np.max(Y_run2),101) 
plt.plot(x1,x1,'k-')
plt.xlabel(r'Concentration (Ion Chromatography) ($\frac{mol}{L}$)')
plt.ylabel(r'Concentration (ATR-FTIR) ($\frac{mol}{L}$)')
for i in range(2):
    plt.scatter(Y_run2[:,i],y_hat_run2[timepoints,i], color=color[1],marker=marker[i])
    plt.scatter(Y_run2[:,i],y_hat_run2_preprocessed[timepoints,i], color=color[0],marker=marker[i])
    
legend_elements1= [Line2D([0], [0], color='w', marker=marker[0], markerfacecolor=color[1], markersize=9, label='Original Nitrate'),
                   Line2D([0], [0], color='w', marker=marker[1], markerfacecolor=color[1], markersize=10, label='Original Nitrite'),
                    Line2D([0], [0], color='w', marker=marker[0], markerfacecolor=color[0], markersize=9, label='Preprocessed Nitrate'),
                    Line2D([0], [0], color='w', marker=marker[1], markerfacecolor=color[0], markersize=10, label='Preprocessed Nitrite')]
plt.legend(handles=legend_elements1, frameon=False,loc='upper left', fontsize=14)


