# Blind_Source_Separation_CLS.PCA.MCR-ALS
Blind Source Separation Functions combined with a PLSR quantification and plotting script using Savannah River National Laboratory Fourier transformed infrared (FTIR) spectroscopy data. The data is from 2-L scale runs of the SRAT and SME processes, designed to mimic conditions at the Defense Waste Processing Facility. SRAT: Sludge Receipt and Adjustment Tank. SME: Slurry Mix Evaporator.

•	Blind_Source_Separation.py: A Python file containing two functions used for performing blind source separation on nuclear waste slurries at SRNL. Inputs are: spectroscopic mixture data, spectroscopic references of target species, (optionally) spectroscopic references of known non-target species, and the total number of sources to use in the algorithm (targets + non-targets + unknown species). Outputs are the preprocessed spectra and calculated sources from MCR-ALS.

•	Quantification.py: A Python script that loads the associated SRNL FTIR data, performs blind source separation, quantifies the spectra, and plots the results.

•	wavenumber.csv: Wavenumbers associated with the utilized region of the Fourier-transformed infrared spectrum. Used for plotting spectra with physical x-axes.

•	X_ref.csv: One-molar references (in order) for water, nitrate, nitrite, low-pH (pH 2) glycolate reference, and high-pH (pH 13) glycolate reference.

•	X_run1.csv: Five FTIR spectra associated with a non-radioactive small-scale run of SRAT/SME processes at SRNL.

•	X_run2.csv: 3902 FTIR spectra associated with a (separate from Run 1) non-radioactive small-scale run of the SRAT/SME processes at SRNL.

•	X_train.csv: Eight FTIR spectra of training experiments with concentrations similar to those seen in SRAT/SME processing.

•	Y_run1.csv: Molar concentrations of nitrate and nitrite (as measured by ion chromatography) associated with each spectrum in X_run1.csv.

•	Y_run2.csv: Molar concentrations of nitrate and nitrite (as measured by ion chromatography) associated with three time points in X_run2.csv: hours 0, 36, and 63. 

•	Y_train.csv: Molar concentrations of nitrate and nitrite (as measured gravimetrically) associated with each spectrum in X_train.csv.


