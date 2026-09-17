"""
Bayesian Master Cosmology Fitter

- Fit cosmology using PyMC Bayesian-Hamiltonian MCMC algorithm
    
Test Groups:
- W1: Time lags in the mid-infrared W1-band (gamma correction found object-to-object)
- W2: Time lags in the mid-infrared W2-band (gamma correction found object-to-object)
- W1_118, W2_118: Assuming a gamma correction factor of 1.18 (Minezaki et al.)
- W1_28, W2_28: Assuming a gamma correction factor of 2.8 (Barvainis)
- W1_WISE, W2_WISE: No gamma correction applied 
- K: Time lags in the near-infrared K-band (MAGNUM data, gamma = 1.18 but is largely insignificant)  

"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde # Kernel density estimation for smoothing histograms
import pytensor.tensor as pt
import pymc as pm

def main():

    scatterInt = 0 #TODO Fit intrinsic scatter
    magnumTog = True # TODO Toggle whether MAGNUM data set is integrated with WISE fits
    omgTog = False  # TODO Fit omg_0: Toggle whether omg_0 is freely fitted
    H0Tog = True # TODO Toggle whether H0 is freely fitted

    flowCorrTog = False # TODO:Toggle use of flow corrected redshifts for MAGNUM objects

    fidH0 = 73 # Fiducial H0 assumed if fitting omg_0
    fidOmg_0 = 0.3 # Fiducial omg_0 assumed if fitting H0

    """
    Read Data (missing values: -999)

    """

    # WISE data
    objID, SDSSID, zReported, zHel, logL, logL_err, tauW1, tauW1_err, tauW1_Err, tauW2, tauW2_err, tauW2_Err =np.genfromtxt(
        "WISEData.csv", delimiter=",",usecols = (0,2,3,5,16,17,26,27,28,39,40,41), 
        skip_header=1, unpack = True, missing_values=("","#NUM!"), filling_values=-999, dtype=None, comments= None
    )

    # MAGNUM data
    zCMB_mag, tauK, tauK_err, tauK_Err, logLmag, logLmag_err, flux_mag, flux_mag_Err, flux_mag_err, obsBand = np.genfromtxt(
        "MAGNUMData.csv", delimiter=",", usecols = (2,5,6,7,10,13,14,15,16, 20), 
        skip_header=1, unpack = True, missing_values=(""), filling_values=-999, dtype=None
    )

    # WISE redshifts
    zCMB_WISE = np.genfromtxt(
        "WISERedshiftsCMB.txt", delimiter="|",unpack = True, dtype=float 
    )

    # Luminosity distances: 2D arrays with shape[0] = object, shape[1] = sample

    # W1 distances
    DL_tauW1 = np.genfromtxt(
        "LuminosityDistances_W1.txt", delimiter = "|", unpack = False, dtype=float # unpack = False -> put into 2D array
    )
    DL_tauW1_118 = np.genfromtxt(
        "LuminosityDistances_W1_118.txt", delimiter = "|", unpack = False, dtype=float 
    )
    DL_tauW1_28 = np.genfromtxt(
        "LuminosityDistances_W1_28.txt", delimiter = "|", unpack = False, dtype=float 
    )
    DL_tauW1_WISE = np.genfromtxt(
        "LuminosityDistances_W1_WISE.txt", delimiter = "|", unpack = False, dtype=float 
    )

    # W2 distances
    DL_tauW2 = np.genfromtxt(
        "LuminosityDistances_W2.txt", delimiter = "|", unpack = False, dtype=float # unpack = False -> put into 2D array
    )
    DL_tauW2_118 = np.genfromtxt(
        "LuminosityDistances_W2_118.txt", delimiter = "|", unpack = False, dtype=float 
    )
    DL_tauW2_28 = np.genfromtxt(
        "LuminosityDistances_W2_28.txt", delimiter = "|", unpack = False, dtype=float 
    )
    DL_tauW2_WISE = np.genfromtxt(
        "LuminosityDistances_W2_WISE.txt", delimiter = "|", unpack = False, dtype=float 
    )

    # K distances
    DL_tauK = np.genfromtxt(
        "LuminosityDistances_K.txt", delimiter = "|", unpack = False, dtype=float # unpack = False -> put into 2D array
    )

    """
    Data lists
    
    """

    # Cleaned luminosity distance lists (-999 removed, in linear Mpc)
    DL_tauW1_clean = []
    DL_tauW2_clean =[]
    DL_tauW1_118_clean = []
    DL_tauW2_118_clean = []
    DL_tauW1_28_clean = []
    DL_tauW2_28_clean = []
    DL_tauW1_WISE_clean = []
    DL_tauW2_WISE_clean = []
    DL_tauK_clean = []
    
    # Clean CMB redshifts (values without corresponding distance removed)
    zCMB_W1 = []
    zCMB_W2 = []
    zCMB_K = []

    """
    Get cleaned distance lists, get cleaned redshift lists

    """

    # W1 objects
    for i in range(len(DL_tauW1)):
        
        if DL_tauW1[i][0] != -999:
            DL_tauW1_clean.append(DL_tauW1[i])
            DL_tauW1_118_clean.append(DL_tauW1_118[i])
            DL_tauW1_28_clean.append(DL_tauW1_28[i])
            DL_tauW1_WISE_clean.append(DL_tauW1_WISE[i])

            zCMB_W1.append(zCMB_WISE[i])

    # W2 objects
    for i in range(len(DL_tauW2)):
        
        if DL_tauW2[i][0] != -999:
            DL_tauW2_clean.append(DL_tauW2[i])
            DL_tauW2_118_clean.append(DL_tauW2_118[i])
            DL_tauW2_28_clean.append(DL_tauW2_28[i])
            DL_tauW2_WISE_clean.append(DL_tauW2_WISE[i])

            zCMB_W2.append(zCMB_WISE[i])

    # K objects  
    for i in range(len(DL_tauK)):
        if DL_tauK[i][0] != -999:
            DL_tauK_clean.append(DL_tauK[i])
            zCMB_K.append(zCMB_mag[i])
        
    """
    Bayesian fit with PyMC
        
    """

    # Get the distributions of DL in logarithmic space
    logDL_tauK = np.log10(DL_tauK_clean)
    logDL_tauW1 = np.log10(DL_tauW1_clean)
    logDL_tauW1_118 = np.log10(DL_tauW1_118_clean)
    logDL_tauW1_28 = np.log10(DL_tauW1_28_clean)
    logDL_tauW1_WISE = np.log10(DL_tauW1_WISE_clean)
    logDL_tauW2 = np.log10(DL_tauW2_clean)
    logDL_tauW2_118 = np.log10(DL_tauW2_118_clean)
    logDL_tauW2_28 = np.log10(DL_tauW2_28_clean)
    logDL_tauW2_WISE = np.log10(DL_tauW2_WISE_clean)

    # Convert to smoothed porbability distributions using Kernel Denisty Estimation
    
    # Lists of KDE objects (distributions autonormalized by SciPy to become PDFs)
    kdeDL_tauK = []
    kdeDL_tauW1 = []
    kdeDL_tauW1_118 = []
    kdeDL_tauW1_28 = []
    kdeDL_tauW1_WISE = []
    kdeDL_tauW2 = []
    kdeDL_tauW2_118 = []
    kdeDL_tauW2_28 = []
    kdeDL_tauW2_WISE = []

    # K objects
    for i in range(len(logDL_tauK)):
        kde = gaussian_kde(
            logDL_tauK[i],
            bw_method="scott"
        )
        kdeDL_tauK.append(kde)
    
    # W1 objects
    for i in range(len(logDL_tauW1)):
        kde = gaussian_kde(
            logDL_tauW1[i],
            bw_method="scott"
        )
        kde118 = gaussian_kde(
            logDL_tauW1_118[i],
            bw_method="scott"
        )
        kde28 = gaussian_kde(
            logDL_tauW1_28[i],
            bw_method="scott"
        )
        kdeWISE = gaussian_kde(
            logDL_tauW1_WISE[i],
            bw_method="scott"
        )
        
        kdeDL_tauW1.append(kde)
        kdeDL_tauW1_118.append(kde118)
        kdeDL_tauW1_28.append(kde28)
        kdeDL_tauW1_WISE.append(kdeWISE)

    # W2 objects
    for i in range(len(logDL_tauW2)):
        kde = gaussian_kde(
            logDL_tauW2[i],
            bw_method="scott"
        )
        kde118 = gaussian_kde(
            logDL_tauW2_118[i],
            bw_method="scott"
        )
        kde28 = gaussian_kde(
            logDL_tauW2_28[i],
            bw_method="scott"
        )
        kdeWISE = gaussian_kde(
            logDL_tauW2_WISE[i],
            bw_method="scott"
        )
        
        kdeDL_tauW2.append(kde)
        kdeDL_tauW2_118.append(kde118)
        kdeDL_tauW2_28.append(kde28)
        kdeDL_tauW2_WISE.append(kdeWISE)

    # Create a shared grid upon which to evaluate KDE probablity densities
    kdeGrid = np.linspace(
        # Minimum and maximum distances from the H0 prior (in log10 space) with added padding (0.5)
        0.9670813842591461-0.5, 4.6055053736714875+0.5, 
        10000 # Grid contains 10000 cell resolution
    )

    # Evaluate the KDEs across the grid
    
    # Lists holding the values of the KDE PDF at each grid cell
    kdeDLVals_tauK = []
    kdeDLVals_tauW1 = []
    kdeDLVals_tauW1_118 = []
    kdeDLVals_tauW1_28 = []
    kdeDLVals_tauW1_WISE = []
    kdeDLVals_tauW2 = []
    kdeDLVals_tauW2_118 = []
    kdeDLVals_tauW2_28 = []
    kdeDLVals_tauW2_WISE = []

    # K objects
    for i in range(len(logDL_tauK)):
        kdeDLVals_tauK.append(kdeDL_tauK[i].evaluate(kdeGrid))

    # W1 objects
    for i in range(len(logDL_tauW1)):
        kdeDLVals_tauW1.append(kdeDL_tauW1[i].evaluate(kdeGrid))
        kdeDLVals_tauW1_118.append(kdeDL_tauW1_118[i].evaluate(kdeGrid))
        kdeDLVals_tauW1_28.append(kdeDL_tauW1_28[i].evaluate(kdeGrid))
        kdeDLVals_tauW1_WISE.append(kdeDL_tauW1_WISE[i].evaluate(kdeGrid))
    
    # W2 objects
    for i in range(len(logDL_tauW2)):
        kdeDLVals_tauW2.append(kdeDL_tauW2[i].evaluate(kdeGrid))
        kdeDLVals_tauW2_118.append(kdeDL_tauW2_118[i].evaluate(kdeGrid))
        kdeDLVals_tauW2_28.append(kdeDL_tauW2_28[i].evaluate(kdeGrid))
        kdeDLVals_tauW2_WISE.append(kdeDL_tauW2_WISE[i].evaluate(kdeGrid))

    # Define the fit/cosmological distance model (luminosity distance integral)
    def getModelD(z, H0, omg_0, zStepNum):
        z = np.asarray(z, dtype= float)
        zSteps = np.linspace(0,np.max(z), zStepNum) # Gives a grid of equally spaced z values for cumulative numeric integration
        
        denom = omg_0*(1+zSteps)**3 + (1-omg_0) # Denominator of the luminosity distance integrand (1-omg_0 = omg_DE)

        # Calculate a trapezoidal cumulative integral across the redshift grid, used to evaluate model distance at each redshift in z
        runningIntegral = cumulative_trapezoid(1/np.sqrt(denom),zSteps,initial = 0.0)
        
        # For each value in the given z, linearly interpolate along the cumulative integral grid and approximate the integral's value at the given z
        integrals = np.interp(z,zSteps,runningIntegral) # Gives array interpolated of integral evaluations at different z values
        
        return (1+z) * 299792.458/H0 *integrals # Calculate the final luminosity distance

    # For the PyMC inference procedure, convert all relevant arrays to tensors 
    kdeDLVals_tauK = pt.as_tensor(np.asarray(kdeDLVals_tauK))
    kdeDLVals_tauW1 = pt.as_tensor(np.asarray(kdeDLVals_tauW1))
    kdeDLVals_tauW1_118 = pt.as_tensor(np.asarray(kdeDLVals_tauW1_118))
    kdeDLVals_tauW1_28 =  pt.as_tensor(np.asarray(kdeDLVals_tauW1_28))
    kdeDLVals_tauW1_WISE = pt.as_tensor(np.asarray(kdeDLVals_tauW1_WISE))
    kdeDLVals_tauW2 = pt.as_tensor(np.asarray(kdeDLVals_tauW2))
    kdeDLVals_tauW2_118 = pt.as_tensor(np.asarray(kdeDLVals_tauW2_118))
    kdeDLVals_tauW2_28 =  pt.as_tensor(np.asarray(kdeDLVals_tauW2_28))
    kdeDLVals_tauW2_WISE = pt.as_tensor(np.asarray(kdeDLVals_tauW2_WISE))

    kdeGrid = pt.as_tensor(np.asarray(kdeGrid))

    # Interpolation function: Calculate the logarithmic probability of a set of model Ds corresponding to redshifts by interpolating the KDE grid for each data point
    def interpolateLogProb(D, *groups): # Takes args for the tensor of model Ds, and the relevant test group/data set
        
        # Linear interpolation: y = y1 + (x-x1)(y2-y1)/(x2-x1)
        # y is the y value you're looking for, y1 is the y value of a point below, y2 is the y value of a point above
        
        rows = pt.arange(D.shape[0]) # Create array corresponding to number of objects in D
        
        # Clipping: Prevent distance values that are out of the grid from causing a runtime error
        # If the sampled x lies outside the grid, it is clipped back into the grid 
        x1_ind = pt.clip(
            pt.searchsorted(kdeGrid, D) - 1, # The raw, unclipped x1 index/insertion index
            0, # Minimum allowed index
            kdeGrid.shape[0]-2 # The maximum allowed index (equivalent to last grid index (grid size - 1) minus 1 since the x2_ind must also be valid)
        )
        
        x2_ind = x1_ind + 1
        
        # For passing in only one data set/test group
        if len(groups) == 1:
            if groups[0] == "K": # For K lags
                kdeDLVals = kdeDLVals_tauK
            elif groups[0] == "W1": # For W1 lags
                kdeDLVals = kdeDLVals_tauW1
            elif groups[0] == "W1_118": # For W1_118 lags
                kdeDLVals = kdeDLVals_tauW1_118
            elif groups[0] == "W1_28": # For W1_28 lags
                kdeDLVals = kdeDLVals_tauW1_28
            elif groups[0] == "W2": # For W2 lags
                kdeDLVals = kdeDLVals_tauW2
            elif groups[0] == "W2_118": # For W2_118 lags
                kdeDLVals = kdeDLVals_tauW2_118
            elif groups[0] == "W2_28": # For W2_28 lags
                kdeDLVals = kdeDLVals_tauW2_28
        
        # For combining K and other data sets/test groups, pass in K as a second arg in *groups, pass in pt.concatenate((zCMB_W1, zCMB_K))
        else: 
            if groups[0] == "W1": # For W1 lags
                kdeDLVals = pt.concatenate((kdeDLVals_tauW1,kdeDLVals_tauK))
            elif groups[0] == "W1_118": # For W1_118 lags
                kdeDLVals = pt.concatenate((kdeDLVals_tauW1_118,kdeDLVals_tauK))
            elif groups[0] == "W1_28": # For W1_28 lags
                kdeDLVals = pt.concatenate((kdeDLVals_tauW1_28,kdeDLVals_tauK))
            elif groups[0] == "W2": # For W2 lags
                kdeDLVals = pt.concatenate((kdeDLVals_tauW2,kdeDLVals_tauK))
            elif groups[0] == "W2_118": # For W2_118 lags
                kdeDLVals = pt.concatenate((kdeDLVals_tauW2_118,kdeDLVals_tauK))
            elif groups[0] == "W2_28": # For W2_28 lags
                kdeDLVals = pt.concatenate((kdeDLVals_tauW2_28,kdeDLVals_tauK))
        
        # Linear interpolation equation
        return(
            pt.log(kdeDLVals[rows, x1_ind]) # y1
            + (D - (kdeGrid[x1_ind])) # (x-x1)
            * (pt.log(kdeDLVals[rows, x1_ind]) - pt.log(kdeDLVals[rows, x1_ind]) ) # (y2-y1)
            / (kdeGrid[x2_ind] - kdeGrid[x1_ind]) # (x2-x1)
        )
        
    # Create new PyMC Model object
    cosmoModel = pm.Model()

    with cosmoModel:
        H0 = pm.Uniform("H0", lower = 50.0, upper = 100.0) # Prior on H0 (uniform distribution)
        D = getModelD(pt.concatenate((zCMB_K,zCMB_W1)), H0, fidOmg_0, 1000) # Produces a set of model Ds corresponding to the redshift array (data) and the current parameter realization
        logP = interpolateLogProb(D, "W1","K") # Logarithmic probability of the current realization
        pm.Potential("D_likelihood",pt.sum(logP)) # Likelihood (additive sum in log space) of the current realization
        
        idata = pm.sample() # MCMC sampling
        
        # Print results
        H0_samples = idata.posterior["H0"].values.flatten() # Get all samples from all chains, flatten into 1D array
        print(f"Median H0: {np.median(H0_samples)}")
        print(f"Mean H0: {np.mean(H0_samples)}")
        print(f"Standard Deviation of H0: {np.std(H0_samples)}")
        print(f"15.9-84.1 Quantile Range of H0: {np.percentile(H0_samples,84.1) - np.percentile(H0_samples,15.9)}")

    # Plot the posterior distribution and then schematic of the MCMC chains
    az.plot_trace(idata, combined=True)
    plt.show()
    
if __name__ == "__main__":
    main()
    
#TODO: Bug fixes