"""
MPFIT Master Cosmology Fitter

- Fit cosmology using the MPFIT least squares minimizer

- Uses a hybrid bootstrapping-MC technique (inspired by Flux Randomization / Random Subset Selection in reverberation mapping)
- Draws MC samples from data point distributions and bootstraps data to obtain and uncertainty distribution of fit parameters
    
Test Groups:
- W1: Time lags in the mid-infrared W1-band (gamma correction found object-to-object)
- W2: Time lags in the mid-infrared W2-band (gamma correction found object-to-object)
- W1_118, W2_118: Assuming a gamma correction factor of 1.18 (Minezaki et al.)
- W1_28, W2_28: Assuming a gamma correction factor of 2.8 (Barvainis)
- W1_WISE, W2_WISE: No gamma correction applied 
- K: Time lags in the near-infrared K-band (MAGNUM data, gamma = 1.18 but is largely insignificant)  

"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import mpfit

n = 10000 # Number of bootstrap iterations
scatter_int = 0.2 # Assign chosen intrinsic scatter (dex), effects data weighing TODO: Obtain from Bayesian fit
errTog = "IQ" # Toggle convention for error in data points: IQ = interquantile range, SD = standard deviation (IQ removes huge errors from outlier samples in distribution)

mcTog = True # Toggle whether MC-boostrap is used (otherwise performs standard, single iteration fit)
magnumTog = True # Toggle whether MAGNUM data set is integrated with WISE fits
omgTog = False  # Toggle whether omg_0 is freely fitted
H0Tog = True # Toggle whether H0 is freely fitted

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
    "WISE_RedshiftsCMB.txt", delimiter="|",unpack = True, dtype=float 
)

# Luminosity distances: 2D arrays with shape[0] = object, shape[1] = sample

# W1 distances
DL_tauW1 = np.genfromtxt(
    "LuminosityDistances_W1.txt", delimiter = "|", unpack = True, dtype=float
)
DL_tauW1_118 = np.genfromtxt(
    "LuminosityDistances_W1_118.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW1_28 = np.genfromtxt(
    "LuminosityDistances_W1_28.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW1_WISE = np.genfromtxt(
    "LuminosityDistances_W1_WISE.txt", delimiter = "|", unpack = True, dtype=float 
)

# W2 distances
DL_tauW2 = np.genfromtxt(
    "LuminosityDistances_W2.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW2_118 = np.genfromtxt(
    "LuminosityDistances_W2_118.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW2_28 = np.genfromtxt(
    "LuminosityDistances_W2_28.txt", delimiter = "|", unpack = True, dtype=float 
)
DL_tauW2_WISE = np.genfromtxt(
    "LuminosityDistances_W2_WISE.txt", delimiter = "|", unpack = True, dtype=float 
)

# K distances
DL_tauK = np.genfromtxt(
    "LuminosityDistances_K.txt", delimiter = "|", unpack = True, dtype=float 
)

"""
Cosmology Parameters and Data Lists

- Lists for the values of the best fit parameters for every realization, for each test group

"""

# Hubble Constant
H0_W1 = []
H0_W1_118 = []
H0_W1_28 = []
H0_W2 = []
H0_W2_118 = []
H0_W2_28 = []
H0_W1_WISE = []
H0_W2_WISE = []

# Dimensionless mass density
omg_0_W1 = []
omg_0_W1_118 = []
omg_0_W1_28 = []
omg_0_W2 = []
omg_0_W2_118 = []
omg_0_W2_28 = []
omg_0_W1_WISE = []
omg_0_W2_WISE = []

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

# Errors in luminosity distances (clean, -999 removed, in log10 scales)
DL_tauW1_err = []
DL_tauW1_118_err = []
DL_tauW1_28_err = []
DL_tauW2_err = []
DL_tauW2_118_err = []
DL_tauW2_28_err = []
DL_tauW1_WISE_err = []
DL_tauW2_WISE_err = []
DL_tauK_err = []

# Clean CMB redshifts (values without corresponding distance removed)
zCMB_W1 = []
zCMB_W2 = []
zCMB_K = []

"""
Get cleaned distance lists and errors, get cleaned redshift lists

"""

# W1 objects
for i in range(len(DL_tauW1)):
    
    if DL_tauW1[i][0] != -999:
        DL_tauW1_clean.append(DL_tauW1[i])
        DL_tauW1_118_clean.append(DL_tauW1_118[i])
        DL_tauW1_28_clean.append(DL_tauW1_28[i])
        DL_tauW1_WISE_clean.append(DL_tauW1_WISE[i])
        
        if errTog == "SD": # Get standard deviation in log space (dex)
            DL_tauW1_err.append(np.std(np.log10(DL_tauW1[i]))) 
            DL_tauW1_118_err.append(np.std(np.log10(DL_tauW1_118[i])))
            DL_tauW1_28_err.append(np.std(np.log10(DL_tauW1_28[i])))
            DL_tauW1_WISE_err.append(np.std(np.log10(DL_tauW1_WISE[i])))
            
        elif errTog == "IQ": # Get 15.9-84.1% interquantile range in log space (dex): Improves error stability, less sensitive to massive sample outliers
            DL_tauW1_err.append((np.percentile(np.log10(DL_tauW1[i]), 84.1)-np.percentile(np.log10(DL_tauW1[i]), 15.9))/2)
            DL_tauW1_118_err.append((np.percentile(np.log10(DL_tauW1_118[i]), 84.1)-np.percentile(np.log10(DL_tauW1_118[i]), 15.9))/2)
            DL_tauW1_28_err.append((np.percentile(np.log10(DL_tauW1_28[i]), 84.1)-np.percentile(np.log10(DL_tauW1_28[i]), 15.9))/2)
            DL_tauW1_WISE_err.append((np.percentile(np.log10(DL_tauW1_WISE[i]), 84.1)-np.percentile(np.log10(DL_tauW1_WISE[i]), 15.9))/2)

        zCMB_W1.append(zCMB_WISE[i])

# W2 objects
for i in range(len(DL_tauW2)):
    
    if DL_tauW2[i][0] != -999:
        DL_tauW2_clean.append(DL_tauW2[i])
        DL_tauW2_118_clean.append(DL_tauW2_118[i])
        DL_tauW2_28_clean.append(DL_tauW2_28[i])
        DL_tauW2_WISE_clean.append(DL_tauW2_WISE[i])

        if errTog == "SD":
            DL_tauW2_err.append(np.std(np.log10(DL_tauW2[i])))
            DL_tauW2_118_err.append(np.std(np.log10(DL_tauW2_118[i])))
            DL_tauW2_28_err.append(np.std(np.log10(DL_tauW2_28[i])))
            DL_tauW2_WISE_err.append(np.std(np.log10(DL_tauW2_WISE[i])))
            
        elif errTog == "IQ":
            DL_tauW2_err.append((np.percentile(np.log10(DL_tauW2[i]), 84.1)-np.percentile(np.log10(DL_tauW2[i]), 15.9))/2)
            DL_tauW2_118_err.append((np.percentile(np.log10(DL_tauW2_118[i]), 84.1)-np.percentile(np.log10(DL_tauW2_118[i]), 15.9))/2)
            DL_tauW2_28_err.append((np.percentile(np.log10(DL_tauW2_28[i]), 84.1)-np.percentile(np.log10(DL_tauW2_28[i]), 15.9))/2)
            DL_tauW2_WISE_err.append((np.percentile(np.log10(DL_tauW2_WISE[i]), 84.1)-np.percentile(np.log10(DL_tauW2_WISE[i]), 15.9))/2)
        
        zCMB_W2.append(zCMB_WISE[i])

# K objects  
for i in range(len(DL_tauK)):
    if DL_tauK[i][0] != -999:
        DL_tauK_clean.append(DL_tauK[i])
        DL_tauK_err.append(np.std(np.log10(DL_tauK[i])))
        zCMB_K.append(zCMB_mag[i])

"""
Fit cosmology with MPFIT

- Minimize residuals to a model (full luminosity distance integral)
- Utilize hybrid MC + bootstrap 
- Fit with logarithmic residuals (avoid sensitivity to strong linear outliers)

"""

# Define the fit/cosmological distance model (luminosity distance integral)
def cosmoModel(z, H0, omg_0, zStepNum): # Takes redshift list, MPFIT guess H0, fiducial or MPFIT guess omg_0, resolution of redshift grid
    
    z = np.asarray(z, dtype= float)
    zSteps = np.linspace(0,np.max(z), zStepNum) # Gives a grid of equally spaced z values for cumulative numeric integration
    
    denom = omg_0*(1+zSteps)**3 + (1-omg_0) # Denominator of the luminosity distance integrand (1-omg_0 = omg_DE)

    # Calculate a trapezoidal cumulative integral across the redshift grid, used to evaluate model distance at each redshift in z
    runningIntegral = cumulative_trapezoid(1/np.sqrt(denom),zSteps,initial = 0.0)
    
    # For each value in the given z, linearly interpolate along the cumulative integral grid and approximate the integral's value at the given z
    integrals = np.interp(z,zSteps,runningIntegral) # Gives array interpolated of integral evaluations at different z values
    
    return (1+z) * 299792.458/H0 *integrals # Calculate the final luminosity distance

# Define the cosmological fit procedure and residual calculation
def cosmoFit(p, fjac = None, z = None, DL = None,  errDL = None):
    
    z = np.asarray(z,dtype=float)
    DL = np.asarray(DL, dtype=float)
    errDL = np.asarray(errDL)
    
    # Obtain model distance values for the objects in z
    if omgTog and H0Tog:
        model = cosmoModel(z,p[0],p[1],1000)  
    elif H0Tog:
        model = cosmoModel(z,p[0],fidOmg_0,1000)
    elif omgTog:
        model = cosmoModel(z,fidH0,p[0],1000)
    
    # Effective error: Combine object measurement error and intrinsic scatter for residual weighing
    errEff = np.sqrt(errDL**2 + scatter_int**2)
    
    # Return error weighted log10 residuals (fit in log10 space)
    return [0, (np.log10(DL)-np.log10(model))/errEff] 

# Define starting parameter guesses
if omgTog and H0Tog:
    p0 = [70,0.3] # Starting parameter estimates set close to true values: reduces computational time, but does not bias result
elif H0Tog:
    p0 = [70]
elif omgTog:
    p0 = [0.3]

"""
Parameter definitions:

- p0[0]: Hubble Constant
- p0[1]: Mass Dimensionless Density Parameter

- Notice the dark energy dimensionless density parameter is not fit for here: Assuming flat cosmology, omg_DE = 1-omg_0, redundant to fit

"""
# Set limits for valid cosmology parameters
if omgTog and H0Tog:
    parinfo = [
        {
            "limited": [1, 0], # Limited only on bottom side, open on top side
            "limits": [1.0,0.0] # Limit H0 to a min of 1.0: prevent negative or 0 H0
        },
        {
            "limited":[1,1], # Limited on both sides
            "limits":[0.0,1.0] # Limit to >= 0, <= 1 (in flat universe, max omega is 1)
        }
    ]
elif H0Tog:
    parinfo = [
        {
            "limited": [1, 0], # Limited only on bottom side, open on top side
            "limits": [1.0,0.0] # Limit H0 to a min of 1.0: prevent negative or 0 H0
        }
    ]    
elif omgTog:
    parinfo = [
        {
            "limited":[1,1], # Limited on both sides
            "limits":[0.0,1.0] # Limit to >= 0, <= 1 (in flat universe, max omega is 1)
        }
    ]
        
"""
Fit cosmology with MC and bootstrapping

"""

if mcTog:
    
    for i in range(n):
        
        # Define temporary lists for each fit realization, containing the resampled and bootstrapped data
        
        DL_W1_fit = []
        DL_W1_118_fit = []
        DL_W1_28_fit = []
        DL_W1_WISE_fit = []
        DL_W2_fit = []
        DL_W2_118_fit = []
        DL_W2_28_fit = []
        DL_W2_WISE_fit = []
        DL_K_fit = []
        
        # Errors for residual weighing
        errDL_W1_fit = []
        errDL_W1_118_fit = []
        errDL_W1_28_fit = []
        errDL_W1_WISE_fit = []
        errDL_W2_fit = []
        errDL_W2_118_fit = []
        errDL_W2_28_fit = []
        errDL_W2_WISE_fit = []
        errDL_K_fit = []
        
        # Redshifts
        zCMB_W1_fit = []
        zCMB_W2_fit = []
        zCMB_K_fit = []
        
        # Bootsrap + MC logic
        
        # W1 objects
        while len(DL_W1_fit) < len(DL_tauW1_clean): # Select objects (with replacement) until bootstrapped sample is as large as OG dataset
            
            sIndex = random.randint(0,n-1) # Random index for MC resampling 
            index = random.randint(0,len(DL_tauW1_clean)-1) # Random index for bootstrapping
            
            # Append the random, resampled object to the fit lists
            DL_W1_fit.append(DL_tauW1_clean[index][sIndex])
            DL_W1_118_fit.append(DL_tauW1_118_clean[index][sIndex])
            DL_W1_28_fit.append(DL_tauW1_28_clean[index][sIndex])
            DL_W1_WISE_fit.append(DL_tauW1_WISE_clean[index][sIndex])
            
            errDL_W1_fit.append(DL_tauW1_err[index])
            errDL_W1_118_fit.append(DL_tauW1_118_err[index])
            errDL_W1_28_fit.append(DL_tauW1_28_err[index])
            errDL_W1_WISE_fit.append(DL_tauW1_WISE_err[index])

            zCMB_W1_fit.append(zCMB_W1[index])
        
        # W2 objects
        while len(DL_W2_fit) < len(DL_tauW2_clean):
            
            sIndex = random.randint(0,n-1)
            index = random.randint(0, len(DL_tauW2_clean)-1)
            
            DL_W2_fit.append(DL_tauW2_clean[index][sIndex])
            DL_W2_118_fit.append(DL_tauW2_118_clean[index][sIndex])
            DL_W2_28_fit.append(DL_tauW2_28_clean[index][sIndex])
            DL_W2_WISE_fit.append(DL_tauW2_WISE_clean[index][sIndex])

            errDL_W2_fit.append(DL_tauW2_err[index])
            errDL_W2_118_fit.append(DL_tauW2_118_err[index])
            errDL_W2_28_fit.append(DL_tauW2_28_err[index])
            errDL_W2_WISE_fit.append(DL_tauW2_WISE_err[index])
        
            zCMB_W2_fit.append(zCMB_W2[index])
        
        # K objects
        while len(DL_K_fit) < len(DL_tauK_clean):
            sIndex = random.randint(0,n-1)
            index = random.randint(0, len(DL_tauK_clean)-1)
            
            DL_K_fit.append(DL_tauK_clean[index][sIndex])
            errDL_K_fit.append(DL_tauK_err[index])
            
            zCMB_K_fit.append(zCMB_K[index])

        if magnumTog: # Combine MAGNUM and WISE data
            
            zCMB_W1_fit = np.concatenate((zCMB_W1_fit, zCMB_K_fit))
            zCMB_W2_fit = np.concatenate((zCMB_W2_fit, zCMB_K_fit))
            DL_W1_fit = np.concatenate((DL_W1_fit, DL_K_fit))
            DL_W1_118_fit = np.concatenate((DL_W1_118_fit, DL_K_fit))
            DL_W1_28_fit = np.concatenate((DL_W1_28_fit, DL_K_fit))
            DL_W1_WISE_fit = np.concatenate((DL_W1_WISE_fit, DL_K_fit))
            DL_W2_fit = np.concatenate((DL_W2_fit, DL_K_fit))
            DL_W2_118_fit = np.concatenate((DL_W2_118_fit, DL_K_fit))
            DL_W2_28_fit = np.concatenate((DL_W2_28_fit, DL_K_fit))
            DL_W2_WISE_fit = np.concatenate((DL_W2_WISE_fit, DL_K_fit))
            
            errDL_W1_fit = np.concatenate((errDL_W1_fit, errDL_K_fit))
            errDL_W1_118_fit = np.concatenate((errDL_W1_118_fit, errDL_K_fit))
            errDL_W1_28_fit = np.concatenate((errDL_W1_28_fit, errDL_K_fit))
            errDL_W1_WISE_fit = np.concatenate((errDL_W1_WISE_fit, errDL_K_fit))
            errDL_W2_fit = np.concatenate((errDL_W2_fit, errDL_K_fit))
            errDL_W2_118_fit = np.concatenate((errDL_W2_118_fit, errDL_K_fit))
            errDL_W2_28_fit = np.concatenate((errDL_W2_28_fit, errDL_K_fit))
            errDL_W2_WISE_fit = np.concatenate((errDL_W2_WISE_fit, errDL_K_fit))
        
        """
        Run MPFIT
        
        """
        
        mW1= mpfit.mpfit(
            cosmoFit, # Repeatedly call the residual getter
            p0, # Initial guesses
            parinfo = parinfo,
            functkw = {"z":np.array(zCMB_W1_fit, dtype=float), "DL":np.array(DL_W1_fit), "errDL":errDL_W1_fit}, # Dictionary: passes in the data and error into the fit
            quiet = 1 # Turn off console logs
        )

        mW1_118= mpfit.mpfit(
            cosmoFit, 
            p0, 
            parinfo = parinfo,
            functkw = {"z":np.array(zCMB_W1_fit, dtype=float), "DL":np.array(DL_W1_118_fit), "errDL":errDL_W1_118_fit}, 
            quiet = 1 
        )

        mW1_28= mpfit.mpfit(
            cosmoFit, 
            p0, 
            parinfo = parinfo,
            functkw = {"z":np.array(zCMB_W1_fit, dtype=float), "DL":np.array(DL_W1_28_fit), "errDL":errDL_W1_28_fit}, 
            quiet = 1 
        )

        mW2= mpfit.mpfit(
            cosmoFit,
            p0,
            parinfo = parinfo,
            functkw = {"z":np.array(zCMB_W2_fit, dtype=float), "DL":np.array(DL_W2_fit), "errDL":errDL_W2_fit},
            quiet = 1
        )

        mW2_118= mpfit.mpfit(
            cosmoFit, 
            p0, 
            parinfo = parinfo,
            functkw = {"z":np.array(zCMB_W2_fit, dtype=float), "DL":np.array(DL_W2_118_fit), "errDL":errDL_W2_118_fit},
            quiet = 1
        )
        
        mW2_28= mpfit.mpfit(
            cosmoFit,
            p0, 
            parinfo = parinfo,
            functkw = {"z":np.array(zCMB_W2_fit, dtype=float), "DL":np.array(DL_W2_28_fit), "errDL":errDL_W2_28_fit},
            quiet = 1
        )

        mW1_WISE = mpfit.mpfit(
            cosmoFit,
            p0, 
            parinfo = parinfo,
            functkw = {"z":np.array(zCMB_W1_fit, dtype=float), "DL":np.array(DL_W1_WISE_fit), "errDL":errDL_W1_WISE_fit}, 
            quiet = 1
        )

        mW2_WISE= mpfit.mpfit(
            cosmoFit, 
            p0, 
            parinfo = parinfo,
            functkw = {"z":np.array(zCMB_W2_fit, dtype=float), "DL":np.array(DL_W2_WISE_fit), "errDL":errDL_W2_WISE_fit},
            quiet = 1
        )
        
        # Append best fits cosmology parameters to lists for each iteration in the bootsrap-MC procedure
        if H0Tog and omgTog:
            
            H0_W1.append(float(mW1.params[0]))
            H0_W1_118.append(float(mW1_118.params[0]))
            H0_W1_28.append(float(mW1_28.params[0]))
            H0_W1_WISE.append(float(mW1_WISE.params[0]))

            H0_W2.append(float(mW2.params[0]))
            H0_W2_118.append(float(mW2_118.params[0]))
            H0_W2_28.append(float(mW2_28.params[0]))
            H0_W2_WISE.append(float(mW2_WISE.params[0]))
            
            omg_0_W1.append(float(mW1.params[1]))
            omg_0_W1_118.append(float(mW1_118.params[1]))
            omg_0_W1_28.append(float(mW1_28.params[1]))
            omg_0_W1_WISE.append(float(mW1_WISE.params[1]))

            omg_0_W2.append(float(mW2.params[1]))
            omg_0_W2_118.append(float(mW2_118.params[1]))
            omg_0_W2_28.append(float(mW2_28.params[1]))
            omg_0_W2_WISE.append(float(mW2_WISE.params[1]))
            
        elif H0Tog:
            
            H0_W1.append(float(mW1.params[0]))
            H0_W1_118.append(float(mW1_118.params[0]))
            H0_W1_28.append(float(mW1_28.params[0]))
            H0_W1_WISE.append(float(mW1_WISE.params[0]))

            H0_W2.append(float(mW2.params[0]))
            H0_W2_118.append(float(mW2_118.params[0]))
            H0_W2_28.append(float(mW2_28.params[0]))
            H0_W2_WISE.append(float(mW2_WISE.params[0]))

            omg_0_W1.append(fidOmg_0)
            omg_0_W1_118.append(fidOmg_0)
            omg_0_W1_28.append(fidOmg_0)
            omg_0_W1_WISE.append(fidOmg_0)
            omg_0_W2.append(fidOmg_0)
            omg_0_W2_118.append(fidOmg_0)
            omg_0_W2_28.append(fidOmg_0)
            omg_0_W2_WISE.append(fidOmg_0)
        
        elif omgTog:
            
            omg_0_W1.append(float(mW1.params[0]))
            omg_0_W1_118.append(float(mW1_118.params[0]))
            omg_0_W1_28.append(float(mW1_28.params[0]))
            omg_0_W1_WISE.append(float(mW1_WISE.params[0]))

            omg_0_W2.append(float(mW2.params[0]))
            omg_0_W2_118.append(float(mW2_118.params[0]))
            omg_0_W2_28.append(float(mW2_28.params[0]))
            omg_0_W2_WISE.append(float(mW2_WISE.params[0]))

            H0_W1.append(fidH0)
            H0_W1_118.append(fidH0)
            H0_W1_28.append(fidH0)
            H0_W1_WISE.append(fidH0)
            H0_W2.append(fidH0)
            H0_W2_118.append(fidH0)
            H0_W2_28.append(fidH0)
            H0_W2_WISE.append(fidH0)

    # Print median cosmologies with standard deviation
    print(f"H0 W1,W2: {np.median(H0_W1)} ± {np.std(H0_W1)}, {np.median(H0_W2)} ± {np.std(H0_W2)}")
    print(f"H0_118 W1,W2: {np.median(H0_W1_118)} ± {np.std(H0_W1_118)}, {np.median(H0_W2_118)} ± {np.std(H0_W2_118)}")
    print(f"H0_28 W1,W2: {np.median(H0_W1_28)} ± {np.std(H0_W1_28)}, {np.median(H0_W2_28)} ± {np.std(H0_W2_28)}")
    print(f"H0_WISE W1,W2: {np.median(H0_W1_WISE)} ± {np.std(H0_W1_WISE)}, {np.median(H0_W2_WISE)} ± {np.std(H0_W2_WISE)}")

    print(f"omg_0 W1,W2: {np.median(omg_0_W1)}, {np.median(omg_0_W2)}")
    print(f"omg_0_118 W1,W2: {np.median(omg_0_W1_118)}, {np.median(omg_0_W2_118)}")
    print(f"omg_0_28 W1,W2: {np.median(omg_0_W1_28)}, {np.median(omg_0_W2_28)}")
    print(f"omg_0_WISE W1,W2: {np.median(omg_0_W1_WISE)}, {np.median(omg_0_W2_WISE)}")

"""
Fit cosmology with MPFIT (no MC + boostrap)

"""

# Data: Medians of the luminosity distance distributions for each object
DL_tauW1_meds = []
DL_tauW1_118_meds = []
DL_tauW1_28_meds = []
DL_tauW1_WISE_meds = []
DL_tauW2_meds = []
DL_tauW2_118_meds = []
DL_tauW2_28_meds = []
DL_tauW2_WISE_meds = []
DL_tauK_meds = []

# Get medians

# W1 objects
for i in range(len(DL_tauW1_clean)):
    DL_tauW1_meds.append(np.median(DL_tauW1_clean[i]))
    DL_tauW1_118_meds.append(np.median(DL_tauW1_118_clean[i]))
    DL_tauW1_28_meds.append(np.median(DL_tauW1_28_clean[i]))
    DL_tauW1_WISE_meds.append(np.median(DL_tauW1_WISE_clean[i]))

# W2 objects
for i in range(len(DL_tauW2_clean)):
    DL_tauW2_meds.append(np.median(DL_tauW2_clean[i]))
    DL_tauW2_118_meds.append(np.median(DL_tauW2_118_clean[i]))
    DL_tauW2_28_meds.append(np.median(DL_tauW2_28_clean[i]))
    DL_tauW2_WISE_meds.append(np.median(DL_tauW2_WISE_clean[i]))

# K objects
for i in range(len(DL_tauK_clean)):
    DL_tauK_meds.append(np.median(DL_tauK_clean[i]))
    
if mcTog == False: 
    
    mW1 = mpfit.mpfit(
        cosmoFit, 
        p0, 
        parinfo = parinfo,
        functkw = {"z":np.array(zCMB_W1, dtype=float), "DL":np.array([np.median(i) for i in DL_tauW1_clean]), "errDL":DL_tauW1_err}, 
        quiet = 1
    )

    mW1_118 = mpfit.mpfit(
        cosmoFit, 
        p0, 
        parinfo = parinfo,
        functkw = {"z":np.array(zCMB_W1, dtype=float), "DL":np.array([np.median(i) for i in DL_tauW1_118_clean]), "errDL":DL_tauW1_118_err}, 
        quiet = 1
    )

    mW1_28 = mpfit.mpfit(
        cosmoFit, 
        p0, 
        parinfo = parinfo,
        functkw = {"z":np.array(zCMB_W1, dtype=float), "DL":np.array([np.median(i) for i in DL_tauW1_28_clean]), "errDL":DL_tauW1_28_err}, 
        quiet = 1
    )
    
    mW2 = mpfit.mpfit(
        cosmoFit, 
        p0, 
        parinfo = parinfo,
        functkw = {"z":np.array(zCMB_W2, dtype=float), "DL":np.array([np.median(i) for i in DL_tauW2_clean]), "errDL":DL_tauW2_err}, 
        quiet = 1
    )

    mW2_118 = mpfit.mpfit(
        cosmoFit, 
        p0, 
        parinfo = parinfo,
        functkw = {"z":np.array(zCMB_W2, dtype=float), "DL":np.array([np.median(i) for i in DL_tauW2_118_clean]), "errDL":DL_tauW2_118_err}, 
        quiet = 1
    )
    
    mW2_28 = mpfit.mpfit(
        cosmoFit, 
        p0, 
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_W2, dtype=float), "DL":np.array([np.median(i) for i in DL_tauW2_28_clean]), "errDL":DL_tauW2_28_err}, 
        quiet = 1
    )

    mW1_WISE = mpfit.mpfit(
        cosmoFit, 
        p0, 
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_W1, dtype=float), "DL":np.array([np.median(i) for i in DL_tauW1_WISE_clean]), "errDL":DL_tauW1_WISE_err}, 
        quiet = 1
    )

    mW2_WISE = mpfit.mpfit(
        cosmoFit, 
        p0, 
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_W2, dtype=float), "DL":np.array([np.median(i) for i in DL_tauW2_WISE_clean]), "errDL":DL_tauW2_WISE_err}, 
        quiet = 1
    )

    if H0Tog and omgTog:
        
        H0_W1=float(mW1.params[0])
        H0_W1_118=float(mW1_118.params[0])
        H0_W1_28=float(mW1_28.params[0])
        H0_W1_WISE = float(mW1_WISE.params[0])

        H0_W2=float(mW2.params[0])
        H0_W2_118=float(mW2_118.params[0])
        H0_W2_28=float(mW2_28.params[0])
        H0_W2_WISE = float(mW2_WISE.params[0])
        
        omg_0_W1=float(mW1.params[1])
        omg_0_W1_118=float(mW1_118.params[1])
        omg_0_W1_28=float(mW1_28.params[1])
        omg_0_W1_WISE = float(mW1_WISE.params[1])

        omg_0_W2=float(mW2.params[1])
        omg_0_W2_118=float(mW2_118.params[1])
        omg_0_W2_28=float(mW2_28.params[1])
        omg_0_W2_WISE = float(mW2_WISE.params[1])
        
    elif H0Tog:
        
        H0_W1=float(mW1.params[0])
        H0_W1_118=float(mW1_118.params[0])
        H0_W1_28=float(mW1_28.params[0])
        H0_W1_WISE = float(mW1_WISE.params[0])

        H0_W2=float(mW2.params[0])
        H0_W2_118=float(mW2_118.params[0])
        H0_W2_28=float(mW2_28.params[0])
        H0_W2_WISE = float(mW2_WISE.params[0])

        omg_0_W1=fidOmg_0
        omg_0_W1_118=fidOmg_0
        omg_0_W1_28=fidOmg_0
        omg_0_W1_WISE=fidOmg_0
        omg_0_W2=fidOmg_0
        omg_0_W2_118=fidOmg_0
        omg_0_W2_28=fidOmg_0
        omg_0_W2_WISE=fidOmg_0
        
    elif omgTog:
        
        omg_0_W1=float(mW1.params[0])
        omg_0_W1_118=float(mW1_118.params[0])
        omg_0_W1_28=float(mW1_28.params[0])
        omg_0_W1_WISE = float(mW1_WISE.params[0])

        omg_0_W2=float(mW2.params[0])
        omg_0_W2_118=float(mW2_118.params[0])
        omg_0_W2_28=float(mW2_28.params[0])
        omg_0_W2_WISE = float(mW2_WISE.params[0])
        
        H0_W1.append(fidH0)
        H0_W1_118.append(fidH0)
        H0_W1_28.append(fidH0)
        H0_W1_WISE.append(fidH0)
        H0_W2.append(fidH0)
        H0_W2_118.append(fidH0)
        H0_W2_28.append(fidH0)
        H0_W2_WISE.append(fidH0)

    print(f"H0 W1,W2: {H0_W1}, {H0_W2}")
    print(f"H0_118 W1,W2: {H0_W1_118}, {H0_W2_118}")
    print(f"H0_28 W1,W2: {H0_W1_28}, {H0_W2_28}")
    print(f"H0_WISE W1,W2: {H0_W1_WISE}, {H0_W2_WISE}")

    print(f"omg_0 W1,W2: {omg_0_W1}, {omg_0_W2}")
    print(f"omg_0_118 W1,W2: {omg_0_W1_118}, {omg_0_W2_118}")
    print(f"omg_0_28 W1,W2: {omg_0_W1_28}, {omg_0_W2_28}")
    print(f"omg_0_WISE W1,W2: {omg_0_W1_WISE}, {omg_0_W2_WISE}")

"""
Plot with matplotlib

"""
# Return list of luminosity distances corresponding to redshift points given a cosmology
def getDLPlot(z, H0, omg_0):
    
    # If H0 stored in a list
    if type(H0) != float:
        return(
            (1+z) * 
            299792.458/np.median(H0) * 
            cumulative_trapezoid(
                1/np.sqrt(np.median(omg_0)* 
                (1+z)**3 + (1-np.median(omg_0))),
                z,initial = 0.0)
               )
    
    # If scalar H0
    else:
        return(
            (1+z) * 
            299792.458/H0 * 
            cumulative_trapezoid(
                1/np.sqrt(omg_0* 
                (1+z)**3 + (1-omg_0)),
                z,initial = 0.0)
               )

# Best fit cosmology plot: log log scales

fig, ax = plt.subplots(2,4)
fig.suptitle("Best fit cosmology, Log Log Scales")

# Set log10 scales
for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax[i,j].set_xscale("log")
        ax[i,j].set_yscale("log")

# Plotted DL errorbars correspond to standard deviation in logarithmic scales

ax[0,0].errorbar(
    zCMB_W1,DL_tauW1_meds, 
    yerr = [np.asarray(DL_tauW1_meds) - 10**(np.log10(np.asarray(DL_tauW1_meds))-np.asarray(DL_tauW1_err)),
        10**(np.log10(np.asarray(DL_tauW1_meds))+np.asarray(DL_tauW1_err)) - np.asarray(DL_tauW1_meds)],
    elinewidth = 0.3, ms = 1, color = "blue", fmt = "o"
    )

ax[1,0].errorbar(
    zCMB_W2,DL_tauW2_meds, 
    yerr = [np.asarray(DL_tauW2_meds) - 10**(np.log10(np.asarray(DL_tauW2_meds))-np.asarray(DL_tauW2_err)),
        10**(np.log10(np.asarray(DL_tauW2_meds))+np.asarray(DL_tauW2_err)) - np.asarray(DL_tauW2_meds)],
    elinewidth = 0.3, ms = 1, color = "red", fmt = "o"
    )

ax[0,1].errorbar(
    zCMB_W1,DL_tauW1_118_meds, 
    yerr = [np.asarray(DL_tauW1_118_meds) - 10**(np.log10(np.asarray(DL_tauW1_118_meds))-np.asarray(DL_tauW1_118_err)),
        10**(np.log10(np.asarray(DL_tauW1_118_meds))+np.asarray(DL_tauW1_118_err)) - np.asarray(DL_tauW1_118_meds)],
    elinewidth = 0.3, ms = 1, color = "blue", fmt = "o"
    )

ax[1,1].errorbar(
    zCMB_W2,DL_tauW2_118_meds, 
    yerr = [np.asarray(DL_tauW2_118_meds) - 10**(np.log10(np.asarray(DL_tauW2_118_meds))-np.asarray(DL_tauW2_118_err)),
        10**(np.log10(np.asarray(DL_tauW2_118_meds))+np.asarray(DL_tauW2_118_err)) - np.asarray(DL_tauW2_118_meds)],
    elinewidth = 0.3, ms = 1, color = "red", fmt = "o"
    )

ax[0,2].errorbar(
    zCMB_W1,DL_tauW1_28_meds, 
    yerr = [np.asarray(DL_tauW1_28_meds) - 10**(np.log10(np.asarray(DL_tauW1_28_meds))-np.asarray(DL_tauW1_28_err)),
        10**(np.log10(np.asarray(DL_tauW1_28_meds))+np.asarray(DL_tauW1_28_err)) - np.asarray(DL_tauW1_28_meds)],
    elinewidth = 0.3, ms = 1, color = "blue", fmt = "o"
    )

ax[1,2].errorbar(
    zCMB_W2,DL_tauW2_28_meds, 
    yerr = [np.asarray(DL_tauW2_28_meds) - 10**(np.log10(np.asarray(DL_tauW2_28_meds))-np.asarray(DL_tauW2_28_err)),
        10**(np.log10(np.asarray(DL_tauW2_28_meds))+np.asarray(DL_tauW2_28_err)) - np.asarray(DL_tauW2_28_meds)],
    elinewidth = 0.3, ms = 1, color = "red", fmt = "o"
    )

ax[0,3].errorbar(
    zCMB_W1,DL_tauW1_WISE_meds, 
    yerr = [np.asarray(DL_tauW1_WISE_meds) - 10**(np.log10(np.asarray(DL_tauW1_WISE_meds))-np.asarray(DL_tauW1_WISE_err)),
        10**(np.log10(np.asarray(DL_tauW1_WISE_meds))+np.asarray(DL_tauW1_WISE_err)) - np.asarray(DL_tauW1_WISE_meds)],
    elinewidth = 0.3, ms = 1, color = "blue", fmt = "o"
    )

ax[1,3].errorbar(
    zCMB_W2,DL_tauW2_WISE_meds, 
    yerr = [np.asarray(DL_tauW2_WISE_meds) - 10**(np.log10(np.asarray(DL_tauW2_WISE_meds))-np.asarray(DL_tauW2_WISE_err)),
        10**(np.log10(np.asarray(DL_tauW2_WISE_meds))+np.asarray(DL_tauW2_WISE_err)) - np.asarray(DL_tauW2_WISE_meds)],
    elinewidth = 0.3, ms = 1, color = "red", fmt = "o"
    )

if magnumTog:
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i,j].errorbar(
                zCMB_K,DL_tauK_meds, 
                yerr = [np.asarray(DL_tauK_meds) - 10**(np.log10(np.asarray(DL_tauK_meds))-np.asarray(DL_tauK_err)),
                    10**(np.log10(np.asarray(DL_tauK_meds))+np.asarray(DL_tauK_err)) - np.asarray(DL_tauK_meds)],
                elinewidth = 0.3, ms = 1, color = "green", fmt = "s"
                )

# Plot lines of best fit and fiducial cosmology

zPlot = np.linspace(0,np.max(zCMB_WISE),1000)

ax[0,0].plot(zPlot,getDLPlot(zPlot,H0_W1,omg_0_W1), color = "black")
ax[1,0].plot(zPlot,getDLPlot(zPlot,H0_W2,omg_0_W2), color = "black")
ax[0,1].plot(zPlot,getDLPlot(zPlot,H0_W1_118,omg_0_W1_118), color = "black")
ax[1,1].plot(zPlot,getDLPlot(zPlot,H0_W2_118,omg_0_W2_118), color = "black")
ax[0,2].plot(zPlot,getDLPlot(zPlot,H0_W1_28,omg_0_W1_28), color = "black")
ax[1,2].plot(zPlot,getDLPlot(zPlot,H0_W2_28,omg_0_W2_28), color = "black")
ax[0,3].plot(zPlot,getDLPlot(zPlot,H0_W1_WISE,omg_0_W1_WISE), color = "black")
ax[1,3].plot(zPlot,getDLPlot(zPlot,H0_W2_WISE,omg_0_W2_WISE), color = "black")

for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax[i,j].plot(zPlot,getDLPlot(zPlot,73.04,0.31),ls = "--",  color = "orange")

if magnumTog:  
    for i in range(len(ax[0])):
        ax[0,i].set_xlim(np.min(np.concatenate((zCMB_W1,zCMB_K))),np.max(np.concatenate((zCMB_W1,zCMB_K))))
    for i in range(len(ax[1])):
        ax[1,i].set_xlim(np.min(np.concatenate((zCMB_W2,zCMB_K))),np.max(np.concatenate((zCMB_W2,zCMB_K))))
        
    ax[0,0].set_ylim(np.min(np.concatenate((DL_tauW1_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW1_meds,DL_tauK_meds))))
    ax[0,1].set_ylim(np.min(np.concatenate((DL_tauW1_118_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW1_118_meds,DL_tauK_meds))))
    ax[0,2].set_ylim(np.min(np.concatenate((DL_tauW1_28_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW1_28_meds,DL_tauK_meds))))
    ax[0,3].set_ylim(np.min(np.concatenate((DL_tauW1_WISE_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW1_WISE_meds,DL_tauK_meds))))

    ax[1,0].set_ylim(np.min(np.concatenate((DL_tauW2_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW2_meds,DL_tauK_meds))))
    ax[1,1].set_ylim(np.min(np.concatenate((DL_tauW2_118_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW2_118_meds,DL_tauK_meds))))
    ax[1,2].set_ylim(np.min(np.concatenate((DL_tauW2_28_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW2_28_meds,DL_tauK_meds))))
    ax[1,3].set_ylim(np.min(np.concatenate((DL_tauW2_WISE_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW2_WISE_meds,DL_tauK_meds))))
    
else:
    for i in range(len(ax[0])):
        ax[0,i].set_xlim(np.min(zCMB_W1),np.max(zCMB_W1))
    for i in range(len(ax[1])):
        ax[1,i].set_xlim(np.min(zCMB_W2),np.max(zCMB_W2))
    
    ax[0,0].set_ylim(np.min(DL_tauW1_meds), np.max(DL_tauW1_meds))
    ax[0,1].set_ylim(np.min(DL_tauW1_118_meds), np.max(DL_tauW1_118_meds))
    ax[0,2].set_ylim(np.min(DL_tauW1_28_meds), np.max(DL_tauW1_28_meds))
    ax[0,3].set_ylim(np.min(DL_tauW1_WISE_meds), np.max(DL_tauW1_WISE_meds))

    ax[1,0].set_ylim(np.min(DL_tauW2_meds), np.max(DL_tauW2_meds))
    ax[1,1].set_ylim(np.min(DL_tauW2_118_meds), np.max(DL_tauW2_118_meds))
    ax[1,2].set_ylim(np.min(DL_tauW2_28_meds), np.max(DL_tauW2_28_meds))
    ax[1,3].set_ylim(np.min(DL_tauW2_WISE_meds), np.max(DL_tauW2_WISE_meds))

# Best fit cosmology plot: linear scales

fig2, ax2 = plt.subplots(2,4)
fig2.suptitle("Best fit cosmology, Linear Scales")

ax2[0,0].errorbar(zCMB_W1,DL_tauW1_meds, yerr = [(np.percentile(i,84.1) - np.percentile(i,15.9))/2 for i in DL_tauW1_clean], elinewidth = 0.3, ms = 1, color = "blue", fmt = "o")
ax2[1,0].errorbar(zCMB_W2,DL_tauW2_meds, yerr = [(np.percentile(i,84.1) - np.percentile(i,15.9))/2 for i in DL_tauW2_clean], elinewidth = 0.3, ms = 1, color = "red", fmt = "o")
ax2[0,1].errorbar(zCMB_W1,DL_tauW1_118_meds, yerr = [(np.percentile(i,84.1) - np.percentile(i,15.9))/2 for i in DL_tauW1_118_clean], elinewidth = 0.3, ms = 1, color = "blue", fmt = "o")
ax2[1,1].errorbar(zCMB_W2,DL_tauW2_118_meds, yerr = [(np.percentile(i,84.1) - np.percentile(i,15.9))/2 for i in DL_tauW2_118_clean], elinewidth = 0.3, ms = 1, color = "red", fmt = "o")
ax2[0,2].errorbar(zCMB_W1,DL_tauW1_28_meds, yerr = [(np.percentile(i,84.1) - np.percentile(i,15.9))/2 for i in DL_tauW1_28_clean], elinewidth = 0.3, ms = 1, color = "blue", fmt = "o")
ax2[1,2].errorbar(zCMB_W2,DL_tauW2_28_meds, yerr = [(np.percentile(i,84.1) - np.percentile(i,15.9))/2 for i in DL_tauW2_28_clean], elinewidth = 0.3, ms = 1, color = "red", fmt = "o")
ax2[0,3].errorbar(zCMB_W1,DL_tauW1_WISE_meds, yerr = [(np.percentile(i,84.1) - np.percentile(i,15.9))/2 for i in DL_tauW1_WISE_clean], elinewidth = 0.3, ms = 1, color = "blue", fmt = "o")
ax2[1,3].errorbar(zCMB_W2,DL_tauW2_WISE_meds, yerr = [(np.percentile(i,84.1) - np.percentile(i,15.9))/2 for i in DL_tauW2_WISE_clean], elinewidth = 0.3, ms = 1, color = "red", fmt = "o")

if magnumTog:
    for i in range(len(ax2)):
        for j in range(len(ax2[i])):
            ax2[i,j].errorbar(zCMB_K,DL_tauK_meds, yerr = [(np.percentile(i,84.1) - np.percentile(i,15.9))/2 for i in DL_tauK_clean], elinewidth = 0.3, ms = 1, color = "green", fmt = "s")

zPlot = np.linspace(0,np.max(zCMB_WISE),1000)

ax2[0,0].plot(zPlot,getDLPlot(zPlot,H0_W1,omg_0_W1), color = "black")
ax2[1,0].plot(zPlot,getDLPlot(zPlot,H0_W2,omg_0_W2), color = "black")
ax2[0,1].plot(zPlot,getDLPlot(zPlot,H0_W1_118,omg_0_W1_118), color = "black")
ax2[1,1].plot(zPlot,getDLPlot(zPlot,H0_W2_118,omg_0_W2_118), color = "black")
ax2[0,2].plot(zPlot,getDLPlot(zPlot,H0_W1_28,omg_0_W1_28), color = "black")
ax2[1,2].plot(zPlot,getDLPlot(zPlot,H0_W2_28,omg_0_W2_28), color = "black")
ax2[0,3].plot(zPlot,getDLPlot(zPlot,H0_W1_WISE,omg_0_W1_WISE), color = "black")
ax2[1,3].plot(zPlot,getDLPlot(zPlot,H0_W2_WISE,omg_0_W2_WISE), color = "black")

for i in range(len(ax2)):
    for j in range(len(ax2[i])):
        ax2[i,j].plot(zPlot,getDLPlot(zPlot,73.04,0.31),ls = "--",  color = "orange")

if magnumTog:  
    for i in range(len(ax2[0])):
        ax2[0,i].set_xlim(np.min(np.concatenate((zCMB_W1,zCMB_K))),np.max(np.concatenate((zCMB_W1,zCMB_K))))
    for i in range(len(ax2[1])):
        ax2[1,i].set_xlim(np.min(np.concatenate((zCMB_W2,zCMB_K))),np.max(np.concatenate((zCMB_W2,zCMB_K))))
        
    ax2[0,0].set_ylim(np.min(np.concatenate((DL_tauW1_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW1_meds,DL_tauK_meds))))
    ax2[0,1].set_ylim(np.min(np.concatenate((DL_tauW1_118_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW1_118_meds,DL_tauK_meds))))
    ax2[0,2].set_ylim(np.min(np.concatenate((DL_tauW1_28_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW1_28_meds,DL_tauK_meds))))
    ax2[0,3].set_ylim(np.min(np.concatenate((DL_tauW1_WISE_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW1_WISE_meds,DL_tauK_meds))))

    ax2[1,0].set_ylim(np.min(np.concatenate((DL_tauW2_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW2_meds,DL_tauK_meds))))
    ax2[1,1].set_ylim(np.min(np.concatenate((DL_tauW2_118_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW2_118_meds,DL_tauK_meds))))
    ax2[1,2].set_ylim(np.min(np.concatenate((DL_tauW2_28_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW2_28_meds,DL_tauK_meds))))
    ax2[1,3].set_ylim(np.min(np.concatenate((DL_tauW2_WISE_meds,DL_tauK_meds))),np.max(np.concatenate((DL_tauW2_WISE_meds,DL_tauK_meds))))
    
else:
    for i in range(len(ax[0])):
        ax[0,i].set_xlim(np.min(zCMB_W1),np.max(zCMB_W1))
    for i in range(len(ax[1])):
        ax[1,i].set_xlim(np.min(zCMB_W2),np.max(zCMB_W2))
    
    ax[0,0].set_ylim(np.min(DL_tauW1_meds), np.max(DL_tauW1_meds))
    ax[0,1].set_ylim(np.min(DL_tauW1_118_meds), np.max(DL_tauW1_118_meds))
    ax[0,2].set_ylim(np.min(DL_tauW1_28_meds), np.max(DL_tauW1_28_meds))
    ax[0,3].set_ylim(np.min(DL_tauW1_WISE_meds), np.max(DL_tauW1_WISE_meds))

    ax[1,0].set_ylim(np.min(DL_tauW2_meds), np.max(DL_tauW2_meds))
    ax[1,1].set_ylim(np.min(DL_tauW2_118_meds), np.max(DL_tauW2_118_meds))
    ax[1,2].set_ylim(np.min(DL_tauW2_28_meds), np.max(DL_tauW2_28_meds))
    ax[1,3].set_ylim(np.min(DL_tauW2_WISE_meds), np.max(DL_tauW2_WISE_meds))

# Plot histograms of cosmology paramater distributions (from MC + bootstrapping)

if mcTog: # Plot H0 distributions
    
    fig3, ax3 = plt.subplots(2,4)
    fig3.suptitle("H0 distributions")

    histH0_W1, binH0_W1 = np.histogram(H0_W1, bins = "auto")
    histH0_W2, binH0_W2 = np.histogram(H0_W2, bins = "auto")
    histH0_W1_118, binH0_W1_118 = np.histogram(H0_W1_118, bins = "auto")
    histH0_W2_118, binH0_W2_118 = np.histogram(H0_W2_118, bins = "auto")
    histH0_W1_28, binH0_W1_28 = np.histogram(H0_W1_28, bins = "auto")
    histH0_W2_28, binH0_W2_28 = np.histogram(H0_W2_28, bins = "auto")
    histH0_W1_WISE, binH0_W1_WISE = np.histogram(H0_W1_WISE, bins = "auto")
    histH0_W2_WISE, binH0_W2_WISE = np.histogram(H0_W2_WISE, bins = "auto")

    ax3[0,0].stairs(histH0_W1, binH0_W1, color = "blue")
    ax3[1,0].stairs(histH0_W2, binH0_W2, color = "red")
    ax3[0,1].stairs(histH0_W1_118, binH0_W1_118, color = "blue")
    ax3[1,1].stairs(histH0_W2_118, binH0_W2_118, color = "red")
    ax3[0,2].stairs(histH0_W1_28, binH0_W1_28, color = "blue")
    ax3[1,2].stairs(histH0_W2_28, binH0_W2_28, color = "red")
    ax3[0,3].stairs(histH0_W1_WISE, binH0_W1_WISE, color = "blue")
    ax3[1,3].stairs(histH0_W2_WISE, binH0_W2_WISE, color = "red")

    ax3[0,0].axvline(x=np.median(H0_W1), color = "green", linestyle = "--")
    ax3[1,0].axvline(x=np.median(H0_W2), color = "green", linestyle = "--")
    ax3[0,1].axvline(x=np.median(H0_W1_118), color = "green", linestyle = "--")
    ax3[1,1].axvline(x=np.median(H0_W2_118), color = "green", linestyle = "--")
    ax3[0,2].axvline(x=np.median(H0_W1_28), color = "green", linestyle = "--")
    ax3[1,2].axvline(x=np.median(H0_W2_28), color = "green", linestyle = "--")
    ax3[0,3].axvline(x=np.median(H0_W1_WISE), color = "green", linestyle = "--")
    ax3[1,3].axvline(x=np.median(H0_W2_WISE), color = "green", linestyle = "--")

    if omgTog: # Plot omg_0 distributions

        fig4, ax4 = plt.subplots(2,3)
        fig4.suptitle("omg_0 distributions")

        histomg_0_W1, binomg_0_W1 = np.histogram(omg_0_W1, bins = "auto")
        histomg_0_W2, binomg_0_W2 = np.histogram(omg_0_W2, bins = "auto")
        histomg_0_W1_118, binomg_0_W1_118 = np.histogram(omg_0_W1_118, bins = "auto")
        histomg_0_W2_118, binomg_0_W2_118 = np.histogram(omg_0_W2_118, bins = "auto")
        histomg_0_W1_28, binomg_0_W1_28 = np.histogram(omg_0_W1_28, bins = "auto")
        histomg_0_W2_28, binomg_0_W2_28 = np.histogram(omg_0_W2_28, bins = "auto")

        ax4[0,0].stairs(histomg_0_W1, binomg_0_W1, edgecolor = "blue", fill = "False")
        ax4[1,0].stairs(histomg_0_W2, binomg_0_W2, edgecolor = "red", fill = "False")
        ax4[0,1].stairs(histomg_0_W1_118, binomg_0_W1_118, edgecolor = "blue", fill = "False")
        ax4[1,1].stairs(histomg_0_W2_118, binomg_0_W2_118, edgecolor = "red", fill = "False")
        ax4[0,2].stairs(histomg_0_W1_28, binomg_0_W1_28, edgecolor = "blue", fill = "False")
        ax4[1,2].stairs(histomg_0_W2_28, binomg_0_W2_28, edgecolor = "red", fill = "False")


        ax4[0,0].axvline(x=np.median(omg_0_W1), color = "green", linestyle = "--")
        ax4[1,0].axvline(x=np.median(omg_0_W2), color = "green", linestyle = "--")
        ax4[0,1].axvline(x=np.median(omg_0_W1_118), color = "green", linestyle = "--")
        ax4[1,1].axvline(x=np.median(omg_0_W2_118), color = "green", linestyle = "--")
        ax4[0,2].axvline(x=np.median(omg_0_W1_28), color = "green", linestyle = "--")
        ax4[1,2].axvline(x=np.median(omg_0_W2_28), color = "green", linestyle = "--")
    
plt.show()