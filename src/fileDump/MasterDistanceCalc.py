"""
Master Distance Calculator

- Uses time lags, flux, and reported errors to generate probability distributions of luminosity distances

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
from astropy.coordinates import SkyCoord
from astropy import units as u
import scipy
from scipy.integrate import quad

n = 10000 # Number of Monte Carlo iterations
alpha = -0.5 # Set fiducial alphaUV (if alphSampTog = False)
g = 10.60 # Set fiducial g calibration constant (if alphSampTog = False)

propTog = True # Toggle use of WISE lag correction from R-L intercept proportionality 
alphSampTog = True # Toggle use of alphaUV sampled values for the g calibration constant
flowCorrTog = True # TODO:Toggle use of flow corrected redshifts for MAGNUM objects

if propTog: # Ratios from K-band to W-band obtained from R-L intercept proportionality

    # Ratio K/W1: 0.8550004300996275 +-0.059315355436090726
    # Ratio K/W1_118: 0.7893572157374362 +-0.05462425611556031
    # Ratio K/W1_28: 0.7202387717642525 +-0.05337453036871022
    # Ratio K/W1_WISE: 0.6226862155040334 +-0.04290857390108669
    # Ratio K/W2: 0.8374698487482309 +-0.05819199657460148
    # Ratio K/W2_118: 0.9111525325435578 +-0.06069048518477022
    # Ratio K/W2_28: 1.3495313544489478 +-0.09366527685822644
    # Ratio K/W2_WISE: 0.5206376443528672 +-0.0352489453583753

    
    # From false cosmology with H0 = 10, omg_0 = 0.27
    propW1 = 0.8339777542062594 #0.858500951779403 
    propW1_118 = 0.7662225926898056 #0.7871855553377065
    propW1_28 = 0.6980561942222575 #0.7171336421897777
    propW2 = 0.8196435577398279 #0.8443455546344151
    propW2_118 = 0.8931597407605285 #0.9173568011167653
    propW2_28 =  1.3261757782690278 #1.3620658764726608
    propW1_WISE = 0.6393042877459436 #0.6520478008341523
    propW2_WISE = 0.49607665224109965 #0.5068266246895858
    
else:
    propW1 = 1 
    propW1_118 = 1 
    propW1_28 = 1
    propW2 = 1
    propW2_118 = 1
    propW2_28 = 1
    propW1_WISE = 1
    propW2_WISE = 1

# Read Data (missing values: -999)

# WISE data
objID, SDSSID, zReported, zHel, logL, logL_err, tauW1, tauW1_err, tauW1_Err, tauW2, tauW2_err, tauW2_Err =np.genfromtxt(
    "WISEData.csv", delimiter=",",usecols = (0,2,3,5,16,17,26,27,28,39,40,41), 
    skip_header=1, unpack = True, missing_values=("","#NUM!"), filling_values=-999, dtype=None, comments= None) 

# MAGNUM data
zCMB_mag, tauK, tauK_err, tauK_Err, logLmag, logLmag_err, flux_mag, flux_mag_Err, flux_mag_err, obsBand = np.genfromtxt(
    "MAGNUMData.csv", delimiter=",", usecols = (2,5,6,7,10,13,14,15,16, 20), 
    skip_header=1, unpack = True, missing_values=(""), filling_values=-999, dtype=None
)

if alphSampTog:
    gDistr, alphDistrL, alphDistrS = np.genfromtxt(
        "SampledG.txt", delimiter="|",unpack = True
    )
    
zCMB_WISE = [] # CMB rest frame redshift for WISE data
zFlowCorr = []

"""
Distances

Each distance list is a 2-dimensional array
- First index = object
- Second index = MC sampling realization
- Each object (sublist) has shape n where n is the number of MC samples

"""

# Luminosity distances (Mpc)
DL_given = [] # Inferred from stated fiducial cosmologies in WISE papers: Used to convert luminosity data back into flux data
DL_tauW1 = []
DL_tauW2 = []
DL_tauW1_118 = []
DL_tauW2_118 = []
DL_tauW1_28 = []
DL_tauW2_28 = []
DL_tauW1_WISE = []
DL_tauW2_WISE = []
DL_tauK = []

# Spectral fluxes for WISE objects (inferred from DL_given and logL, erg/s/Hz/cm**2)
flux = []
flux_err = []
flux_Err = []

"""
Time Lags (light days)

Each time lag list is a 2-dimensional array
- First index = object
- Second index = MC sampling realization
- Each object (sublist) has shape n where n is the number of MC samples

"""

# Raw time lags (Mandal's correction factor to the observational band with gamma = 0.62 removed)
rTauW1 = []
rTauW1_err = []
rTauW1_Err = []
rTauW2 = []
rTauW2_err = []
rTauW2_Err = []

# Sampled time lags from split-normal distribution
sTauW1 = [] 
sTauW2 = []
sTauW1_118  = [] 
sTauW2_118 = []
sTauW1_28 = [] 
sTauW2_28 = []
sTauW1_WISE = []
sTauW2_WISE = []
sTauK = []

# Errors from median sampled tau corrected for adopted gamma value
tauW1_corr_err = [] 
tauW1_corr_Err = []
tauW1_118_err= []
tauW1_118_Err= []
tauW1_28_err= []
tauW1_28_Err= []

tauW2_corr_err = []
tauW2_corr_Err = []
tauW2_118_err= []
tauW2_118_Err= []
tauW2_28_err= []
tauW2_28_Err= []

gammas = [] # Sampled gammas for each object
medGammaDistr = [] # Median sampled gammas for each object

# Modes of reported time lag distributions (used to construct split normal distribution)
modeW2 = [] 
modeW1 = []
modeK = []

# Get raw time lags
for i in range(len(tauW1)):
    if tauW1[i] != -999:
        # Removes Mandal's correction factor (1+z) ** -0.38 (time dilation + observational band shift) through division
        rTauW1.append(tauW1[i]/(1+zReported[i])**-0.38) 
        rTauW1_err.append(tauW1_err[i]/(1+zReported[i])**-0.38)
        rTauW1_Err.append(tauW1_Err[i]/(1+zReported[i])**-0.38)
    else:
        rTauW1.append(-999)
        rTauW1_Err.append(-999)
        rTauW1_err.append(-999)
    if tauW2[i] != -999:
        rTauW2.append(tauW2[i]/(1+zReported[i])**-0.38)
        rTauW2_err.append(tauW2_err[i]/(1+zReported[i])**-0.38)
        rTauW2_Err.append(tauW2_Err[i]/(1+zReported[i])**-0.38)
    else:
        rTauW2.append(-999)
        rTauW2_Err.append(-999)
        rTauW2_err.append(-999)

"""
Get CMB rest frame redshifts

Use galactic coordinates and heliocentric redshift
- Formula from Peterson et al. 2022
- Dipole direction and sun velocity relative to the CMB is from the Planck Collaboration

"""

# Parse RA and Dec from SDSS IDs
RA = []
Dec = []

for obj in SDSSID:
    RAH = float(obj[:2]) # RA hours
    RAM = float(obj[2:4]) # RA minutes
    if "+" in obj: 
        RAS = float(obj[4:obj.index("+")]) # RA seconds
        DecD = float(obj[obj.index("+")+1:obj.index("+")+3]) # Dec degrees
        DecM = float(obj[obj.index("+")+3:obj.index("+")+5]) # Dec minutes
        DecS = float(obj[obj.index("+")+5:]) # Dec seconds
    else:
        RAS = float(obj[4:obj.index("-")])
        DecD = float(obj[obj.index("-")+1:obj.index("-")+3])
        DecM = float(obj[obj.index("-")+3:obj.index("-")+5])
        DecS = float(obj[obj.index("-")+5:])
    
    RA.append(15*(RAH + (RAM/60) + (RAS/3600))) # RA in degrees: Multiplicative factor of 15 to convert 1 archour into 15 degrees
    
    # Dec in degrees
    if "-" in obj:
        Dec.append(-1*(DecD + (DecM/60) + (DecS/3600)))
    else:
        Dec.append(DecD + (DecM/60) + (DecS/3600))
        
# Convert RA and Dec to galactic coordinates
lat = [] # Galactic latitude
long = [] # Galactic longitude

for coord1, coord2 in zip(RA, Dec):
    # Convert to galactic coordinates with SciPy
    coord = SkyCoord(ra=float(coord1)*u.degree, dec = float(coord2)*u.degree, frame = "icrs")  
    galCoord = coord.galactic
    lat.append(galCoord.b.deg)
    long.append(galCoord.l.deg)

# Calculate CMB rest frame redshift
for i in range(len(zHel)):
    # Use the formula given by Peterson et al. 2022
    zCMB_WISE.append(
        (1+float(zHel[i]))/
        (1-((369.82/299792.458)*
            (np.sin(np.radians(lat[i]))*
                np.sin(np.radians(48.253))+
                np.cos(np.radians(lat[i]))*
                np.cos(np.radians(48.253))*
                np.cos(np.radians(long[i]-264.021)))))
        -1)

with open("WISE_RedshiftsCMB.txt", "w") as file:
    for i in zCMB_WISE:
        if i != zCMB_WISE[len(zCMB_WISE)-1]:
            file.write(f"{i}\n")
        else:
            file.write(f"{i}")

"""
Correct W-band time lags to K-band

Utilize power law correction with power law index gamma

Test Groups:
- Object-to-object gamma (obtained from MC sampling)
- Gamma = 1.18 (Minezaki et al.)
- Gamma = 2.8 (Barvainis)

Split-normal distributions are assumed for the WISE time lags to account for asymmetric errorbars

"""

# Obtain mode time lags from reported medians (assuming a split-normal distribution)

for i in range(len(rTauW1)):
    
    # W1 lags
    if rTauW1[i] != -999:
        if rTauW1_err[i] > rTauW1_Err[i]: #Left skewed distribution
            modeW1.append(rTauW1[i] - rTauW1_err[i] * scipy.stats.norm.ppf((rTauW1_err[i] + rTauW1_Err[i])/(4*rTauW1_err[i]))) #Calculate mode from median based on skew
        elif rTauW1_err[i] < rTauW1_Err[i]: #Right skewed distribution
            modeW1.append(rTauW1[i] - rTauW1_Err[i] * scipy.stats.norm.ppf(1-((rTauW1_err[i] + rTauW1_Err[i])/(4*rTauW1_Err[i]))))
        elif rTauW1_err[i] == rTauW1_Err[i]: #Symmetric distribution
            modeW1.append(rTauW1[i])
    else:
        modeW1.append(-999)             
    
    # W2 lags
    if rTauW2[i] != -999:
        if rTauW2_err[i] > rTauW2_Err[i]: #Left skewed distribution
            modeW2.append(rTauW2[i] - rTauW2_err[i] * scipy.stats.norm.ppf((rTauW2_err[i] + rTauW2_Err[i])/(4*rTauW2_err[i]))) #Calculate mode from median based on skew
        elif rTauW2_err[i] < rTauW2_Err[i]: #Right skewed distribution
            modeW2.append(rTauW2[i] - rTauW2_Err[i] * scipy.stats.norm.ppf(1-((rTauW2_err[i] + rTauW2_Err[i])/(4*rTauW2_Err[i]))))
        elif rTauW2_err[i] == rTauW2_Err[i]: #Symmetric distribution
            modeW2.append(rTauW2[i])
    else: 
        modeW2.append(-999)

# K lags
for i in range(len(tauK)):
    if tauK[i] != -999:
        if tauK_err[i] > tauK_Err[i]: #Left skewed distribution
            modeK.append(tauK[i] - tauK_err[i] * scipy.stats.norm.ppf((tauK_err[i] + tauK_Err[i])/(4*tauK_err[i]))) #Calculate mode from median based on skew
        elif tauK_err[i] < tauK_Err[i]: #Right skewed distribution
            modeK.append(tauK[i] - tauK_Err[i] * scipy.stats.norm.ppf(1-((tauK_err[i] + tauK_Err[i])/(4*tauK_Err[i]))))
        elif tauK_err[i] == tauK_Err[i]: #Symmetric distribution
            modeK.append(tauK[i])
    else:
        modeK.append(-999)     

# Sample randomly from split-normal distribution (assuming independent W1 and W2 uncertainties)
def sampleSplitNormal(mode, sigmaL, sigmaR): # Takes mode, LH uncertainty, and RH uncertainty
    pLeft = sigmaL/(sigmaL+sigmaR) # Define cumulative probability of LH side
    if random.uniform(0,1) <= pLeft: # Randomly choose left or right
        return -1*abs(random.gauss(0, sigmaL))+mode # Sample from LH Gaussian
    else:
        return abs(random.gauss(0,sigmaR))+mode # Sample from RH Gaussian

# Run n sampling realizations
for i in range(n):
    
    for j in range(len(modeW1)): # Iterate through all objects for each sampling realization
        
        if modeW1[j] != -999 and modeW2[j] != -999: # If the object has a W1 and W2 time lag, calculate gamma
        
            W2 = sampleSplitNormal(modeW2[j], rTauW2_err[j], rTauW2_Err[j]) # Sample a W2 lag
            W1 = sampleSplitNormal(modeW1[j],rTauW1_err[j],rTauW1_Err[j]) # Sample a W1 lag
            while W2 <= 0 or W1 <= 0: # If less than or equal to 0 (physically impossible), repeat sampling
                W2 = sampleSplitNormal(modeW2[j], rTauW2_err[j], rTauW2_Err[j])
                W1 = sampleSplitNormal(modeW1[j],rTauW1_err[j],rTauW1_Err[j])
            gam = np.log10(W2/W1)/np.log10(4.6/3.4) # Calculate the power law index from the lag and wavelength ratio (use WISE nominal wavelengths W1 = 3.4 and W2 = 4.6)
            
            # Correct each sample by the derived gamma and append to 2D list
            if len(sTauW1) != len(modeW1) and len(sTauW2)!= len(modeW2): # Append new sublist if this is the first MC realization for an object
                gammas.append([gam])
                
                # Apply gamma correction with sampled gamma and time dilation correction with zCMB
                sTauW2.append([(W2/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**gam)*(2.2**gam))/(4.6**gam)]) 
                sTauW1.append([(W1/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**gam)*(2.2**gam))/(3.4**gam)])
                # Apply fiducial universal gamma corrections for other test groups
                sTauW2_118.append([(W2/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**1.18)*(2.2**1.18))/(4.6**1.18)])
                sTauW1_118.append([(W1/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**1.18)*(2.2**1.18))/(3.4**1.18)])
                sTauW2_28.append([(W2/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**2.8)*(2.2**2.8))/(4.6**2.8)])
                sTauW1_28.append([(W1/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**2.8)*(2.2**2.8))/(3.4**2.8)])
                sTauW1_WISE.append([W1*(1+zReported[j])**-0.38]) # Equivalent to sampling Mandal's original time lags
                sTauW2_WISE.append([W2*(1+zReported[j])**-0.38])
                
            else: # Append to existing sublists it this is not first MC realization for an object
                gammas[j].append(gam)
                sTauW2[j].append((W2/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**gam)*(2.2**gam))/(4.6**gam))
                sTauW1[j].append((W1/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**gam)*(2.2**gam))/(3.4**gam))
                sTauW2_118[j].append((W2/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**1.18)*(2.2**1.18))/(4.6**1.18))
                sTauW1_118[j].append((W1/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**1.18)*(2.2**1.18))/(3.4**1.18))
                sTauW2_28[j].append((W2/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**2.8)*(2.2**2.8))/(4.6**2.8))
                sTauW1_28[j].append((W1/(1+zCMB_WISE[j])*((1+zCMB_WISE[j])**2.8)*(2.2**2.8))/(3.4**2.8)) 
                sTauW1_WISE[j].append(W1*(1+zReported[j])**-0.38)
                sTauW2_WISE[j].append(W2*(1+zReported[j])**-0.38)
                
        else:
            if len(sTauW1) != len(modeW1) and len(sTauW2)!= len(modeW2):
                gammas.append([-999]) 
                sTauW1.append([-999])
                sTauW2.append([-999])
                sTauW2_118.append([-999])
                sTauW1_118.append([-999])
                sTauW2_28.append([-999])
                sTauW1_28.append([-999])
                sTauW1_WISE.append([-999])
                sTauW2_WISE.append([-999])

for i in range(n):
    for j in range(len(modeK)):
        if modeK[j] != -999:
            # Sample K lag for each object j 
            K = sampleSplitNormal(modeK[j], tauK_err[j], tauK_Err[j])
            while K <= 0:
                K = sampleSplitNormal(modeK[j], tauK_err[j], tauK_Err[j])
            if len(sTauK) != len(modeK):
                sTauK.append([K])
            else:
                sTauK[j].append(K)
        else:
            if len(sTauK) != len(modeK):
                sTauK.append([-999])

"""
Correct W-band time lags to K-band for objects without a W1 and W2 lag

Cannot determine unique gamma for these objects
- Instead, randomly sample from the distribution of gamma implied by other objects

"""

for i in gammas:
    if i[0] != -999:
        medGammaDistr.append(np.median(i)) # Append median gamma for each object to the gamma sampling distribution

# Correct objects by sampling from gamma distribution (assume gamma indepedent from measured single-band lag)

for i in range(len(modeW1)):
    
    # W1 lags
    if sTauW1[i][0] == -999 and modeW1[i] != -999: # Replace missing objects with empty sublists
        sTauW1[i] = []
        sTauW1_118[i] = []
        sTauW1_28[i] = []
        sTauW1_WISE[i] = []
        for j in range(n):
            W1 = sampleSplitNormal(modeW1[i],rTauW1_err[i],rTauW1_Err[i]) # Sample a W1 lag
            while W1 <= 0:
                W1 = sampleSplitNormal(modeW1[i],rTauW1_err[i],rTauW1_Err[i])
            gamma = random.choice(medGammaDistr) # Sample a gamma
            sTauW1[i].append((W1/(1+zCMB_WISE[i])*((1+zCMB_WISE[i])**gamma)*(2.2**gamma))/(3.4**gamma))
            sTauW1_118[i].append((W1/(1+zCMB_WISE[i])*((1+zCMB_WISE[i])**1.18)*(2.2**1.18))/(3.4**1.18))
            sTauW1_28[i].append((W1/(1+zCMB_WISE[i])*((1+zCMB_WISE[i])**2.8)*(2.2**2.8))/(3.4**2.8))
            sTauW1_WISE[i].append(W1*(1+zReported[i])**-0.38)
            
    # W2 lags    
    if sTauW2[i][0] == -999 and modeW2[i] != -999:
        sTauW2[i] = []
        sTauW2_118[i] = []
        sTauW2_28[i] = []
        sTauW2_WISE[i] = []
        for j in range(n):
            W2 = sampleSplitNormal(modeW2[i],rTauW2_err[i],rTauW2_Err[i])
            while W2 <= 0:
                W2 = sampleSplitNormal(modeW2[i],rTauW2_err[i],rTauW2_Err[i])
            gamma = random.choice(medGammaDistr)              
            sTauW2[i].append((W2/(1+zCMB_WISE[i])*((1+zCMB_WISE[i])**gamma*(2.2**gamma/(4.6**gamma)))))
            sTauW2_118[i].append((W2/(1+zCMB_WISE[i])*((1+zCMB_WISE[i])**1.18)*(2.2**1.18))/(4.6**1.18))
            sTauW2_28[i].append((W2/(1+zCMB_WISE[i])*((1+zCMB_WISE[i])**2.8)*(2.2**2.8))/(4.6**2.8))
            sTauW2_WISE[i].append(W2/(1+zCMB_WISE[i])*(1+zReported[i])**-0.38)        

# Overall median gamma
medGam = np.median(medGammaDistr)
print(f"Population level median gamma: {medGam} +- {np.std(medGammaDistr)} (1-sigma)")

"""
Correct the reported errors by gamma

Multiplicative error scaling: Errors scale with central value
- Correct errors with the same central value scale factor

"""

# Correct errors from median by the object-to-object gammas 
for i in range(len(gammas)):
    
    # If W1 and W2 lags exist for the object (use median sampled gamma for that object)
    if gammas[i][0] != -999:
        gam = np.median(gammas[i])
        tauW2_corr_Err.append((rTauW2_Err[i]*((1+zCMB_WISE[i])**gam)*(2.2**gam))/(4.6**gam))
        tauW2_corr_err.append((rTauW2_err[i]*((1+zCMB_WISE[i])**gam)*(2.2**gam))/(4.6**gam))
        tauW1_corr_Err.append((rTauW1_Err[i]*((1+zCMB_WISE[i])**gam)*(2.2**gam))/(3.4**gam))
        tauW1_corr_err.append((rTauW1_err[i]*((1+zCMB_WISE[i])**gam)*(2.2**gam))/(3.4**gam))
        
    # If only W2 lag exists (use overall median gamma)
    elif gammas[i][0] == -999 and rTauW2_Err[i] != -999:
        tauW2_corr_Err.append((rTauW2_Err[i]*((1+zCMB_WISE[i])**medGam)*(2.2**medGam))/(4.6**medGam))
        tauW2_corr_err.append((rTauW2_err[i]*((1+zCMB_WISE[i])**medGam)*(2.2**medGam))/(4.6**medGam))
        tauW1_corr_Err.append(-999)
        tauW1_corr_err.append(-999)
        
    # If only W1 lag exists (use overall median gamma)
    elif gammas[i][0] == -999 and rTauW1_Err[i] != -999:
        tauW1_corr_Err.append((rTauW1_Err[i]*((1+zCMB_WISE[i])**medGam)*(2.2**medGam))/(3.4**medGam))
        tauW1_corr_err.append((rTauW1_err[i]*((1+zCMB_WISE[i])**medGam)*(2.2**medGam))/(3.4**medGam))
        tauW2_corr_Err.append(-999)
        tauW2_corr_err.append(-999)
        
    # If no lags exist
    else:
        tauW1_corr_Err.append(-999)
        tauW1_corr_err.append(-999)
        tauW2_corr_Err.append(-999)
        tauW2_corr_err.append(-999)

# Correct errors from median by the test group fixed gammas
for i in range(len(modeW1)):
    
    # W1 lags
    if rTauW1_Err[i] != -999:
        tauW1_118_Err.append((rTauW1_Err[i]*((1+zCMB_WISE[i])**1.18)*(2.2**1.18))/(3.4**1.18))
        tauW1_118_err.append((rTauW1_err[i]*((1+zCMB_WISE[i])**1.18)*(2.2**1.18))/(3.4**1.18))
        tauW1_28_Err.append((rTauW1_Err[i]*((1+zCMB_WISE[i])**2.8)*(2.2**2.8))/(3.4**2.8))
        tauW1_28_err.append((rTauW1_err[i]*((1+zCMB_WISE[i])**2.8)*(2.2**2.8))/(3.4**2.8))
    else:
        tauW1_118_Err.append(-999)
        tauW1_118_err.append(-999)
        tauW1_28_Err.append(-999)
        tauW1_28_err.append(-999) 
    
    # W2 lags
    if rTauW2_Err[i] != -999:
        tauW2_118_Err.append((rTauW2_Err[i]*((1+zCMB_WISE[i])**1.18)*(2.2**1.18))/(4.6**1.18))
        tauW2_118_err.append((rTauW2_err[i]*((1+zCMB_WISE[i])**1.18)*(2.2**1.18))/(4.6**1.18))
        tauW2_28_Err.append((rTauW2_Err[i]*((1+zCMB_WISE[i])**2.8)*(2.2**2.8))/(4.6**2.8))
        tauW2_28_err.append((rTauW2_err[i]*((1+zCMB_WISE[i])**2.8)*(2.2**2.8))/(4.6**2.8))
    else:
        tauW2_118_Err.append(-999)
        tauW2_118_err.append(-999)
        tauW2_28_Err.append(-999)
        tauW2_28_err.append(-999)

"""
Write corrected time lags to a .txt file

"""

with open("SampledTauW1.txt", "w") as file: 
    file.write(f"Median Corrected Time Lag|Lower Uncertainty|Upper Uncertainty|Gamma")
    for i in range(len(sTauW1)):
        file.write(f"{np.median(sTauW1[i])}|{tauW1_corr_err[i]}|{tauW1_corr_Err[i]}|{np.median(gammas[i])} \n")

with open("SampledTauW2.txt", "w") as file: 
    file.write(f"Median Corrected Time Lag|Lower Uncertainty|Upper Uncertainty|Gamma")
    for i in range(len(sTauW2)):
        file.write(f"{np.median(sTauW2[i])}|{tauW2_corr_err[i]}|{tauW2_corr_Err[i]}|{np.median(gammas[i])} \n")

"""
Get flux from reported luminosities in the WISE sample

Calculate the luminosity distance given the adopted cosmology
Calculate the flux given the reported luminosity and the luminosity distance

"""

# Get luminosity distances using the adopted cosmology of Shen et al. and Liu et al. (which report luminosities)
def integrand(z): # Integrand for luminosity distance integral
    return 1/np.sqrt((0.3*(1+z)**3+0.7))

for i in range(len(zHel)):
    integral = quad(integrand, 0, zHel[i]) # Integrate with respect to z, with each object's redshift as the upper bound
    DL_given.append(3.0856776e24*(1+zHel[i])*299792.458/70*integral[0]) # Large conversion factor in front to convert Mpc to cm (to report flux in erg/s/Hz/cm**2)

# Get fluxes from luminosity distances 
for i in range(len(logL)):
    
    # Correct back to spectral luminosity in erg/s/Hz from SED (use reported measurement wavelength for each redshift group)
    if zHel[i] < 0.7: 
        L = 10**logL[i]/(299792458/(5100*1e-10))
    elif zHel[i] < 1.9:
        L = 10**logL[i]/(299792458/(3000*1e-10))
    else:
        L = 10**logL[i]/(299792458/(1350*1e-10))
        
    L_err = 10**np.log10(L)-10**(np.log10(L)-logL_err[i])
    L_Err = 10**(np.log10(L)+logL_err[i])-10**np.log10(L)
    
    flux.append(L/(4*np.pi*DL_given[i]**2)) # Calculate flux for WISE objects: erg/s/Hz/cm**2
    flux_err.append(L_err/(4*np.pi*DL_given[i]**2))
    flux_Err.append(L_Err/(4*np.pi*DL_given[i]**2))
    
"""
Get distances given time lags

Uses Minezaki et al. / Yoshii et al.'s formula to calculate distance with the g calibration constant
Use a Monte Carlo method that calculates n realizations of distance from draws from flux and time lag distributions

"""

# Sample fluxes randomly from log-normal distribution (WISE fluxes reported with symmetric uncertainties in log space)
def sampleLogNormal(mu, sigma): # Sample flux values from lognormal distribution
    mu = mu/np.log10(np.e) # Convert log10 to ln for use in lognormvariate (base change theorem)
    sigma = sigma/np.log10(np.e)
    randF = random.lognormvariate(mu, sigma) # Sample random flux from log-normal distribution
    while randF <= 0: # Resample if flux is less than or equal to 0 (unphysical)
        randF = random.lognormvariate(mu, sigma)
    return randF

# Store indices of sampled g calibration constant
gIndex = []

# Run n MC realizations (sampling from g, alphaS, alphaL, flux, and time lags) to determine luminosity distance distributions

# W1 lags
for i in range(len(sTauW1)): # Iterate through all W1 objects
    if sTauW1[i][0] != -999:
        # Append empty sublist for new object
        DL_tauW1.append([])
        DL_tauW1_118.append([])
        DL_tauW1_28.append([])
        DL_tauW1_WISE.append([])
        
        # n MC iterations
        for j in range(n):
            
            if alphSampTog:
                ind = np.random.choice(len(gDistr)) # Choose random index from distribution of g calibration constants
                
                if j == 0:
                    gIndex.append([ind])
                else:
                    gIndex[len(gIndex)-1].append(ind)
                
                # Use the random index to sample the g, alphaS, and alphaL values that correspond to each other
                alphaS = alphDistrS[ind]
                alphaL = alphDistrL[ind]
                g = gDistr[ind]
            
            else:
                alphaS = alpha
                alphaL = alpha
            
            # Apply Minezaki et al. / Yoshii et al.'s formula to calculate DL
            
            # K-correction to convert the observed X-band flux into V-band
            if zCMB_WISE[i] < 0.7:
                kCorr = -2.5 * alphaL * np.log10(5500/5100) #Frq X / Frq V = Wavelength V / Wavelength X
            elif zCMB_WISE[i] < 1.9:
                kCorr = -2.5 * alphaL * np.log10(5500/3000)
            else:
                kCorr = -2.5 * (alphaS * np.log10(2200/1350) + alphaL * np.log10(5500/2200))
                
            # V-band magnitude (sample from flux distribution)
            mV = -2.5 * np.log10(sampleLogNormal(np.log10(flux[i]),logL_err[i])/(3640*1e-23)) # Assume a zero point magnitude of 3640 Jy, convert to erg/s/cm^2/Hz by factor of 10^-23
            
            # Calculate DL from the sampled time lags, g calibration constant, V-band magnitude, and K-correction
            DL_tauW1[i].append(propW1 * sTauW1[i][j] * 10**(0.2*(mV-kCorr-25+g)))
            DL_tauW1_118[i].append(propW1_118 * sTauW1_118[i][j] * 10**(0.2*(mV-kCorr-25+g)))
            DL_tauW1_28[i].append(propW1_28 * sTauW1_28[i][j] * 10**(0.2*(mV-kCorr-25+g)))
            DL_tauW1_WISE[i].append(propW1_WISE * sTauW1_WISE[i][j]* 10**(0.2*(mV-kCorr-25+g)))
    else:
        DL_tauW1.append([-999])
        DL_tauW1_118.append([-999])
        DL_tauW1_28.append([-999])
        DL_tauW1_WISE.append([-999])

ind0 = -1 # Index for iterating through gIndex (necessary such that W2 distances of the same object use the same sampled parameters as W1)

# W2 lags
for i in range(len(sTauW2)):
    if sTauW2[i][0] != -999:
        ind0 += 1 # Iterate through gIndex

        DL_tauW2.append([])
        DL_tauW2_118.append([])
        DL_tauW2_28.append([])
        DL_tauW2_WISE.append([])
        
        # n MC iterations
        for j in range(n):
            
            if alphSampTog:
                alphaS = alphDistrS[gIndex[ind0][j]]
                alphaL = alphDistrL[gIndex[ind0][j]]
                g = gDistr[gIndex[ind0][j]]
            else:
                alphaS = alpha
                alphaL = alpha
                
            if zCMB_WISE[i] < 0.7:
                kCorr = -2.5 * alphaL * np.log10(5500/5100)
            elif zCMB_WISE[i] < 1.9:
                kCorr = -2.5 * alphaL * np.log10(5500/3000)
            else:
                kCorr = -2.5 * (alphaS * np.log10(2200/1350) + alphaL * np.log10(5500/2200))
                
            mV = -2.5 * np.log10(sampleLogNormal(np.log10(flux[i]),logL_err[i])/(3640*1e-23)) 
            
            DL_tauW2[i].append(propW2 * sTauW2[i][j] * 10**(0.2*(mV-kCorr-25+g)))
            DL_tauW2_118[i].append(propW2_118 * sTauW2_118[i][j] * 10**(0.2*(mV-kCorr-25+g)))
            DL_tauW2_28[i].append(propW2_28 * sTauW2_28[i][j] * 10**(0.2*(mV-kCorr-25+g)))
            DL_tauW2_WISE[i].append(propW2_WISE * sTauW2_WISE[i][j]* 10**(0.2*(mV-kCorr-25+g)))
    else:
        DL_tauW2.append([-999])
        DL_tauW2_118.append([-999])
        DL_tauW2_28.append([-999])
        DL_tauW2_WISE.append([-999])

# K lags
for i in range(len(sTauK)):
    if sTauK[i][0] != -999 and flux_mag[i] != -999:
        DL_tauK.append([])
        
        # Flux reported by MAGNUM is asymmetric and in linear space: Sample from a split-normal distribution
        
        # Get mode flux from median for split-normal sampling
        if flux_mag_err[i] > flux_mag_Err[i]: # Left skewed distribution
            modeFlux = flux_mag[i] - flux_mag_err[i] * scipy.stats.norm.ppf((flux_mag_err[i] + flux_mag_Err[i])/(4*flux_mag_err[i]))
        elif flux_mag_err[i] < flux_mag_Err[i]: # Right skewed distribution
            modeFlux = flux_mag[i] - flux_mag_Err[i] * scipy.stats.norm.ppf(1-((flux_mag_err[i] + flux_mag_Err[i])/(4*flux_mag_Err[i])))
        elif flux_mag_err[i] == flux_mag_Err[i]: # Symmetric distribution
            modeFlux = flux_mag[i]
        
        # n MC iterations
        for j in range(n):
            
            if alphSampTog:
                ind = np.random.choice(len(gDistr))
                alphaS = alphDistrS[ind]
                alphaL = alphDistrL[ind]
                g = gDistr[ind]
            else: 
                alphaS = alpha
                alphaL = alpha
            
            # K-correction to convert the observed X-band flux into V-band            
            if obsBand[i] == "R": # Observer rest frame R band
                if 6283/(1+zCMB_mag[i]) >= 2200: 
                    kCorr = -2.5 * np.log10(1+zCMB_mag[i]) - 2.5 * alphaL * np.log10(5500 * (1+zCMB_mag[i]) / 6283)
                else:
                    kCorr = -2.5 * np.log10(1+zCMB_mag[i]) - 2.5 * (alphaL * np.log10(2200 * (1+zCMB_mag[i]) / 6283) + alphaS * np.log10(5500 / 2200))
            elif obsBand[i] == "I": # Observer rest frame I band
                if 7774/(1+zCMB_mag[i]) >= 2200:
                    kCorr = -2.5 * np.log10(1+zCMB_mag[i]) - 2.5 * alphaL * np.log10(5500 * (1+zCMB_mag[i]) / 7774)
                else:
                    kCorr = -2.5 * np.log10(1+zCMB_mag[i]) - 2.5 * (alphaL * np.log10(2200 * (1+zCMB_mag[i]) / 7774) + alphaS * np.log10(5500 / 2200))
            else:
                kCorr = -2.5 * np.log10(1+zCMB_mag[i]) - 2.5 * alphaL * np.log10(5500 * (1+zCMB_mag[i]) / 5500)
                
            f = sampleSplitNormal(modeFlux,flux_mag_err[i], flux_mag_Err[i]) * 1e-26 # Sample flux
            while f <= 0: # Resample if less or equal to 0 (unphysical)
                f = sampleSplitNormal(modeFlux,flux_mag_err[i], flux_mag_Err[i]) * 1e-26
                
            mV = -2.5 * np.log10(f/(3640*1e-23)) 
            
            DL_tauK[i].append(sTauK[i][j] * 10**(0.2*(mV-kCorr-25+g)))
    else:
        DL_tauK.append([-999])

"""
Write luminosity distances to a .txt file

- Columns: object
- Rows: sample

"""
# W1 lags
with open("LuminosityDistances_W1.txt", "w") as file:
    for i in range(n):
        writeDL = ""
        for j in range(len(DL_tauW1)):
            try: 
                if j != 0:
                    writeDL += f"|{DL_tauW1[j][i]}"
                else:
                    writeDL += f"{DL_tauW1[j][i]}"
            except IndexError: # Catches objects with -999 for luminosity distance
                if j != 0:
                    writeDL += f"|{-999}"
                else:
                    writeDL += f"{-999}"
        if i != n-1:
            writeDL += "\n"
        file.write(writeDL)

with open("LuminosityDistances_W1_118.txt", "w") as file:
    for i in range(n):
        writeDL = ""
        for j in range(len(DL_tauW1_118)):
            try: 
                if j != 0:
                    writeDL += f"|{DL_tauW1_118[j][i]}"
                else:
                    writeDL += f"{DL_tauW1_118[j][i]}"
            except IndexError: 
                if j != 0:
                    writeDL += f"|{-999}"
                else:
                    writeDL += f"{-999}"
        if i != n-1:
            writeDL += "\n"
        file.write(writeDL)
        
with open("LuminosityDistances_W1_28.txt", "w") as file:
    for i in range(n):
        writeDL = ""
        for j in range(len(DL_tauW1_28)):
            try: 
                if j != 0:
                    writeDL += f"|{DL_tauW1_28[j][i]}"
                else:
                    writeDL += f"{DL_tauW1_28[j][i]}"
            except IndexError: 
                if j != 0:
                    writeDL += f"|{-999}"
                else:
                    writeDL += f"{-999}"
        if i != n-1:
            writeDL += "\n"
        file.write(writeDL)
        
with open("LuminosityDistances_W1_WISE.txt", "w") as file:
    for i in range(n):
        writeDL = ""
        for j in range(len(DL_tauW1_WISE)):
            try: 
                if j != 0:
                    writeDL += f"|{DL_tauW1_WISE[j][i]}"
                else:
                    writeDL += f"{DL_tauW1_WISE[j][i]}"
            except IndexError: 
                if j != 0:
                    writeDL += f"|{-999}"
                else:
                    writeDL += f"{-999}"
        if i != n-1:
            writeDL += "\n"
        file.write(writeDL)

# W2 lags
with open("LuminosityDistances_W2.txt", "w") as file:
    for i in range(n):
        writeDL = ""
        for j in range(len(DL_tauW2)):
            try: 
                if j != 0:
                    writeDL += f"|{DL_tauW2[j][i]}"
                else:
                    writeDL += f"{DL_tauW2[j][i]}"
            except IndexError: 
                if j != 0:
                    writeDL += f"|{-999}"
                else:
                    writeDL += f"{-999}"
        if i != n-1:
            writeDL += "\n"
        file.write(writeDL)

with open("LuminosityDistances_W2_118.txt", "w") as file:
    for i in range(n):
        writeDL = ""
        for j in range(len(DL_tauW2_118)):
            try: 
                if j != 0:
                    writeDL += f"|{DL_tauW2_118[j][i]}"
                else:
                    writeDL += f"{DL_tauW2_118[j][i]}"
            except IndexError: 
                if j != 0:
                    writeDL += f"|{-999}"
                else:
                    writeDL += f"{-999}"
        if i != n-1:
            writeDL += "\n"
        file.write(writeDL)
        
with open("LuminosityDistances_W2_28.txt", "w") as file:
    for i in range(n):
        writeDL = ""
        for j in range(len(DL_tauW2_28)):
            try: 
                if j != 0:
                    writeDL += f"|{DL_tauW2_28[j][i]}"
                else:
                    writeDL += f"{DL_tauW2_28[j][i]}"
            except IndexError: 
                if j != 0:
                    writeDL += f"|{-999}"
                else:
                    writeDL += f"{-999}"
        if i != n-1:
            writeDL += "\n"
        file.write(writeDL)
        
with open("LuminosityDistances_W2_WISE.txt", "w") as file:
    for i in range(n):
        writeDL = ""
        for j in range(len(DL_tauW2_WISE)):
            try: 
                if j != 0:
                    writeDL += f"|{DL_tauW2_WISE[j][i]}"
                else:
                    writeDL += f"{DL_tauW2_WISE[j][i]}"
            except IndexError: 
                if j != 0:
                    writeDL += f"|{-999}"
                else:
                    writeDL += f"{-999}"
        if i != n-1:
            writeDL += "\n"
        file.write(writeDL)
        
# K lags
with open("LuminosityDistances_K.txt", "w") as file:
    for i in range(n):
        writeDL = ""
        for j in range(len(DL_tauK)):
            try: 
                if j != 0:
                    writeDL += f"|{DL_tauK[j][i]}"
                else:
                    writeDL += f"{DL_tauK[j][i]}"
            except IndexError: 
                if j != 0:
                    writeDL += f"|{-999}"
                else:
                    writeDL += f"{-999}"
        if i != n-1:
            writeDL += "\n"
        file.write(writeDL)