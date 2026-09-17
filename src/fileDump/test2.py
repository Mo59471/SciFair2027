#MC sampler that samples corrected time lags using a power law index determined from comparison of W1 and W2 time lags
#Samples from the time lag error margins to produce gammas and corrected time lags
#Produces 10,000, sampled, corrected time lags for each object

from astropy.coordinates import SkyCoord
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid
from astropy import units as u
import numpy as np
import random
import scipy
import statistics
import math
import matplotlib.pyplot as plt
import mpfit

#Paramaters

objID, SDSSID, zReported, zHel, logL, logL_err, tauW1, tauW1_err, tauW1_Err, tauW2, tauW2_err, tauW2_Err =np.genfromtxt(
    "WISEData.csv", delimiter=",",usecols = (0,2,3,5,16,17,26,27,28,39,40,41), 
    skip_header=1, unpack = True, missing_values=("","#NUM!"), filling_values=-999, dtype=None, comments= None) #-999 indicates no value

zCMB = []
zFlowCorr = []

#Cosmology Parameters
H0_W1 = []
H0_W1_118 = []
H0_W1_28 = []
H0_W2 = []
H0_W2_118 = []
H0_W2_28 = []

omg_0_W1 = []
omg_0_W1_118 = []
omg_0_W1_28 = []
omg_0_W2 = []
omg_0_W2_118 = []
omg_0_W2_28 = []

#Luminosity distances
DL_given = []
DL_tauW1 = []
DL_tauW2 = []
DL_tauW1_118 = []
DL_tauW2_118 = []
DL_tauW1_28 = []
DL_tauW2_28 = []

DL_tauW1_err = []
DL_tauW1_Err = []
DL_tauW1_118_err = []
DL_tauW1_118_Err = []
DL_tauW1_28_err = []
DL_tauW1_28_Err = []

DL_tauW2_err = []
DL_tauW2_Err = []
DL_tauW2_118_err = []
DL_tauW2_118_Err = []
DL_tauW2_28_err = []
DL_tauW2_28_Err = []

flux = []
flux_err = []
flux_Err = []

#Lists of raw time lags (Mandal's correction with gamma = 0.62 removed)
rTauW1 = []
rTauW1_err = []
rTauW1_Err = []
rTauW2 = []
rTauW2_err = []
rTauW2_Err = []

#Lists of sampled time lags
sTauW1 = [] #Lists of sampled time lags for each object (corrected with gamma)
sTauW2 = []
sTauW1_118  = [] #Sampled time lags corrected with Minezaki et al.'s gamma = 1.18
sTauW2_118 = []
sTauW1_28 = [] #Sampled time lags corrected with Barvainis' gamma = 2.8
sTauW2_28 = []

#Lists of corrected errors for median sampled time lags
tauW1_corr_err = [] #Errors scaled by each object's individually determined gamma (correspond to sTauW1)
tauW1_corr_Err = []
tauW1_118_err= []
tauW1_118_Err= []
tauW1_28_err= []
tauW1_28_Err= []

tauW2_corr_err = [] #Errors scaled by each object's individually determined gamma (correspond to sTauW2)
tauW2_corr_Err = []
tauW2_118_err= []
tauW2_118_Err= []
tauW2_28_err= []
tauW2_28_Err= []

gammas = [] #Lists of sampled gammas for each object
medGammaDistr = [] #List of median sampled gammas for each object

modeW2 = [] #Convert from the median time lag value (reported by Mandal) to modes for split-normal sampling
modeW1 = []

#Get raw time lags
for i in range(len(tauW1)):
    if tauW1[i] != -999:
        rTauW1.append(tauW1[i]/(1+zReported[i])**0.62)
        rTauW1_err.append(tauW1_err[i]/(1+zReported[i])**0.62)
        rTauW1_Err.append(tauW1_Err[i]/(1+zReported[i])**0.62)
    else:
        rTauW1.append(-999)
        rTauW1_Err.append(-999)
        rTauW1_err.append(-999)
    if tauW2[i] != -999:
        rTauW2.append(tauW2[i]/(1+zReported[i])**0.62)
        rTauW2_err.append(tauW2_err[i]/(1+zReported[i])**0.62)
        rTauW2_Err.append(tauW2_Err[i]/(1+zReported[i])**0.62)
    else:
        rTauW2.append(-999)
        rTauW2_Err.append(-999)
        rTauW2_err.append(-999)

#Extract RA and Dec from SDSS IDs
RA = []
Dec = []

for obj in SDSSID:
    RAH = float(obj[:2])
    RAM = float(obj[2:4])
    if "+" in obj:
        RAS = float(obj[4:obj.index("+")])
        DecD = float(obj[obj.index("+")+1:obj.index("+")+3])
        DecM = float(obj[obj.index("+")+3:obj.index("+")+5])
        DecS = float(obj[obj.index("+")+5:])
    else:
        RAS = float(obj[4:obj.index("-")])
        DecD = float(obj[obj.index("-")+1:obj.index("-")+3])
        DecM = float(obj[obj.index("-")+3:obj.index("-")+5])
        DecS = float(obj[obj.index("-")+5:])
    
    RA.append(15*(RAH + (RAM/60) + (RAS/3600)))
    if "-" in obj:
        Dec.append(-1*(DecD + (DecM/60) + (DecS/3600)))
    else:
        Dec.append(DecD + (DecM/60) + (DecS/3600))
        
#Calculate CMB rest frame redshift based on coordinates and zHel
lat = [] #Galactic latitude and longitude 
long = []

for coord1, coord2 in zip(RA,Dec):
    coord = SkyCoord(ra=float(coord1)*u.degree, dec = float(coord2)*u.degree, frame = "icrs") #Convert to galactic coordinates
    galCoord = coord.galactic
    lat.append(galCoord.b.deg)
    long.append(galCoord.l.deg)
for i in range(len(zHel)):
    #Uses the formula given by Peterson et al. 2022: Dipole direction and sun velocity relative to CMB rest frame is from the Planck Collab
    zCMB.append(
        (1+float(zHel[i]))/
        (1-((369.82/299792.458)*
            (np.sin(np.radians(lat[i]))*
                np.sin(np.radians(48.253))+
                np.cos(np.radians(lat[i]))*
                np.cos(np.radians(48.253))*
                np.cos(np.radians(long[i]-264.021)))))
        -1)

#Obtain mode time lags from medians assuming a split-normal distribution
for i in range(len(rTauW1)):
    if rTauW1[i] != -999:
        if rTauW1_err[i] > rTauW1_Err[i]: #Left skewed
            modeW1.append(rTauW1[i] - rTauW1_err[i] * scipy.stats.norm.ppf((rTauW1_err[i] + rTauW1_Err[i])/(4*rTauW1_err[i]))) #Calculate mode from median based on skew
        elif rTauW1_err[i] < rTauW1_Err[i]: #Right skewed
            modeW1.append(rTauW1[i] - rTauW1_Err[i] * scipy.stats.norm.ppf(1-((rTauW1_err[i] + rTauW1_Err[i])/(4*rTauW1_Err[i]))))
        elif rTauW1_err[i] == rTauW1_Err[i]: #Symmetric
            modeW1.append(rTauW1[i])
    else:
        modeW1.append(-999)
        
    if rTauW2[i] != -999:
        if rTauW2_err[i] > rTauW2_Err[i]: #Left skewed
            modeW2.append(rTauW2[i] - rTauW2_err[i] * scipy.stats.norm.ppf((rTauW2_err[i] + rTauW2_Err[i])/(4*rTauW2_err[i]))) #Calculate mode from median based on skew
        elif rTauW2_err[i] < rTauW2_Err[i]: #Right skewed
            modeW2.append(rTauW2[i] - rTauW2_Err[i] * scipy.stats.norm.ppf(1-((rTauW2_err[i] + rTauW2_Err[i])/(4*rTauW2_Err[i]))))
        elif rTauW2_err[i] == rTauW2_Err[i]: #Symmetric
            modeW2.append(rTauW2[i])
    else: 
        modeW2.append(-999)

#Sample randomly from the split normal distributions of the time lags
#Here, the uncertainties between W1 and W2 are assumed to be independent which is not strictly true, but approximately so
def sampleSplitNormal(mode, sigmaL, sigmaR):
    pLeft = sigmaL/(sigmaL+sigmaR) #Define cumulative probability of LH side
    if random.uniform(0,1) <= pLeft: #random float between 0-1, inclusive; if less than prob of LH side -> falls in LH side
        return -1*abs(random.gauss(0, sigmaL))+mode #If on left hand side, sample from LH Gaussian
    else:
        return abs(random.gauss(0,sigmaR))+mode #If on right hand side, sample from RH Gaussian

#Run 10,000 sampling realizations to get 10,000 corrected time lags for each object
for i in range(10000):
    for j in range(len(modeW1)):
        if modeW1[j] != -999 and modeW2[j] != -999:  
            #Sample W1 and W2 for each object j 
            W2 = sampleSplitNormal(modeW2[j], rTauW2_err[j], rTauW2_Err[j])
            W1 = sampleSplitNormal(modeW1[j],rTauW1_err[j],rTauW1_Err[j])
            while W2/W1 <= 0: #If less than 0, repeat sampling
                W2 = sampleSplitNormal(modeW2[j], rTauW2_err[j], rTauW2_Err[j])
                W1 = sampleSplitNormal(modeW1[j],rTauW1_err[j],rTauW1_Err[j])
            gam = math.log10(W2/W1)/math.log10(4.6/3.4) #Find the power law corresponding to the given ratio; uses WISE nominal wavelengths W1 = 3.4 and W2 = 4.6
            
            #Append the raw sampled value to lists of lists (each sublist = distribution of sampled values for each object)
            if len(sTauW1) != len(modeW1) and len(sTauW2)!= len(modeW2):
                gammas.append([gam])
                sTauW2.append([(W2*((1+zCMB[j])**gam)*(2.2**gam))/(4.6**gam)])
                sTauW1.append([(W1*((1+zCMB[j])**gam)*(2.2**gam))/(3.4**gam)])
                sTauW2_118.append([(W2*((1+zCMB[j])**1.18)*(2.2**1.18))/(4.6**1.18)])
                sTauW1_118.append([(W1*((1+zCMB[j])**1.18)*(2.2**1.18))/(3.4**1.18)])
                sTauW2_28.append([(W2*((1+zCMB[j])**2.8)*(2.2**2.8))/(4.6**2.8)])
                sTauW1_28.append([(W1*((1+zCMB[j])**2.8)*(2.2**2.8))/(3.4**2.8)])
            else:
                gammas[j].append(gam)
                sTauW2[j].append((W2*((1+zCMB[j])**gam)*(2.2**gam))/(4.6**gam))
                sTauW1[j].append((W1*((1+zCMB[j])**gam)*(2.2**gam))/(3.4**gam))
                sTauW2_118[j].append((W2*((1+zCMB[j])**1.18)*(2.2**1.18))/(4.6**1.18))
                sTauW1_118[j].append((W1*((1+zCMB[j])**1.18)*(2.2**1.18))/(3.4**1.18))
                sTauW2_28[j].append((W2*((1+zCMB[j])**2.8)*(2.2**2.8))/(4.6**2.8))
                sTauW1_28[j].append((W1*((1+zCMB[j])**2.8)*(2.2**2.8))/(3.4**2.8))                
        else:
            if len(sTauW1) != len(modeW1) and len(sTauW2)!= len(modeW2):
                gammas.append([-999]) 
                sTauW1.append([-999])
                sTauW2.append([-999])
                sTauW2_118.append([-999])
                sTauW1_118.append([-999])
                sTauW2_28.append([-999])
                sTauW1_28.append([-999])

#Calculate distribution of median gammas from which to sample from for objects without a W1 and W2 lag
for i in gammas:
    if i[0] != -999:
        medGammaDistr.append(statistics.median(i)) 

#Handle corrections for objects without a W1 and W2 lag (no independent distribution of gamma)
#This assumes that the sampled W1 or W2 lag is independent from gamma in isolation (i.e. gamma can't be determined without both)
for i in range(len(modeW1)):
    if sTauW1[i][0] == -999 and modeW1[i] != -999:
        sTauW1[i] = []
        sTauW1_118[i] = []
        sTauW1_28[i] = []
        for j in range(10000):
            W1 = sampleSplitNormal(modeW1[i],rTauW1_err[i],rTauW1_Err[i])
            while W1 <= 0:
                W1 = sampleSplitNormal(modeW1[i],rTauW1_err[i],rTauW1_Err[i])
            gamma = random.choice(medGammaDistr)       
            sTauW1[i].append((W1*((1+zCMB[i])**gamma)*(2.2**gamma))/(3.4**gamma))
            sTauW1_118[i].append((W1*((1+zCMB[i])**1.18)*(2.2**1.18))/(3.4**1.18))
            sTauW1_28[i].append((W1*((1+zCMB[i])**2.8)*(2.2**2.8))/(3.4**2.8))                
    if sTauW2[i][0] == -999 and modeW2[i] != -999:
        sTauW2[i] = []
        sTauW2_118[i] = []
        sTauW2_28[i] = []
        for j in range(10000):
            W2 = sampleSplitNormal(modeW2[i],rTauW2_err[i],rTauW2_Err[i])
            while W2 <= 0:
                W2 = sampleSplitNormal(modeW2[i],rTauW2_err[i],rTauW2_Err[i])
            gamma = random.choice(medGammaDistr)              
            sTauW2[i].append((W2*((1+zCMB[i])**gamma*(2.2**gamma/(4.6**gamma)))))
            sTauW2_118[i].append((W2*((1+zCMB[i])**1.18)*(2.2**1.18))/(4.6**1.18))
            sTauW2_28[i].append((W2*((1+zCMB[i])**2.8)*(2.2**2.8))/(4.6**2.8))       

medGam = statistics.median(medGammaDistr)

#Correct errors from median by gamma
for i in range(len(gammas)):
    #If W1 and W2 lags exist for the object
    if gammas[i][0] != -999:
        gam = statistics.median(gammas[i])
        tauW2_corr_Err.append((rTauW2_Err[i]*((1+zCMB[i])**gam)*(2.2**gam))/(4.6**gam))
        tauW2_corr_err.append((rTauW2_err[i]*((1+zCMB[i])**gam)*(2.2**gam))/(4.6**gam))
        tauW1_corr_Err.append((rTauW1_Err[i]*((1+zCMB[i])**gam)*(2.2**gam))/(3.4**gam))
        tauW1_corr_err.append((rTauW1_err[i]*((1+zCMB[i])**gam)*(2.2**gam))/(3.4**gam))
    #If only W2 lag exists
    elif gammas[i][0] == -999 and rTauW2_Err[i] != -999:
        tauW2_corr_Err.append((rTauW2_Err[i]*((1+zCMB[i])**medGam)*(2.2**medGam))/(4.6**medGam))
        tauW2_corr_err.append((rTauW2_err[i]*((1+zCMB[i])**medGam)*(2.2**medGam))/(4.6**medGam))
        tauW1_corr_Err.append(-999)
        tauW1_corr_err.append(-999)
    #If only W1 lag exists
    elif gammas[i][0] == -999 and rTauW1_Err[i] != -999:
        tauW1_corr_Err.append((rTauW1_Err[i]*((1+zCMB[i])**medGam)*(2.2**medGam))/(3.4**medGam))
        tauW1_corr_err.append((rTauW1_err[i]*((1+zCMB[i])**medGam)*(2.2**medGam))/(3.4**medGam))
        tauW2_corr_Err.append(-999)
        tauW2_corr_err.append(-999)
    #If no lags exist
    else:
        tauW1_corr_Err.append(-999)
        tauW1_corr_err.append(-999)
        tauW2_corr_Err.append(-999)
        tauW2_corr_err.append(-999)

#Correct errors from median by adopted fixed gammas
for i in range(len(modeW1)):
    if rTauW1_Err[i] != -999:
        tauW1_118_Err.append((rTauW1_Err[i]*((1+zCMB[i])**1.18)*(2.2**1.18))/(3.4**1.18))
        tauW1_118_err.append((rTauW1_err[i]*((1+zCMB[i])**1.18)*(2.2**1.18))/(3.4**1.18))
        tauW1_28_Err.append((rTauW1_Err[i]*((1+zCMB[i])**2.8)*(2.2**2.8))/(3.4**2.8))
        tauW1_28_err.append((rTauW1_err[i]*((1+zCMB[i])**2.8)*(2.2**2.8))/(3.4**2.8))
    else:
        tauW1_118_Err.append(-999)
        tauW1_118_err.append(-999)
        tauW1_28_Err.append(-999)
        tauW1_28_err.append(-999) 
    if rTauW2_Err[i] != -999:
        tauW2_118_Err.append((rTauW2_Err[i]*((1+zCMB[i])**1.18)*(2.2**1.18))/(4.6**1.18))
        tauW2_118_err.append((rTauW2_err[i]*((1+zCMB[i])**1.18)*(2.2**1.18))/(4.6**1.18))
        tauW2_28_Err.append((rTauW2_Err[i]*((1+zCMB[i])**2.8)*(2.2**2.8))/(4.6**2.8))
        tauW2_28_err.append((rTauW2_err[i]*((1+zCMB[i])**2.8)*(2.2**2.8))/(4.6**2.8))
    else:
        tauW2_118_Err.append(-999)
        tauW2_118_err.append(-999)
        tauW2_28_Err.append(-999)
        tauW2_28_err.append(-999)

writeTauW1 = []
writeTauW2 = []
for i in range(len(sTauW2)):
    writeTauW1.append(f"{statistics.median(sTauW1[i])}|{tauW1_corr_err[i]}|{tauW1_corr_Err[i]} \n")
    writeTauW2.append(f"{statistics.median(sTauW2[i])}|{tauW2_corr_err[i]}|{tauW2_corr_Err[i]} \n")
    
with open("SampledTauW1.txt", "w") as file:
    file.writelines(writeTauW1)
with open("SampledTauW2.txt", "w") as file:
    file.writelines(writeTauW2)
    
#Get luminosity distances using the adopted cosmology of Shen and Liu
def integrand(z):
    return 1/np.sqrt((0.3*(1+z)**3+0.7))

for i in range(len(zCMB)):
    integral = quad(integrand, 0, zCMB[i])
    DL_given.append(3.0856776e24*(1+zCMB[i])*299792.458/70*integral[0]) #Large conversion factor in front to convert Mpc to cm

#Get fluxes from luminosity distances 
for i in range(len(logL)):
    if zCMB[i] < 0.7: #Correct back to spectral luminosity in erg/s/Hz
        L = 10**logL[i]/(299792458/(5100*1e-10))
    elif zCMB[i] < 1.9:
        L = 10**logL[i]/(299792458/(3000*1e-10))
    else:
        L = 10**logL[i]/(299792458/(1350*1e-10))
    L_err = 10**math.log10(L)-10**(math.log10(L)-logL_err[i])
    L_Err = 10**(math.log10(L)+logL_err[i])-10**math.log10(L)
    flux.append(L/(4*math.pi*DL_given[i]**2)) #erg/s/Hz/cm^2
    flux_err.append(L_err/(4*math.pi*DL_given[i]**2))
    flux_Err.append(L_Err/(4*math.pi*DL_given[i]**2))
    
#Get distances from Minezaki's formula given time lags
#Temporarily, alphaUV is adopted as -0.5 (following Yoshii)
#Future implementation of an MC sampling from alphaUV distributions to come
def sampleLogNormal(mu, sigma): #Sample flux values from lognormal distribution
    mu = mu/math.log10(math.e) #Convert log10 mode and sigma to ln for lognormvariate (base change theorem)
    sigma = sigma/math.log10(math.e)
    randF = random.lognormvariate(mu, sigma)
    while randF <= 0:
        randF = random.lognormvariate(mu, sigma)
    return randF

#10,000 realizations of MC sampled time lags and fluxes to obtain DL
for i in range(len(sTauW1)):
    if sTauW1[i][0] != -999:
        if zCMB[i] < 0.7:
            kCorr = -2.5 * -0.5 * math.log10(5100/5500)
        elif zCMB[i] < 1.9:
            kCorr = -2.5 * -0.5 * math.log10(3000/5500)
        else:
            kCorr = -2.5 * -0.5 * math.log10(1350/5500)
        DL_tauW1.append([])
        DL_tauW1_118.append([])
        DL_tauW1_28.append([])
        for j in range(10000):
            mV = -2.5 * math.log10(sampleLogNormal(math.log10(flux[i]),logL_err[i])/(3640*1e-23)) #Assume a normalization of 3640 Jy, converted to erg/s/cm^2/Hz by factor of 10^-23
            DL_tauW1[i].append(sTauW1[i][j] * 10**(0.2*(mV-kCorr-25+10.60)))
            DL_tauW1_118[i].append(sTauW1_118[i][j] * 10**(0.2*(mV-kCorr-25+10.60)))
            DL_tauW1_28[i].append(sTauW1_28[i][j] * 10**(0.2*(mV-kCorr-25+10.60)))
    else:
        DL_tauW1.append([-999])
        DL_tauW1_118.append([-999])
        DL_tauW1_28.append([-999])
        
for i in range(len(sTauW2)):
    if sTauW2[i][0] != -999:
        if zCMB[i] < 0.7:
            kCorr = -2.5 * -0.5 * math.log10(5100/5500)
        elif zCMB[i] < 1.9:
            kCorr = -2.5 * -0.5 * math.log10(3000/5500)
        else:
            kCorr = -2.5 * -0.5 * math.log10(1350/5500)
        DL_tauW2.append([])
        DL_tauW2_118.append([])
        DL_tauW2_28.append([])
        for j in range(10000):
            mV = -2.5 * math.log10(sampleLogNormal(math.log10(flux[i]),logL_err[i])/(3640*1e-23)) #Assume a normalization of 3640 Jy, converted to erg/s/cm^2/Hz by factor of 10^-23
            DL_tauW2[i].append(sTauW2[i][j] * 10**(0.2*(mV-kCorr-25+10.60)))
            DL_tauW2_118[i].append(sTauW2_118[i][j] * 10**(0.2*(mV-kCorr-25+10.60)))
            DL_tauW2_28[i].append(sTauW2_28[i][j] * 10**(0.2*(mV-kCorr-25+10.60)))
    else:
        DL_tauW2.append([-999])
        DL_tauW2_118.append([-999])
        DL_tauW2_28.append([-999])
        
#Fit cosmology: bootstrapping + MC resampling
#Least squares residual minimization using MPFIT

#Cleaned MC sampled luminosity distance lists (-999 removed)
DL_tauW1_clean = []
DL_tauW2_clean =[]
DL_tauW1_118_clean = []
DL_tauW2_118_clean = []
DL_tauW1_28_clean = []
DL_tauW2_28_clean = []

zCMB_W1 = []
zCMB_W2 = []

for i in range(len(DL_tauW1)):
    if DL_tauW1[i][0] != -999:
        DL_tauW1_clean.append(DL_tauW1[i])
        DL_tauW1_118_clean.append(DL_tauW1_118[i])
        DL_tauW1_28_clean.append(DL_tauW1_28[i])
        zCMB_W1.append(zCMB[i])

for i in range(len(DL_tauW2)):
    if DL_tauW2[i][0] != -999:
        DL_tauW2_clean.append(DL_tauW2[i])
        DL_tauW2_118_clean.append(DL_tauW2_118[i])
        DL_tauW2_28_clean.append(DL_tauW2_28[i])
        zCMB_W2.append(zCMB[i])

#Fit cosmology with MPFIT
#Minimize residual DLs by calculating the luminosity distance integral

#Construct the cosmological model to be iterated upon (DL integral)
def cosmoModel(z, H0, omg_0, zStepNum):
    z = np.asarray(z, dtype= float)
    zSteps = np.linspace(0,np.max(z), zStepNum) #Gives an array of equally spaced z values up to the max for numeric integration
    
    denom = omg_0*(1+zSteps)**3 + (1-omg_0) #omg_DE in flat cosmology

    if (denom <= 0).any(): #if the denominator (inside sqrt) is negative or 0 -> DL undefined
        return np.full(z, np.nan) #Returns array of NaN values which MPFIT recognizes and indicating the fit is very wrong -> sever penalty
    
    #Calculatges a trapezoidal-cumulative integral up to the max redshift using the integrand (y(x)), and evenly spaced x step values (zSteps)
    runningIntegral = cumulative_trapezoid(1/np.sqrt(denom),zSteps,initial = 0.0)
    
    #There now is a cumulative integral up to the max redshift, with values for each zStep
    #Next, for each value in the given z, linearly interpolate between the z-step integral vlaues and approximate the integral's value at the given z
    integrals = np.interp(z,zSteps,runningIntegral) #gives np array of integrals at different z values
    
    return (1+z) * 299792.458/H0 *integrals

def cosmoFit(p, fjac = None, z = None, DL = None):
    z = np.asarray(z,dtype=float)
    DL = np.asarray(DL, dtype=float)
    
    model = cosmoModel(z,p[0],p[1],1000) 
    #status = 0
    return [0, DL-model] 

p0 = [70,0.3] #Starting parameter estimates: reduces computational time to list them close to true values, but does not bias result

"""
p[0]: Hubble Constant
p[1]: Mass Dimensionless Density Parameter
Notice the dark energy dimensionless density parameter is not fit for here:
assuming flat cosmology, omg_DE = 1-omg_0, redundant to fit
"""
for i in range(100):
    DL_W1_fit = []
    DL_W1_118_fit = []
    DL_W1_28_fit = []
    zCMB_W1_fit = []
    DL_W2_fit = []
    DL_W2_118_fit = []
    DL_W2_28_fit = []
    zCMB_W2_fit = []
    while len(DL_W1_fit) < len(DL_tauW1_clean):
        #Bootsraps: randomly selects objects (with replacement) until the boostrapped sample is as large as the original
        index = random.randint(0,len(DL_tauW1_clean)-1)
        #MC sampling: randomly selects a value from the random object's DL uncertainty distribution
        DL_W1_fit.append(random.choice(DL_tauW1_clean[index]))
        DL_W1_118_fit.append(random.choice(DL_tauW1_118_clean[index]))
        DL_W1_28_fit.append(random.choice(DL_tauW1_28_clean[index]))
        zCMB_W1_fit.append(zCMB_W1[index])
    
    while len(DL_W2_fit) < len(DL_tauW2_clean):
        index = random.randint(0, len(DL_tauW2_clean)-1)
        DL_W2_fit.append(random.choice(DL_tauW2_clean[index]))
        DL_W2_118_fit.append(random.choice(DL_tauW2_118_clean[index]))
        DL_W2_28_fit.append(random.choice(DL_tauW2_28_clean[index]))
        zCMB_W2_fit.append(zCMB_W2[index])
    
    #Set limits for valid cosmology parameters
    parinfo = [
        {
            "value": 70.0,
            "limited": [1, 0], #Limited only on bottom side, open on top side
            "limits": [1.0,0.0] #Limit H0 to a min of 1.0: prevent negative or 0 H0
        },
        {
            "value":0.3,
            "limited":[1,1], #Limited on both sides
            "limits":[0.0,1.0] #Limit to >= 0, <= 1 (in flat universe, max omega is 1)
        }
    ]
    
    mW1= mpfit.mpfit(
        cosmoFit, #repeatedly called and iterated
        p0, #Initial guesses
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_W1_fit, dtype=float), "DL":np.array(DL_W1_fit)}, #Dictionary: passes in x, y, and err into linefit
        quiet = 1 #Turn off console logs
        )

    mW1_118= mpfit.mpfit(
        cosmoFit, #repeatedly called and iterated
        p0, #Initial guesses
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_W1_fit, dtype=float), "DL":np.array(DL_W1_118_fit)}, #Dictionary: passes in x, y, and err into linefit
        quiet = 1 #Turn off console logs
        )

    mW1_28= mpfit.mpfit(
        cosmoFit, #repeatedly called and iterated
        p0, #Initial guesses
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_W1_fit, dtype=float), "DL":np.array(DL_W1_28_fit)}, #Dictionary: passes in x, y, and err into linefit
        quiet = 1 #Turn off console logs
        )

    mW2= mpfit.mpfit(
        cosmoFit, #repeatedly called and iterated
        p0, #Initial guesses
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_W2_fit, dtype=float), "DL":np.array(DL_W2_fit)}, #Dictionary: passes in x, y, and err into linefit
        quiet = 1
        )

    mW2_118= mpfit.mpfit(
        cosmoFit, #repeatedly called and iterated
        p0, #Initial guesses
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_W2_fit, dtype=float), "DL":np.array(DL_W2_118_fit)}, #Dictionary: passes in x, y, and err into linefit
        quiet = 1
        )
    
    mW2_28= mpfit.mpfit(
        cosmoFit, #repeatedly called and iterated
        p0, #Initial guesses
        parinfo=parinfo,
        functkw = {"z":np.array(zCMB_W2_fit, dtype=float), "DL":np.array(DL_W2_28_fit)}, #Dictionary: passes in x, y, and err into linefit
        quiet = 1
        )

    H0_W1.append(float(mW1.params[0]))
    H0_W1_118.append(float(mW1_118.params[0]))
    H0_W1_28.append(float(mW1_28.params[0]))

    H0_W2.append(float(mW2.params[0]))
    H0_W2_118.append(float(mW2_118.params[0]))
    H0_W2_28.append(float(mW2_28.params[0]))

    omg_0_W1.append(float(mW1.params[1]))
    omg_0_W1_118.append(float(mW1_118.params[1]))
    omg_0_W1_28.append(float(mW1_28.params[1]))

    omg_0_W2.append(float(mW2.params[1]))
    omg_0_W2_118.append(float(mW2_118.params[1]))
    omg_0_W2_28.append(float(mW2_28.params[1]))

print(f"H0 W1,W2: {statistics.median(H0_W1)}, {statistics.median(H0_W2)}")
print(f"H0_118 W1,W2: {statistics.median(H0_W1_118)}, {statistics.median(H0_W2_118)}")
print(f"H0_28 W1,W2: {statistics.median(H0_W1_28)}, {statistics.median(H0_W2_28)}")

print(f"omg_0 W1,W2: {statistics.median(omg_0_W1)}, {statistics.median(omg_0_W2)}")
print(f"omg_0_118 W1,W2: {statistics.median(omg_0_W1_118)}, {statistics.median(omg_0_W2_118)}")
print(f"omg_0_28 W1,W2: {statistics.median(omg_0_W1_28)}, {statistics.median(omg_0_W2_28)}")

# fig, ax = plt.subplots(2,2)
# histH0_W1, binH0_W1 = np.histogram(H0_W1_118, bins = "auto")
# histH0_W2, binH0_W2 = np.histogram(H0_W2_118, bins = "auto")
# histOmg_0_W1, binOmg_0_W1 = np.histogram(omg_0_W1_118, bins = "auto")
# histOmg_0_W2, binOmg_0_W2 = np.histogram(omg_0_W2_118, bins = "auto")

# ax[0][0].stairs(histH0_W1, binH0_W1, edgecolor = "blue")
# ax[1][0].stairs(histH0_W2, binH0_W2, edgecolor = "red")
# ax[0][1].stairs(histOmg_0_W1, binOmg_0_W1, edgecolor = "blue")
# ax[1][1].stairs(histOmg_0_W2, binOmg_0_W2, edgecolor = "red")
# plt.show()

DL_tauW1_meds = []
DL_tauW1_118_meds = []
DL_tauW1_28_meds = []
DL_tauW2_meds = []
DL_tauW2_118_meds = []
DL_tauW2_28_meds = []

for i in range(len(DL_tauW1_clean)):
    DL_tauW1_meds.append(statistics.median(DL_tauW1_clean[i]))
    DL_tauW1_118_meds.append(statistics.median(DL_tauW1_118_clean[i]))
    DL_tauW1_28_meds.append(statistics.median(DL_tauW1_28_clean[i]))

for i in range(len(DL_tauW2_clean)):
    DL_tauW2_meds.append(statistics.median(DL_tauW2_clean[i]))
    DL_tauW2_118_meds.append(statistics.median(DL_tauW2_118_clean[i]))
    DL_tauW2_28_meds.append(statistics.median(DL_tauW2_28_clean[i]))

fig, ax = plt.subplots(2,3)

for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax[i,j].set_xscale("log")
        ax[i,j].set_yscale("log")

ax[0,0].plot(zCMB_W1,DL_tauW1_meds, "bo", ms = 1)
ax[1,0].plot(zCMB_W2,DL_tauW2_meds, "ro", ms = 1)
ax[0,1].plot(zCMB_W1,DL_tauW1_118_meds, "bo", ms = 1)
ax[1,1].plot(zCMB_W2,DL_tauW2_118_meds, "ro", ms = 1)
ax[0,2].plot(zCMB_W1,DL_tauW1_28_meds, "bo", ms = 1)
ax[1,2].plot(zCMB_W2,DL_tauW2_28_meds, "ro", ms = 1)

for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax[i,j].plot(np.linspace(0,3,100),(299792.458/73.04)*np.linspace(0,3,100),ls = "--",  color = "red")

plt.show()

            