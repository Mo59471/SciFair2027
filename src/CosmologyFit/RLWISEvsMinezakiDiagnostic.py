import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
import math
from astropy.coordinates import SkyCoord
from astropy import units as u
import scipy
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid
import mpfit
import linmix

lmTog = False #Toggle use of linmix

fid_H0 = 10 #Fiducial/false H0 for determining a cosmology independent ratio RK/RW
fid_Omg0 = 0.27 #Fiducial/false omg_0 for determining a cosmology independent ratio RK/RW

def main():
    #WISE Data
    objID, SDSSID, zReported, zHel, logL, logL_err, tauW1, tauW1_err, tauW1_Err, tauW2, tauW2_err, tauW2_Err =np.genfromtxt(
        "WISEData.csv", delimiter=",",usecols = (0,2,3,5,16,17,26,27,28,39,40,41), 
        skip_header=1, unpack = True, missing_values=("","#NUM!"), filling_values=-999, dtype=None, comments= None) #-999 indicates no value

    #MAGNUM Data
    zCMB_mag, tau_mag, tau_mag_err, tau_mag_Err, logLmag, logLmag_err, flux_MAG, flux_MAG_Err, flux_MAG_err, obsBand = np.genfromtxt(
        "MAGNUMData.csv", delimiter=",", usecols = (2,5,6,7,10,13,14,15,16, 20), 
        skip_header=1, unpack = True, missing_values=(""), filling_values=-999, dtype=None
    )

    # delIndex = []
    # for i in range(len(zHel)):
    #     if not (zHel[i] <= 1.3 and zHel[i] >= 1.090909091): #not (zHel[i] <= 0.6 and zHel[i] >= 0.5454545455):
    #         delIndex.append(i)
    
    # zHel = np.delete(zHel, delIndex)
    # logL = np.delete(logL,delIndex)
    # logL_err = np.delete(logL_err,delIndex)
    # tauW1 = np.delete(tauW1,delIndex)
    # tauW1_err = np.delete(tauW1_err,delIndex)
    # tauW1_Err = np.delete(tauW1_Err,delIndex)
    # tauW2 = np.delete(tauW2,delIndex)
    # tauW2_err = np.delete(tauW2_err,delIndex)
    # tauW2_Err = np.delete(tauW2_Err,delIndex)

    #WISE flux
    flux_WISE = []
    flux_WISE_err = []
    flux_WISE_Err = []
    
    zCMB = []

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
    sTauW1_WISE = []
    sTauW2_WISE = []

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

    #V-band luminosities assuming fiducial/false cosmology
    logLV_WISE = []
    logLV_MAG = []

    cleanLogTauW1 = []
    cleanLogTauW1_118 = []
    cleanLogTauW1_28 = []
    cleanLogTauW1_err = [] #Standard deviation of the sampled taus in log space
    cleanLogTauW1_118_err = []
    cleanLogTauW1_28_err = []

    cleanLV_W1 = []
    cleanLV_W1_err = []

    cleanLogTauW2 = []
    cleanLogTauW2_118 = []
    cleanLogTauW2_28 = []
    cleanLogTauW2_err = []
    cleanLogTauW2_118_err = []
    cleanLogTauW2_28_err = []

    cleanLV_W2 = []
    cleanLV_W2_err = []

    cleanLogTauK = []
    cleanLogTauK_err = []
    cleanLV_K = []
    cleanLV_K_err = []
    
    DL_given = []

    #Get raw time lags
    for i in range(len(tauW1)):
        if tauW1[i] != -999:
            rTauW1.append(tauW1[i]/(1+zReported[i])**-0.38) #Removes Mandal's net correction factor, including time dilation, of (1+z)^gamma-1 (gamma-1 =-0.38)
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
                while W2 <= 0 or W1 <= 0: #If less than 0, repeat sampling
                    W2 = sampleSplitNormal(modeW2[j], rTauW2_err[j], rTauW2_Err[j])
                    W1 = sampleSplitNormal(modeW1[j],rTauW1_err[j],rTauW1_Err[j])
                gam = math.log10(W2/W1)/math.log10(4.6/3.4) #Find the power law corresponding to the given ratio; uses WISE nominal wavelengths W1 = 3.4 and W2 = 4.6
                
                #Append the raw sampled value to lists of lists (each sublist = distribution of sampled values for each object)
                if len(sTauW1) != len(modeW1) and len(sTauW2)!= len(modeW2):
                    gammas.append([gam])
                    sTauW2.append([(W2/(1+zCMB[j])*((1+zCMB[j])**gam)*(2.2**gam))/(4.6**gam)]) #Reapplies gamma correction and time dilation correction with zCMB
                    sTauW1.append([(W1/(1+zCMB[j])*((1+zCMB[j])**gam)*(2.2**gam))/(3.4**gam)])
                    sTauW2_118.append([(W2/(1+zCMB[j])*((1+zCMB[j])**1.18)*(2.2**1.18))/(4.6**1.18)])
                    sTauW1_118.append([(W1/(1+zCMB[j])*((1+zCMB[j])**1.18)*(2.2**1.18))/(3.4**1.18)])
                    sTauW2_28.append([(W2/(1+zCMB[j])*((1+zCMB[j])**2.8)*(2.2**2.8))/(4.6**2.8)])
                    sTauW1_28.append([(W1/(1+zCMB[j])*((1+zCMB[j])**2.8)*(2.2**2.8))/(3.4**2.8)])
                    sTauW1_WISE.append([W1*(1+zReported[j])**-0.38]) #Equivalent to sampling Mandal's original time lags
                    sTauW2_WISE.append([W2*(1+zReported[j])**-0.38])
                else:
                    gammas[j].append(gam)
                    sTauW2[j].append((W2/(1+zCMB[j])*((1+zCMB[j])**gam)*(2.2**gam))/(4.6**gam))
                    sTauW1[j].append((W1/(1+zCMB[j])*((1+zCMB[j])**gam)*(2.2**gam))/(3.4**gam))
                    sTauW2_118[j].append((W2/(1+zCMB[j])*((1+zCMB[j])**1.18)*(2.2**1.18))/(4.6**1.18))
                    sTauW1_118[j].append((W1/(1+zCMB[j])*((1+zCMB[j])**1.18)*(2.2**1.18))/(3.4**1.18))
                    sTauW2_28[j].append((W2/(1+zCMB[j])*((1+zCMB[j])**2.8)*(2.2**2.8))/(4.6**2.8))
                    sTauW1_28[j].append((W1/(1+zCMB[j])*((1+zCMB[j])**2.8)*(2.2**2.8))/(3.4**2.8)) 
                    sTauW1_WISE[j].append(W1*(1+zReported[j])**-0.38) #Equivalent to sampling Mandal's original time lags
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

    #Calculate distribution of median gammas from which to sample from for objects without a W1 and W2 lag
    for i in gammas:
        if i[0] != -999:
            medGammaDistr.append(statistics.median(i)) 
    
    # #Test of gamma distributions
    # gamDistr_W1_greater = []
    # gamDistr_W1_lesser = []
    # gamDistr_W2_greater = []
    # gamDistr_W2_lesser = []
    
    # for i in range(len(gammas)):
    #     if gammas[i][0] != -999:
    #         if zCMB[i] >= 0.5454545455:
    #             gamDistr_W1_greater.append(np.median(gammas[i]))
    #         else:
    #             gamDistr_W1_lesser.append(np.median(gammas[i]))
    #         if zCMB[i] >= 1.090909091:
    #             gamDistr_W2_greater.append(np.median(gammas[i]))
    #         else:
    #             gamDistr_W2_lesser.append(np.median(gammas[i]))
    
    # print(f"W1 med gamma >= 0.55: {np.median(gamDistr_W1_greater)}")
    # print(f"W1 med gamma <= 0.55: {np.median(gamDistr_W1_lesser)}")
    # print(f"W2 med gamma >= 1.09: {np.median(gamDistr_W2_greater)}")
    # print(f"W2 med gamma <= 1.09: {np.median(gamDistr_W2_lesser)}")
    # print("")
    # print(f"W1 avg gamma >= 0.55: {np.mean(gamDistr_W1_greater)}")
    # print(f"W1 avg gamma <= 0.55: {np.mean(gamDistr_W1_lesser)}")
    # print(f"W2 avg gamma >= 1.09: {np.mean(gamDistr_W2_greater)}")
    # print(f"W2 avg gamma <= 1.09: {np.mean(gamDistr_W2_lesser)}")
    # print("")
    
    # fig, ax = plt.subplots(1,2)    
    # ax[0].plot([zCMB[i] for i in range(len(zCMB)) if gammas[i][0] != -999], [statistics.median(i) for i in gammas if i[0] != -999], "bo", ms = 1)
    # ax[0].plot(np.linspace(0,0.5454545455,10), np.median(gamDistr_W1_lesser)*np.ones(10), ls = "--", color = "green")
    # ax[0].plot(np.linspace(0.5454545455,np.max(zCMB),10), np.median(gamDistr_W1_greater)*np.ones(10), ls = "--", color = "green")
    # ax[0].plot(np.linspace(0,0.5454545455,10), np.mean(gamDistr_W1_lesser)*np.ones(10), ls = "--", color = "orange")
    # ax[0].plot(np.linspace(0.5454545455,np.max(zCMB),10), np.mean(gamDistr_W1_greater)*np.ones(10), ls = "--", color = "orange")

    # ax[1].plot([zCMB[i] for i in range(len(zCMB)) if gammas[i][0] != -999], [statistics.median(i) for i in gammas if i[0] != -999], "ro", ms = 1)
    # ax[1].plot(np.linspace(0,1.090909091,10), np.median(gamDistr_W2_lesser)*np.ones(10), ls = "--", color = "green")
    # ax[1].plot(np.linspace(1.090909091,np.max(zCMB),10), np.median(gamDistr_W2_greater)*np.ones(10), ls = "--", color = "green")
    # ax[1].plot(np.linspace(0,1.090909091,10), np.mean(gamDistr_W2_lesser)*np.ones(10), ls = "--", color = "orange")
    # ax[1].plot(np.linspace(1.090909091,np.max(zCMB),10), np.mean(gamDistr_W2_greater)*np.ones(10), ls = "--", color = "orange")
    
    # plt.show()

    #Handle corrections for objects without a W1 and W2 lag (no independent distribution of gamma)
    #This assumes that the sampled W1 or W2 lag is independent from gamma in isolation (i.e. gamma can't be determined without both)
    for i in range(len(modeW1)):
        if sTauW1[i][0] == -999 and modeW1[i] != -999:
            sTauW1[i] = []
            sTauW1_118[i] = []
            sTauW1_28[i] = []
            sTauW1_WISE[i] = []
            for j in range(10000):
                W1 = sampleSplitNormal(modeW1[i],rTauW1_err[i],rTauW1_Err[i])
                while W1 <= 0:
                    W1 = sampleSplitNormal(modeW1[i],rTauW1_err[i],rTauW1_Err[i])
                gamma = random.choice(medGammaDistr)       
                sTauW1[i].append((W1/(1+zCMB[i])*((1+zCMB[i])**gamma)*(2.2**gamma))/(3.4**gamma))
                sTauW1_118[i].append((W1/(1+zCMB[i])*((1+zCMB[i])**1.18)*(2.2**1.18))/(3.4**1.18))
                sTauW1_28[i].append((W1/(1+zCMB[i])*((1+zCMB[i])**2.8)*(2.2**2.8))/(3.4**2.8))
                sTauW1_WISE[i].append(W1*(1+zReported[i])**-0.38)                
        if sTauW2[i][0] == -999 and modeW2[i] != -999:
            sTauW2[i] = []
            sTauW2_118[i] = []
            sTauW2_28[i] = []
            sTauW2_WISE[i] = []
            for j in range(10000):
                W2 = sampleSplitNormal(modeW2[i],rTauW2_err[i],rTauW2_Err[i])
                while W2 <= 0:
                    W2 = sampleSplitNormal(modeW2[i],rTauW2_err[i],rTauW2_Err[i])
                gamma = random.choice(medGammaDistr)              
                sTauW2[i].append((W2/(1+zCMB[i])*((1+zCMB[i])**gamma*(2.2**gamma/(4.6**gamma)))))
                sTauW2_118[i].append((W2/(1+zCMB[i])*((1+zCMB[i])**1.18)*(2.2**1.18))/(4.6**1.18))
                sTauW2_28[i].append((W2/(1+zCMB[i])*((1+zCMB[i])**2.8)*(2.2**2.8))/(4.6**2.8))
                sTauW2_WISE[i].append(W2/(1+zCMB[i])*(1+zReported[i])**-0.38)        

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

    #Get WISE luminosity distances using the adopted cosmology of Shen and Liu
    def integrand(z, omg_0):
        return 1/np.sqrt((omg_0*(1+z)**3+(1-omg_0)))

    for i in range(len(zHel)):
        integral = quad(integrand, 0, zHel[i], args = (0.3,))
        DL_given.append(3.0856776e22*(1+zHel[i])*299792.458/70*integral[0]) #Large conversion factor in front to convert Mpc to m

    #Get WISE fluxes from luminosity distances 
    for i in range(len(logL)):
        if logL[i] != -999:
            if zHel[i] < 0.7: #Correct back to spectral luminosity in watts/Hz
                L = 10**logL[i]/(299792458/(5100*1e-10)) * 1e-7 #1e-7 converts from erg/s to watts
            elif zHel[i] < 1.9:
                L = 10**logL[i]/(299792458/(3000*1e-10)) * 1e-7
            else:
                L = 10**logL[i]/(299792458/(1350*1e-10)) * 1e-7
            L_err = 10**math.log10(L)-10**(math.log10(L)-logL_err[i])
            L_Err = 10**(math.log10(L)+logL_err[i])-10**math.log10(L)
            flux_WISE.append(L*(1+zHel[i])/(4*math.pi*DL_given[i]**2)) #watts/m2/Hz
            flux_WISE_err.append(L_err*(1+zHel[i])/(4*math.pi*DL_given[i]**2))
            flux_WISE_Err.append(L_Err*(1+zHel[i])/(4*math.pi*DL_given[i]**2))
        
    #Convert MAGNUM fluxes into watts/Hz/m^2 from mJy
    flux_MAG[:] = [i * 1e-29 if i != -999 else i for i in flux_MAG]
    flux_MAG_err[:] = [i * 1e-29 if i != -999 else i for i in flux_MAG_err]
    flux_MAG_Err[:] = [i * 1e-29 if i != -999 else i for i in flux_MAG_Err]
    
    #Get MAGNUM luminosities using MAGNUM fluxes corrected to V band assuming power law index 0.5
    for i in range(len(flux_MAG)):
        if flux_MAG[i] != -999:
            DL = 3.0856776e22 * (1+zCMB_mag[i]) * 299792.458/fid_H0 * quad(integrand, 0, zCMB_mag[i], args = (fid_Omg0,))[0] #In meters
            if str(obsBand[i]) == "R":
                logLV_MAG.append(
                    math.log10(flux_MAG[i] * (6283 / (5500*(1+zCMB_mag[i]))) ** -0.5 * 4 * math.pi * DL**2 / (1+zCMB_mag[i])) 
                )
            elif str(obsBand[i]) == "I":
                logLV_MAG.append(
                    math.log10(flux_MAG[i] * (7774 / (5500*(1+zCMB_mag[i]))) ** -0.5 * 4 * math.pi * DL**2 / (1+zCMB_mag[i]))
                )
            else:
                logLV_MAG.append(
                    math.log10(flux_MAG[i] * 4 * math.pi * DL**2 / (1+zCMB_mag[i])) 
                )
        else:
            logLV_MAG.append(-999)
    
    #Convert WISE fluxes to V band luminosity using fiducial/false cosmology
    #Power law takes the form: LV = LX * (freqV/freqX)^-0.5
    for i in range(len(flux_WISE)):
        DL = 3.0856776e22 * (1+zCMB[i]) * 299792.458/fid_H0 * quad(integrand, 0, zCMB[i], args = (fid_Omg0,))[0] #In meters
        if flux_WISE[i] != -999:
            if zHel[i] < 0.7:
                logLV_WISE.append(
                    math.log10(flux_WISE[i] * (5100 / 5500) ** -0.5 * 4 * math.pi * DL**2 / (1+zCMB[i]))
                )
            elif zHel[i] < 1.9:
                logLV_WISE.append(
                    math.log10(flux_WISE[i] * (3000 / 5500) ** -0.5 * 4 * math.pi * DL**2 / (1+zCMB[i]))
                )
            else:
                logLV_WISE.append(
                    math.log10(flux_WISE[i] * (1350 / 5500) ** -0.5 * 4 * math.pi * DL**2 / (1+zCMB[i]))
                )
        else:
            logLV_WISE.append(-999)

    for i in range(len(sTauW1)):
        if sTauW1[i][0] != -999:
            cleanLogTauW1.append(math.log10(statistics.median(sTauW1[i])))
            cleanLogTauW1_118.append(math.log10(statistics.median(sTauW1_118[i])))
            cleanLogTauW1_28.append(math.log10(statistics.median(sTauW1_28[i])))
            cleanLogTauW1_err.append(np.std(np.log10(sTauW1[i])))
            cleanLogTauW1_118_err.append(np.std(np.log10(sTauW1_118[i])))
            cleanLogTauW1_28_err.append(np.std(np.log10(sTauW1_28[i])))
            
            cleanLV_W1.append(logLV_WISE[i])
            cleanLV_W1_err.append(logL_err[i]) #Log errors don't change with multiplication -> logL_err translates to logLV_err
            
        if sTauW2[i][0] != -999:
            cleanLogTauW2.append(math.log10(statistics.median(sTauW2[i])))
            cleanLogTauW2_118.append(math.log10(statistics.median(sTauW2_118[i])))
            cleanLogTauW2_28.append(math.log10(statistics.median(sTauW2_28[i])))
            cleanLogTauW2_err.append(np.std(np.log10(sTauW2[i])))
            cleanLogTauW2_118_err.append(np.std(np.log10(sTauW2_118[i])))
            cleanLogTauW2_28_err.append(np.std(np.log10(sTauW2_28[i])))
            
            cleanLV_W2.append(logLV_WISE[i])
            cleanLV_W2_err.append(logL_err[i])

    for i in range(len(tau_mag)):
        if tau_mag[i] != -999 and logLV_MAG[i] != -999:
            cleanLogTauK.append(math.log10(tau_mag[i]))
            cleanLogTauK_err.append(
                (math.log10(tau_mag[i]+tau_mag_Err[i])-math.log10(tau_mag[i]-tau_mag_err[i]))/2
            )
            cleanLV_K.append(logLV_MAG[i])
            cleanLV_K_err.append(logLmag_err[i])

    #Fit R-L relationship for MAGNUM and WISE data
    #Fit to medians of WISE sampled time lags

    #MPFIT for slope fixed at 0.5:
    def linefit(p, fjac = None, x = None, y = None, err = None):
        model = p[0] + 0.5* x #modelY = a +bx; slope fixed at 0.5
        status = 0
        effErr = np.sqrt(err**2+0.2**2) #Intrinsic scatter of 0.2 conservatively assumed
        return [status, (y-model)/effErr]

    p0 = [1]

    mW1 = mpfit.mpfit(
        linefit,
        p0,
        functkw = {"x":np.array(cleanLV_W1, dtype=float), "y":np.array(cleanLogTauW1),"err":np.array(cleanLogTauW1_err)},
        quiet=True
    )

    mW1_118 = mpfit.mpfit(
        linefit,
        p0,
        functkw = {"x":np.array(cleanLV_W1, dtype=float), "y":np.array(cleanLogTauW1_118),"err":np.array(cleanLogTauW1_118_err)},
        quiet=True
    )

    mW1_28 = mpfit.mpfit(
        linefit,
        p0,
        functkw = {"x":np.array(cleanLV_W1, dtype=float), "y":np.array(cleanLogTauW1_28),"err":np.array(cleanLogTauW1_28_err)},
        quiet=True
    )

    mW2 = mpfit.mpfit(
        linefit,
        p0,
        functkw = {"x":np.array(cleanLV_W2, dtype=float), "y":np.array(cleanLogTauW2),"err":np.array(cleanLogTauW2_err)},
        quiet=True
    )

    mW2_118 = mpfit.mpfit(
        linefit,
        p0,
        functkw = {"x":np.array(cleanLV_W2, dtype=float), "y":np.array(cleanLogTauW2_118),"err":np.array(cleanLogTauW2_118_err)},
        quiet=True
    )

    mW2_28 = mpfit.mpfit(
        linefit,
        p0,
        functkw = {"x":np.array(cleanLV_W2, dtype=float), "y":np.array(cleanLogTauW2_28),"err":np.array(cleanLogTauW2_28_err)},
        quiet=True
    )

    mK = mpfit.mpfit(
        linefit,
        p0,
        functkw = {"x":np.array(cleanLV_K, dtype=float), "y":np.array(cleanLogTauK),"err":np.array(cleanLogTauK_err)},
        quiet=True
    )

    MPFIT_interceptW1 =  float(mW1.params[0])
    MPFIT_interceptW1_err = float(mW1.perror[0])
    MPFIT_interceptW1_118 =  float(mW1_118.params[0])
    MPFIT_interceptW1_118_err = float(mW1_118.perror[0])
    MPFIT_interceptW1_28 =  float(mW1_28.params[0])
    MPFIT_interceptW1_28_err = float(mW1_28.perror[0])
    
    MPFIT_interceptW2 =  float(mW2.params[0])
    MPFIT_interceptW2_err = float(mW2.perror[0])
    MPFIT_interceptW2_118 =  float(mW2_118.params[0])
    MPFIT_interceptW2_118_err = float(mW2_118.perror[0])
    MPFIT_interceptW2_28 =  float(mW2_28.params[0])
    MPFIT_interceptW2_28_err = float(mW2_28.perror[0])

    MPFIT_interceptK = float(mK.params[0])
    MPFIT_interceptK_err = float(mK.perror[0])

    print("MPFIT best fit parameters:")
    print(f"MPFIT intercept W1, W2: {MPFIT_interceptW1}, {MPFIT_interceptW2}")
    print(f"MPFIT intercept W1_118, W2_118: {MPFIT_interceptW1_118}, {MPFIT_interceptW2_118}")
    print(f"MPFIT intercept W1_28, W2_28: {MPFIT_interceptW1_28}, {MPFIT_interceptW2_28}")
    print(f"MPFIT intercept K: {MPFIT_interceptK}")
    print("")
    print("MPFIT errors:")
    print(f"MPFIT error W1, W2: {MPFIT_interceptW1_err}, {MPFIT_interceptW2_err}")
    print(f"MPFIT error W1_118, W2_118: {MPFIT_interceptW1_118_err}, {MPFIT_interceptW2_118_err}")
    print(f"MPFIT error W1_28, W2_28: {MPFIT_interceptW1_28_err}, {MPFIT_interceptW2_28_err}")
    print(f"MPFIT error K: {MPFIT_interceptK_err}")
    print("")
    print(f"Ratio K/W1: {10**(MPFIT_interceptK-MPFIT_interceptW1)} +{10**(MPFIT_interceptK-MPFIT_interceptK_err-MPFIT_interceptW1+MPFIT_interceptW1_err)-10**(MPFIT_interceptK-MPFIT_interceptW1)}, -{10**(MPFIT_interceptK-MPFIT_interceptW1)-10**(MPFIT_interceptK+MPFIT_interceptK_err-MPFIT_interceptW1-MPFIT_interceptW1_err)}")
    print(f"Ratio K/W1_118: {10**(MPFIT_interceptK-MPFIT_interceptW1_118)} +{10**(MPFIT_interceptK-MPFIT_interceptK_err-MPFIT_interceptW1_118+MPFIT_interceptW1_118_err)-10**(MPFIT_interceptK-MPFIT_interceptW1_118)}, -{10**(MPFIT_interceptK-MPFIT_interceptW1_118)-10**(MPFIT_interceptK+MPFIT_interceptK_err-MPFIT_interceptW1_118-MPFIT_interceptW1_118_err)}")
    print(f"Ratio K/W1_28: {10**(MPFIT_interceptK-MPFIT_interceptW1_28)} +{10**(MPFIT_interceptK-MPFIT_interceptK_err-MPFIT_interceptW1_28+MPFIT_interceptW1_28_err)-10**(MPFIT_interceptK-MPFIT_interceptW1_28)}, -{10**(MPFIT_interceptK-MPFIT_interceptW1_28)-10**(MPFIT_interceptK+MPFIT_interceptK_err-MPFIT_interceptW1_28-MPFIT_interceptW1_28_err)}")
    print(f"Ratio K/W2: {10**(MPFIT_interceptK-MPFIT_interceptW2)} +{10**(MPFIT_interceptK-MPFIT_interceptK_err-MPFIT_interceptW2+MPFIT_interceptW2_err)-10**(MPFIT_interceptK-MPFIT_interceptW2)}, -{10**(MPFIT_interceptK-MPFIT_interceptW2)-10**(MPFIT_interceptK+MPFIT_interceptK_err-MPFIT_interceptW2-MPFIT_interceptW2_err)}")
    print(f"Ratio K/W2_118: {10**(MPFIT_interceptK-MPFIT_interceptW2_118)} +{10**(MPFIT_interceptK-MPFIT_interceptK_err-MPFIT_interceptW2_118+MPFIT_interceptW2_118_err)-10**(MPFIT_interceptK-MPFIT_interceptW2_118)}, -{10**(MPFIT_interceptK-MPFIT_interceptW2_118)-10**(MPFIT_interceptK+MPFIT_interceptK_err-MPFIT_interceptW2_118-MPFIT_interceptW2_118_err)}")
    print(f"Ratio K/W2_28: {10**(MPFIT_interceptK-MPFIT_interceptW2_28)} +{10**(MPFIT_interceptK-MPFIT_interceptK_err-MPFIT_interceptW2_28+MPFIT_interceptW2_28_err)-10**(MPFIT_interceptK-MPFIT_interceptW2_28)}, -{10**(MPFIT_interceptK-MPFIT_interceptW2_28)-10**(MPFIT_interceptK+MPFIT_interceptK_err-MPFIT_interceptW2_28-MPFIT_interceptW2_28_err)}")
    print("")

    if lmTog:
        #linmix for free slope:
        lmW1 = linmix.LinMix(
            np.array(cleanLV_W1, dtype = float), np.array(cleanLogTauW1, dtype = float), 
            np.array(cleanLV_W1_err, dtype = float), np.array(cleanLogTauW1_err, dtype = float), K=2
            )
        
        lmW1_118 = linmix.LinMix(
            np.array(cleanLV_W1, dtype = float), np.array(cleanLogTauW1_118, dtype = float), 
            np.array(cleanLV_W1_err, dtype = float), np.array(cleanLogTauW1_118_err, dtype = float), K=2
            )
        
        lmW1_28 = linmix.LinMix(
            np.array(cleanLV_W1, dtype = float), np.array(cleanLogTauW1_28, dtype = float), 
            np.array(cleanLV_W1_err, dtype = float), np.array(cleanLogTauW1_28_err, dtype = float), K=2
            )

        lmW2 = linmix.LinMix(
            np.array(cleanLV_W2, dtype = float), np.array(cleanLogTauW2, dtype = float), 
            np.array(cleanLV_W2_err, dtype = float), np.array(cleanLogTauW2_err, dtype = float), K=2
            )
        
        lmW2_118 = linmix.LinMix(
            np.array(cleanLV_W2, dtype = float), np.array(cleanLogTauW2_118, dtype = float), 
            np.array(cleanLV_W2_err, dtype = float), np.array(cleanLogTauW2_118_err, dtype = float), K=2
            )
        
        lmW2_28 = linmix.LinMix(
            np.array(cleanLV_W2, dtype = float), np.array(cleanLogTauW2_28, dtype = float), 
            np.array(cleanLV_W2_err, dtype = float), np.array(cleanLogTauW2_28_err, dtype = float), K=2
            )
        
        lmK = linmix.LinMix(
            np.array(cleanLV_K, dtype = float), np.array(cleanLogTauK, dtype = float), 
            np.array(cleanLV_K_err, dtype = float), np.array(cleanLogTauK_err, dtype = float), K=2
            )
        
        lmW1.run_mcmc(silent=True)
        lmW1_118.run_mcmc(silent=True)
        lmW1_28.run_mcmc(silent=True)

        lmW2.run_mcmc(silent=True)
        lmW2_118.run_mcmc(silent=True)
        lmW2_28.run_mcmc(silent=True)
        
        lmK.run_mcmc(silent=True)
        
        aChainW1 = lmW1.chain['alpha']
        aChainW1_118 = lmW1_118.chain['alpha']
        aChainW1_28 = lmW1_28.chain['alpha']

        aChainW2 = lmW2.chain['alpha']
        aChainW2_118 = lmW2_118.chain['alpha']
        aChainW2_28 = lmW2_28.chain['alpha']
        
        aChainK = lmK.chain['alpha']
        
        bChainW1 = lmW1.chain['beta']
        bChainW1_118 = lmW1_118.chain['beta']
        bChainW1_28 = lmW1_28.chain['beta']

        bChainW2 = lmW2.chain['beta']
        bChainW2_118 = lmW2_118.chain['beta']
        bChainW2_28 = lmW2_28.chain['beta']

        bChainK = lmK.chain['beta']
        
        linmix_interceptW1 = np.median(aChainW1)
        linmix_interceptW1_err = np.std(aChainW1)
        linmix_interceptW1_118 = np.median(aChainW1_118)
        linmix_interceptW1_118_err = np.std(aChainW1_118)
        linmix_interceptW1_28 = np.median(aChainW1_28)
        linmix_interceptW1_28_err = np.std(aChainW1_28)
        
        linmix_interceptW2 = np.median(aChainW2)
        linmix_interceptW2_err = np.std(aChainW2)
        linmix_interceptW2_118 = np.median(aChainW2_118)
        linmix_interceptW2_118_err = np.std(aChainW2_118)
        linmix_interceptW2_28 = np.median(aChainW2_28)
        linmix_interceptW2_28_err = np.std(aChainW2_28)
        
        linmix_interceptK = np.median(aChainK)
        linmix_interceptK_err = np.std(aChainK)

        linmix_slopeW1 = np.median(bChainW1)
        linmix_slopeW1_err = np.std(bChainW1)
        linmix_slopeW1_118 = np.median(bChainW1_118)
        linmix_slopeW1_118_err = np.std(bChainW1_118)
        linmix_slopeW1_28 = np.median(bChainW1_28)
        linmix_slopeW1_28_err = np.std(bChainW1_28)

        linmix_slopeW2 = np.median(bChainW2)
        linmix_slopeW2_err = np.std(bChainW2)
        linmix_slopeW2_118 = np.median(bChainW2_118)
        linmix_slopeW2_118_err = np.std(bChainW2_118)
        linmix_slopeW2_28 = np.median(bChainW2_28)
        linmix_slopeW2_28_err = np.std(bChainW2_28)
        
        linmix_slopeK = np.median(bChainK)
        linmix_slopeK_err = np.std(bChainK)

        print("")
        print("LINMIX best fit parameters:")
        print(f"LINMIX intercept, slope W1: {linmix_interceptW1}, {linmix_slopeW1}")
        print(f"LINMIX intercept, slope W1_118: {linmix_interceptW1_118}, {linmix_slopeW1_118}")
        print(f"LINMIX intercept, slope W1_28: {linmix_interceptW1_28}, {linmix_slopeW1_28}")
        print(f"LINMIX intercept, slope W2: {linmix_interceptW2}, {linmix_slopeW2}")
        print(f"LINMIX intercept, slope W2_118: {linmix_interceptW2_118}, {linmix_slopeW2_118}")
        print(f"LINMIX intercept, slope W2_28: {linmix_interceptW2_28}, {linmix_slopeW2_28}")
        print(f"LINMIX intercept, slope K: {linmix_interceptK}, {linmix_slopeK}")
        print("")
        print("LINMIX errors:")
        print(f"LINMIX error, intercept, slope W1: {linmix_interceptW1_err}, {linmix_slopeW1_err}")
        print(f"LINMIX error, intercept, slope W1_118: {linmix_interceptW1_118_err}, {linmix_slopeW1_118_err}")
        print(f"LINMIX error, intercept, slope W1_28: {linmix_interceptW1_28_err}, {linmix_slopeW1_28_err}")
        print(f"LINMIX error, intercept, slope W2: {linmix_interceptW2_err}, {linmix_slopeW2_err}")
        print(f"LINMIX error, intercept, slope W2_118: {linmix_interceptW2_118_err}, {linmix_slopeW2_118_err}")
        print(f"LINMIX error, intercept, slope W2_28: {linmix_interceptW2_28_err}, {linmix_slopeW2_28_err}")
        print(f"LINMIX error, intercept, slope K: {linmix_interceptK_err}, {linmix_slopeK_err}")

    fig, ax = plt.subplots(2,5)

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i,j].set_xscale('log')
            ax[i,j].set_yscale('log')
    
    cleanLV_W1 = np.asarray(cleanLV_W1)
    cleanLV_W2 = np.asarray(cleanLV_W2)
    cleanLV_K = np.asarray(cleanLV_K)
    cleanLogTauW1 = np.asarray(cleanLogTauW1)
    cleanLogTauW1_118 = np.asarray(cleanLogTauW1_118)
    cleanLogTauW1_28 = np.asarray(cleanLogTauW1_28)
    cleanLogTauW2 = np.asarray(cleanLogTauW2)
    cleanLogTauW2_118 = np.asarray(cleanLogTauW2_118)
    cleanLogTauW2_28 = np.asarray(cleanLogTauW2_28)
    cleanLogTauK = np.asarray(cleanLogTauK)

    #Plot W1 R-L    
    ax[0,0].errorbar(
        10**cleanLV_W1, 10**cleanLogTauW1, 
        xerr=[10**np.asarray(cleanLV_W1)-10**(np.asarray(cleanLV_W1)-np.asarray(cleanLV_W1_err)),
            10**(np.asarray(cleanLV_W1)+np.asarray(cleanLV_W1_err))-10**np.asarray(cleanLV_W1)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauW1))-10**(np.asarray(cleanLogTauW1)-np.asarray(cleanLogTauW1_err)),
                10**(np.asarray(cleanLogTauW1)+np.asarray(cleanLogTauW1_err))-10**np.asarray(cleanLogTauW1)],
        ms = 1, color = "blue", elinewidth = 0.3, fmt = "o", zorder = 1
        )
    ax[0,1].errorbar(
        10**cleanLV_W1, 10**cleanLogTauW1_118, 
        xerr=[10**np.asarray(cleanLV_W1)-10**(np.asarray(cleanLV_W1)-np.asarray(cleanLV_W1_err)),
            10**(np.asarray(cleanLV_W1)+np.asarray(cleanLV_W1_err))-10**np.asarray(cleanLV_W1)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauW1_118))-10**(np.asarray(cleanLogTauW1_118)-np.asarray(cleanLogTauW1_118_err)),
                10**(np.asarray(cleanLogTauW1_118)+np.asarray(cleanLogTauW1_118_err))-10**np.asarray(cleanLogTauW1_118)],
        ms = 1, color = "blue", elinewidth = 0.3, fmt = "o", zorder = 1
        )
    ax[0,2].errorbar(
        10**cleanLV_W1, 10**cleanLogTauW1_28, 
        xerr=[10**np.asarray(cleanLV_W1)-10**(np.asarray(cleanLV_W1)-np.asarray(cleanLV_W1_err)),
            10**(np.asarray(cleanLV_W1)+np.asarray(cleanLV_W1_err))-10**np.asarray(cleanLV_W1)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauW1_28))-10**(np.asarray(cleanLogTauW1_28)-np.asarray(cleanLogTauW1_28_err)),
                10**(np.asarray(cleanLogTauW1_28)+np.asarray(cleanLogTauW1_28_err))-10**np.asarray(cleanLogTauW1_28)],
        ms = 1, color = "blue", elinewidth = 0.3, fmt = "o", zorder = 1
        )

    #Plot W1 R-L with K R-L overplotted
    ax[0,3].errorbar(
        10**cleanLV_W1, 10**cleanLogTauW1, 
        xerr=[10**np.asarray(cleanLV_W1)-10**(np.asarray(cleanLV_W1)-np.asarray(cleanLV_W1_err)),
            10**(np.asarray(cleanLV_W1)+np.asarray(cleanLV_W1_err))-10**np.asarray(cleanLV_W1)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauW1))-10**(np.asarray(cleanLogTauW1)-np.asarray(cleanLogTauW1_err)),
                10**(np.asarray(cleanLogTauW1)+np.asarray(cleanLogTauW1_err))-10**np.asarray(cleanLogTauW1)],
        ms = 1, color = "blue", elinewidth = 0.3, fmt = "o", zorder = 1
        )
    ax[0,3].errorbar(
        10**cleanLV_K, 10**cleanLogTauK, 
        xerr=[10**np.asarray(cleanLV_K)-10**(np.asarray(cleanLV_K)-np.asarray(cleanLV_K_err)),
            10**(np.asarray(cleanLV_K)+np.asarray(cleanLV_K_err))-10**np.asarray(cleanLV_K)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauK))-10**(np.asarray(cleanLogTauK)-np.asarray(cleanLogTauK_err)),
                10**(np.asarray(cleanLogTauK)+np.asarray(cleanLogTauK_err))-10**np.asarray(cleanLogTauK)],
        ms = 1, color = "green", elinewidth = 0.3, fmt = "s", zorder = 2
        )

    #Plot W2 R-L
    ax[1,0].errorbar(
        10**cleanLV_W2, 10**cleanLogTauW2, 
        xerr=[10**np.asarray(cleanLV_W2)-10**(np.asarray(cleanLV_W2)-np.asarray(cleanLV_W2_err)),
            10**(np.asarray(cleanLV_W2)+np.asarray(cleanLV_W2_err))-10**np.asarray(cleanLV_W2)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauW2))-10**(np.asarray(cleanLogTauW2)-np.asarray(cleanLogTauW2_err)),
                10**(np.asarray(cleanLogTauW2)+np.asarray(cleanLogTauW2_err))-10**np.asarray(cleanLogTauW2)],
        ms = 1, color = "red", elinewidth = 0.3, fmt = "o", zorder = 1
        )
    ax[1,1].errorbar(
        10**cleanLV_W2, 10**cleanLogTauW2_118, 
        xerr=[10**np.asarray(cleanLV_W2)-10**(np.asarray(cleanLV_W2)-np.asarray(cleanLV_W2_err)),
            10**(np.asarray(cleanLV_W2)+np.asarray(cleanLV_W2_err))-10**np.asarray(cleanLV_W2)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauW2_118))-10**(np.asarray(cleanLogTauW2_118)-np.asarray(cleanLogTauW2_118_err)),
                10**(np.asarray(cleanLogTauW2_118)+np.asarray(cleanLogTauW2_118_err))-10**np.asarray(cleanLogTauW2_118)],
        ms = 1, color = "red", elinewidth = 0.3, fmt = "o", zorder = 1
        )
    ax[1,2].errorbar(
        10**cleanLV_W2, 10**cleanLogTauW2_28, 
        xerr=[10**np.asarray(cleanLV_W2)-10**(np.asarray(cleanLV_W2)-np.asarray(cleanLV_W2_err)),
            10**(np.asarray(cleanLV_W2)+np.asarray(cleanLV_W2_err))-10**np.asarray(cleanLV_W2)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauW2_28))-10**(np.asarray(cleanLogTauW2_28)-np.asarray(cleanLogTauW2_28_err)),
                10**(np.asarray(cleanLogTauW2_28)+np.asarray(cleanLogTauW2_28_err))-10**np.asarray(cleanLogTauW2_28)],
        ms = 1, color = "red", elinewidth = 0.3, fmt = "o", zorder = 1
        )

    #Plot W2 R-L with K R-L overplotted
    ax[1,3].errorbar(
        10**cleanLV_W2, 10**cleanLogTauW2, 
        xerr=[10**np.asarray(cleanLV_W2)-10**(np.asarray(cleanLV_W2)-np.asarray(cleanLV_W2_err)),
            10**(np.asarray(cleanLV_W2)+np.asarray(cleanLV_W2_err))-10**np.asarray(cleanLV_W2)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauW2))-10**(np.asarray(cleanLogTauW2)-np.asarray(cleanLogTauW2_err)),
                10**(np.asarray(cleanLogTauW2)+np.asarray(cleanLogTauW2_err))-10**np.asarray(cleanLogTauW2)],
        ms = 1, color = "red", elinewidth = 0.3, fmt = "o", zorder = 1
        )
    ax[1,3].errorbar(
        10**cleanLV_K, 10**cleanLogTauK, 
        xerr=[10**np.asarray(cleanLV_K)-10**(np.asarray(cleanLV_K)-np.asarray(cleanLV_K_err)),
            10**(np.asarray(cleanLV_K)+np.asarray(cleanLV_K_err))-10**np.asarray(cleanLV_K)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauK))-10**(np.asarray(cleanLogTauK)-np.asarray(cleanLogTauK_err)),
                10**(np.asarray(cleanLogTauK)+np.asarray(cleanLogTauK_err))-10**np.asarray(cleanLogTauK)],
        ms = 1, color = "green", elinewidth = 0.3, fmt = "s", zorder = 2
        )

    #Plot K R-L
    ax[0,4].errorbar(
        10**cleanLV_K, 10**cleanLogTauK, 
        xerr=[10**np.asarray(cleanLV_K)-10**(np.asarray(cleanLV_K)-np.asarray(cleanLV_K_err)),
            10**(np.asarray(cleanLV_K)+np.asarray(cleanLV_K_err))-10**np.asarray(cleanLV_K)], 
        yerr = [10**np.asarray(np.asarray(cleanLogTauK))-10**(np.asarray(cleanLogTauK)-np.asarray(cleanLogTauK_err)),
                10**(np.asarray(cleanLogTauK)+np.asarray(cleanLogTauK_err))-10**np.asarray(cleanLogTauK)],
        ms = 1, color = "green", elinewidth = 0.3, fmt = "s", zorder = 1
        )

    #Plot lines for best fit relationships

    def getRLPlot(L_list,intercept,slope = None):
        LPlot = np.logspace(np.min(L_list),np.max(L_list), 100)
        if slope is None: 
            RPlot = 10 ** intercept * LPlot ** 0.5
        else:
            RPlot = 10 ** intercept * LPlot ** slope
        return LPlot, RPlot

    #Plot MPFIT best fit lines
    ax[0,0].plot(getRLPlot(cleanLV_W1,MPFIT_interceptW1)[0],getRLPlot(cleanLV_W1,MPFIT_interceptW1)[1],color = "orange", zorder = 2)
    ax[0,1].plot(getRLPlot(cleanLV_W1,MPFIT_interceptW1_118)[0],getRLPlot(cleanLV_W1,MPFIT_interceptW1_118)[1],color = "orange", zorder = 2)
    ax[0,2].plot(getRLPlot(cleanLV_W1,MPFIT_interceptW1_28)[0],getRLPlot(cleanLV_W1,MPFIT_interceptW1_28)[1],color = "orange", zorder = 2)
    ax[0,3].plot(getRLPlot(np.concatenate((cleanLV_W1,cleanLV_K)),MPFIT_interceptW1)[0],getRLPlot(np.concatenate((cleanLV_W1,cleanLV_K)),MPFIT_interceptW1)[1],color = "orange", zorder = 2)
    ax[0,3].plot(getRLPlot(np.concatenate((cleanLV_W1,cleanLV_K)),MPFIT_interceptK)[0],getRLPlot(np.concatenate((cleanLV_W1,cleanLV_K)),MPFIT_interceptK)[1],color = "black", zorder = 3)

    ax[1,0].plot(getRLPlot(cleanLV_W2,MPFIT_interceptW2)[0],getRLPlot(cleanLV_W2,MPFIT_interceptW2)[1],color = "orange", zorder = 2)
    ax[1,1].plot(getRLPlot(cleanLV_W2,MPFIT_interceptW2_118)[0],getRLPlot(cleanLV_W2,MPFIT_interceptW2_118)[1],color = "orange", zorder = 2)
    ax[1,2].plot(getRLPlot(cleanLV_W2,MPFIT_interceptW2_28)[0],getRLPlot(cleanLV_W2,MPFIT_interceptW2_28)[1],color = "orange", zorder = 2)
    ax[1,3].plot(getRLPlot(np.concatenate((cleanLV_W2,cleanLV_K)),MPFIT_interceptW2)[0],getRLPlot(np.concatenate((cleanLV_W2,cleanLV_K)),MPFIT_interceptW2)[1],color = "orange", zorder = 2)
    ax[1,3].plot(getRLPlot(np.concatenate((cleanLV_W2,cleanLV_K)),MPFIT_interceptK)[0],getRLPlot(np.concatenate((cleanLV_W2,cleanLV_K)),MPFIT_interceptK)[1],color = "black", zorder = 3)

    ax[0,4].plot(getRLPlot(cleanLV_K,MPFIT_interceptK)[0],getRLPlot(cleanLV_K,MPFIT_interceptK)[1],color = "orange", zorder = 2)

    #Plot linmix best fit lines
    if lmTog:
        ax[0,0].plot(getRLPlot(cleanLV_W1,linmix_interceptW1,linmix_slopeW1)[0],getRLPlot(cleanLV_W1,linmix_interceptW1,linmix_slopeW1)[1],color = "mediumorchid", zorder = 3)
        ax[0,1].plot(getRLPlot(cleanLV_W1,linmix_interceptW1_118,linmix_slopeW1_118)[0],getRLPlot(cleanLV_W1,linmix_interceptW1_118,linmix_slopeW1_118)[1],color = "mediumorchid", zorder = 3)
        ax[0,2].plot(getRLPlot(cleanLV_W1,linmix_interceptW1_28,linmix_slopeW1_28)[0],getRLPlot(cleanLV_W1,linmix_interceptW1_28,linmix_slopeW1_28)[1],color = "mediumorchid", zorder = 3)
        ax[0,3].plot(getRLPlot(np.concatenate((cleanLV_W1,cleanLV_K)),linmix_interceptW1,linmix_slopeW1)[0],getRLPlot(np.concatenate((cleanLV_W1,cleanLV_K)),linmix_interceptW1,linmix_slopeW1)[1],color = "mediumorchid", zorder = 3)
        ax[0,3].plot(getRLPlot(np.concatenate((cleanLV_W1,cleanLV_K)),linmix_interceptK,linmix_slopeK)[0],getRLPlot(np.concatenate((cleanLV_W1,cleanLV_K)),linmix_interceptK,linmix_slopeK)[1],color = "darkslategray", ls = "--", zorder = 4)

        ax[1,0].plot(getRLPlot(cleanLV_W2,linmix_interceptW2,linmix_slopeW2)[0],getRLPlot(cleanLV_W2,linmix_interceptW2,linmix_slopeW2)[1],color = "mediumorchid", zorder = 3)
        ax[1,1].plot(getRLPlot(cleanLV_W2,linmix_interceptW2_118,linmix_slopeW2_118)[0],getRLPlot(cleanLV_W2,linmix_interceptW2_118,linmix_slopeW2_118)[1],color = "mediumorchid", zorder = 3)
        ax[1,2].plot(getRLPlot(cleanLV_W2,linmix_interceptW2_28,linmix_slopeW2_28)[0],getRLPlot(cleanLV_W2,linmix_interceptW2_28,linmix_slopeW2_28)[1],color = "mediumorchid", zorder = 3)
        ax[1,3].plot(getRLPlot(np.concatenate((cleanLV_W2,cleanLV_K)),linmix_interceptW2,linmix_slopeW2)[0],getRLPlot(np.concatenate((cleanLV_W2,cleanLV_K)),linmix_interceptW2,linmix_slopeW2)[1],color = "mediumorchid", zorder = 3)
        ax[1,3].plot(getRLPlot(np.concatenate((cleanLV_W2,cleanLV_K)),linmix_interceptK,linmix_slopeK)[0],getRLPlot(np.concatenate((cleanLV_W2,cleanLV_K)),linmix_interceptK,linmix_slopeK)[1],color = "darkslategray", ls = "--", zorder = 4)

        ax[0,4].plot(getRLPlot(cleanLV_K,linmix_interceptK,linmix_slopeK)[0],getRLPlot(cleanLV_K,linmix_interceptK,linmix_slopeK)[1],color = "mediumorchid", zorder = 3)

    ax[0,0].set_xlim(10**np.min(cleanLV_W1),10**np.max(cleanLV_W1))
    ax[0,0].set_ylim(10**np.min(cleanLogTauW1), 10**np.max(cleanLogTauW1))
    ax[0,1].set_xlim(10**np.min(cleanLV_W1),10**np.max(cleanLV_W1))
    ax[0,1].set_ylim(10**np.min(cleanLogTauW1_118), 10**np.max(cleanLogTauW1_118))
    ax[0,2].set_xlim(10**np.min(cleanLV_W1),10**np.max(cleanLV_W1))
    ax[0,2].set_ylim(10**np.min(cleanLogTauW1_28), 10**np.max(cleanLogTauW1_28))
    ax[0,3].set_xlim(10**np.min(np.concatenate((cleanLV_W1,cleanLV_K))),10**np.max(np.concatenate((cleanLV_W1,cleanLV_K))))
    ax[0,3].set_ylim(10**np.min(np.concatenate((cleanLogTauW1,cleanLogTauK))),10**np.max(np.concatenate((cleanLogTauW1,cleanLogTauK))))
    ax[0,4].set_xlim(10**np.min(cleanLV_K),10**np.max(cleanLV_K))
    ax[0,4].set_ylim(10**np.min(cleanLogTauK), 10**np.max(cleanLogTauK))
    
    ax[1,0].set_xlim(10**np.min(cleanLV_W2),10**np.max(cleanLV_W2))
    ax[1,0].set_ylim(10**np.min(cleanLogTauW2), 10**np.max(cleanLogTauW2))
    ax[1,1].set_xlim(10**np.min(cleanLV_W2),10**np.max(cleanLV_W2))
    ax[1,1].set_ylim(10**np.min(cleanLogTauW2_118), 10**np.max(cleanLogTauW2_118))
    ax[1,2].set_xlim(10**np.min(cleanLV_W2),10**np.max(cleanLV_W2))
    ax[1,2].set_ylim(10**np.min(cleanLogTauW2_28), 10**np.max(cleanLogTauW2_28))
    ax[1,3].set_xlim(10**np.min(np.concatenate((cleanLV_W2,cleanLV_K))),10**np.max(np.concatenate((cleanLV_W2,cleanLV_K))))
    ax[1,3].set_ylim(10**np.min(np.concatenate((cleanLogTauW2,cleanLogTauK))),10**np.max(np.concatenate((cleanLogTauW2,cleanLogTauK))))
    
    plt.show()

if __name__ == "__main__": #Prevents runtime error with infinite recursion from LINMIX multiprocessing:
    main()