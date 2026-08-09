import numpy as np
import linmix
import math
import statistics
import random
import matplotlib.pyplot as plt
import mpfit

def main():
    
    #Used for hybrid bootstrap-Monte Carlo later on to estimate uncertainty in median
    def sampleLogNormal(mode, sigma):
        return random.gauss(0, sigma)+mode 
    
    objID = []
    coordData = []
    RADec = []
    narrowLineSeyferts = []
    WISEnls = []
    NLSindex = []

    #WISE Data
    logLBol = []
    logLBol_err = []
    logTauW1 = []
    logTauW1_err = []
    logTauW2 = []
    logTauW2_err = []
    logEdd = []
    logEdd_err = []
    eddW1NLS = []
    eddW1NLS_err = []
    eddW2NLS = []
    eddW2NLS_err = []
    eddW1BLS = []
    eddW1BLS_err = []
    eddW2BLS = []
    eddW2BLS_err = []

    #Cleaned lists; "none" removed, used for fitting
    cleanLogTauW1 = []
    cleanLogTauW2 = []
    cleanLogTauW1_err = []
    cleanLogTauW2_err = []
    cleanLBolW1 =[]
    cleanLBolW2 = []
    cleanLBolW1_err = []
    cleanLBolW2_err = []

    #Residuals
    resW1NLS = []
    resW1NLS_err = []
    resW2NLS = []
    resW2NLS_err = []
    resW1BLS = []
    resW1BLS_err = []
    resW2BLS = []
    resW2BLS_err = []
    
    with open("WISEData.txt", "r") as file:
        lines = file.readlines()
        for i in range(len(lines)):
            try:
                W1 = float(lines[i][97:101].strip())
                W1_Err = float(lines[i][106:109].strip())
                W1_err = float(lines[i][102:105].strip())
            except:
                W1 = "none"
                W1_Err = "none"
                W1_err = "none"
            try:
                W2 = float(lines[i][148:152].strip())
                W2_Err = float(lines[i][158:162].strip())
                W2_err = float(lines[i][153:157].strip())
            except: 
                W2 = "none"
                W2_err = "none"
                W2_Err = "none"
            
            if W1 != "none" and W2 != "none":
                logLBol.append(float(lines[i][34:39].strip()))
                logLBol_err.append(float(lines[i][40:44].strip()))
                cleanLBolW1.append(float(lines[i][34:39].strip()))
                cleanLBolW2.append(float(lines[i][34:39].strip()))
                cleanLBolW1_err.append(float(lines[i][40:44].strip()))
                cleanLBolW2_err.append(float(lines[i][40:44].strip()))
                logTauW1.append(math.log10(W1))
                logTauW2.append(math.log10(W2))
                logTauW1_err.append((math.log10(W1+W1_Err)-math.log10(W1-W1_err))/2)
                logTauW2_err.append((math.log10(W2+W2_Err)-math.log10(W2-W2_err))/2)
                cleanLogTauW1.append(math.log10(W1))
                cleanLogTauW2.append(math.log10(W2))
                cleanLogTauW1_err.append((math.log10(W1+W1_Err)-math.log10(W1-W1_err))/2)
                cleanLogTauW2_err.append((math.log10(W2+W2_Err)-math.log10(W2-W2_err))/2)
                if lines[i][45:50].strip() != "":
                    logEdd.append(float(lines[i][45:50].strip()))
                    logEdd_err.append(float(lines[i][51:55].strip()))
                else:
                    logEdd.append("none")
                    logEdd_err.append("none")
            elif W1 != "none":
                logLBol.append(float(lines[i][34:39].strip()))
                logLBol_err.append(float(lines[i][40:44].strip()))
                cleanLBolW1.append(float(lines[i][34:39].strip()))
                cleanLBolW1_err.append(float(lines[i][40:44].strip()))
                logTauW1.append(math.log10(W1))
                cleanLogTauW1.append(math.log10(W1))
                logTauW2.append("none")
                logTauW1_err.append((math.log10(W1+W1_Err)-math.log10(W1-W1_err))/2)
                cleanLogTauW1_err.append((math.log10(W1+W1_Err)-math.log10(W1-W1_err))/2)
                logTauW2_err.append("none")
                if lines[i][45:50].strip() != "":
                    logEdd.append(float(lines[i][45:50].strip()))
                    logEdd_err.append(float(lines[i][51:55].strip()))
                else:
                    logEdd.append("none")
                    logEdd_err.append("none")
            elif W2 != "none":
                logLBol.append(float(lines[i][34:39].strip()))
                logLBol_err.append(float(lines[i][40:44].strip()))
                cleanLBolW2.append(float(lines[i][34:39].strip()))
                cleanLBolW2_err.append(float(lines[i][40:44].strip()))
                logTauW1.append("none")
                logTauW2.append(math.log10(W2))
                cleanLogTauW2.append(math.log10(W2))
                logTauW1_err.append("none")
                logTauW2_err.append((math.log10(W2+W2_Err)-math.log10(W2-W2_err))/2)
                cleanLogTauW2_err.append((math.log10(W2+W2_Err)-math.log10(W2-W2_err))/2)
                if lines[i][45:50].strip() != "":
                    logEdd.append(float(lines[i][45:50].strip()))
                    logEdd_err.append(float(lines[i][51:55].strip()))
                else:
                    logEdd.append("none")
                    logEdd_err.append("none")
            else:
                logLBol.append("none")
                logLBol_err.append("none")
                logTauW1.append("none")
                logTauW2.append("none")
                logTauW1_err.append("none")
                logTauW2_err.append("none")
                if lines[i][45:50].strip() != "":
                    logEdd.append(float(lines[i][45:50].strip()))
                    logEdd_err.append(float(lines[i][51:55].strip()))
                else:
                    logEdd.append("none")
                    logEdd_err.append("none")

    # GET NARROW LINE SEYFERTS IN THE WISE SAMPLE
    with open("SDSSIDs.txt", "r") as file:
        objID = file.readlines()
        for i in range(len(objID)):
            objID[i] = objID[i].strip()

    for obj in objID:
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
        
        if "-" in obj:
            coordData.append(f"{15*(RAH + (RAM/60) + (RAS/3600))}|-{DecD + (DecM/60) + (DecS/3600)}\n")
            RADec.append([15*(RAH + (RAM/60) + (RAS/3600)), -1*(DecD + (DecM/60) + (DecS/3600))])
        else:
            coordData.append(f"{15*(RAH + (RAM/60) + (RAS/3600))}|{DecD + (DecM/60) + (DecS/3600)}\n")
            RADec.append([15*(RAH + (RAM/60) + (RAS/3600)), (DecD + (DecM/60) + (DecS/3600))])

    with open("WISECoordsData.txt", "w") as file:
        file.writelines(coordData)
    
    with open("NarrowLineSeyferts.txt", "r") as file:
        lines = file.readlines()
        for i in range(38,len(lines)):
            narrowLineSeyferts.append([lines[i][16:25],lines[i][26:35]])
        for i in RADec:
            for j in narrowLineSeyferts:
                if round(float(i[0]),3) == round(float(j[0]),3) and round(float(i[1]), 3) == round(float(j[1]),3):
                    WISEnls.append(objID[RADec.index(i)])
                    print(RADec.index(i))
                    NLSindex.append(RADec.index(i))
                    
    toggle = "MPFIT"
    
    if toggle == "LINMIX":
        lmW1 = linmix.LinMix(np.array(cleanLBolW1, dtype = float), np.array(cleanLogTauW1, dtype = float), np.array(cleanLBolW1_err, dtype = float), np.array(cleanLogTauW1_err, dtype = float), K=2)
        lmW2 = linmix.LinMix(np.array(cleanLBolW2, dtype = float), np.array(cleanLogTauW2, dtype = float), np.array(cleanLBolW2_err, dtype = float), np.array(cleanLogTauW2_err, dtype = float), K=2)
        lmW1.run_mcmc(silent=True)
        lmW2.run_mcmc(silent=True)
        
        aChainW1 = lmW1.chain['alpha']
        aChainW2 = lmW2.chain['alpha']
        bChainW1 = lmW1.chain['beta']
        bChainW2 = lmW2.chain['beta']
        
        W1intercept = np.median(aChainW1)
        W2intercept = np.median(aChainW2)
        W1slope = np.median(bChainW1)
        W2slope = np.median(bChainW2)
        
        #Compute errors; find interquantile range between lower 15.9% and 84.1% quantiles
        W1intercept_err = np.percentile(aChainW1, 84.1)-np.percentile(aChainW1, 15.9)
        W2intercept_err = np.percentile(aChainW2, 84.1)-np.percentile(aChainW2, 15.9)
        W1slope_err = np.percentile(bChainW1, 84.1)-np.percentile(bChainW1, 15.9)
        W2slope_err = np.percentile(bChainW2, 84.1)-np.percentile(bChainW2, 15.9)
        
        #Compute covariances
        W1Cov = np.cov(aChainW1, bChainW1, bias = True)[0,1]
        W2Cov = np.cov(aChainW2, bChainW2, bias = True)[0,1]
    
    elif toggle == "MPFIT":
        # FIT THE R-Lbol RELATIONSHIP FOR WISE AND EVALUATE RESIDUALS
        #p: array of parameters to fit (slope and intercept for straight line)
        def linefit(p, fjac = None, x = None, y = None, err = None):
            model = p[0] + 0.5* x #modelY = a +bx; slope fixed at 0.5
            status = 0
            return [status, (y-model)/err] #returns the residuals: measured values - current iteration model values (weighted by uncertainty); these are minimized

        #Initial guesses (MPFIT iterates from these); mostly arbitrary
        p0 = [0.0]
        
        effectiveErrW1 = []
        effectiveErrW2 = []
        
        for i in range(len(cleanLogTauW1_err)): #intScatter**2
            effectiveErrW1.append(math.sqrt(cleanLogTauW1_err[i]**2+cleanLBolW1_err[i]**2+0.2**2)) #For an effective error used to weight, combine luminosity error, time lag error, and intrinsic scatter
        for i in range(len(cleanLogTauW2_err)):
            effectiveErrW2.append(math.sqrt(cleanLogTauW2_err[i]**2+cleanLBolW2_err[i]**2+0.2**2))
            
        mW1= mpfit.mpfit(
            linefit, #repeatedly called and iterated
            p0, #Initial guesses
            functkw = {"x":np.array(cleanLBolW1, dtype=float), "y":np.array(cleanLogTauW1),"err":np.array(effectiveErrW1)} #Dictionary: passes in x, y, and err into linefit
            )

        mW2= mpfit.mpfit(
            linefit, #repeatedly called and iterated
            p0, #Initial guesses
            functkw = {"x":np.array(cleanLBolW2, dtype= float), "y":np.array(cleanLogTauW2),"err":np.array(effectiveErrW2)} #Dictionary: passes in x, y, and err into linefit
            )

        W1intercept = float(mW1.params[0])
        W1intercept_err = float(mW1.perror[0])
        W1slope = 0.5

        W2intercept = float(mW2.params[0])
        W2intercept_err = float(mW2.perror[0])
        W2slope = 0.5
        
        print(W1intercept,W2intercept)
    
    print(W1intercept,W1slope)
    print(W2intercept,W2slope)
    
    for i in range(len(logTauW1)):
        if logTauW1[i] != "none":
            if logEdd[i] != "none":
                if i in NLSindex:
                    resW1NLS.append(logTauW1[i] - (W1intercept + logLBol[i]*W1slope))
                    resW1NLS_err.append(math.sqrt((logTauW1_err[i]**2)+(W1slope**2 * logLBol_err[i]**2)))
                    eddW1NLS.append(logEdd[i])
                    eddW1NLS_err.append(logEdd_err[i])
                else:
                    resW1BLS.append(logTauW1[i] - (W1intercept + logLBol[i]*W1slope))
                    resW1BLS_err.append(math.sqrt((logTauW1_err[i]**2)+(W1slope**2 * logLBol_err[i]**2)))
                    eddW1BLS.append(logEdd[i])
                    eddW1BLS_err.append(logEdd_err[i])
        if logTauW2[i] != "none":
            if logEdd[i] != "none":
                if i in NLSindex:
                    resW2NLS.append(logTauW2[i] - (W2intercept + logLBol[i]*W2slope))
                    resW2NLS_err.append(math.sqrt((logTauW2_err[i]**2)+(W2slope**2 * logLBol_err[i]**2)))
                    eddW2NLS.append(logEdd[i])
                    eddW2NLS_err.append(logEdd_err[i])
                else:
                    resW2BLS.append(logTauW2[i] - (W2intercept + logLBol[i]*W2slope))
                    resW2BLS_err.append(math.sqrt((logTauW2_err[i]**2)+(W2slope**2 * logLBol_err[i]**2)))
                    eddW2BLS.append(logEdd[i])
                    eddW2BLS_err.append(logEdd_err[i])    
    
    medNLSeddW1 = statistics.median([10**i for i in eddW1NLS])
    medNLSeddW2 = statistics.median([10**i for i in eddW2NLS])
    medBLSeddW1 = statistics.median([10**i for i in eddW1BLS])
    medBLSeddW2 = statistics.median([10**i for i in eddW2BLS])
    
    lmResW1 = linmix.LinMix(np.concatenate((eddW1BLS, eddW1NLS), dtype=float), np.concatenate((resW1BLS,resW1NLS), dtype = float), np.concatenate((eddW1BLS_err,eddW1NLS_err), dtype = float), np.concatenate((resW1BLS_err,resW1NLS_err)), K=2)
    lmResW2 = linmix.LinMix(np.concatenate((eddW2BLS, eddW2NLS), dtype=float), np.concatenate((resW2BLS,resW2NLS), dtype = float), np.concatenate((eddW2BLS_err,eddW2NLS_err), dtype = float), np.concatenate((resW2BLS_err,resW2NLS_err)), K=2)
    lmResW1.run_mcmc(silent=True)
    lmResW2.run_mcmc(silent=True)
    
    aResW1 = lmResW1.chain['alpha']
    aResW2 = lmResW2.chain['alpha']
    bResW1 = lmResW1.chain['beta']
    bResW2 = lmResW2.chain['beta']
    
    ResW1intercept = np.median(aResW1)
    ResW2intercept = np.median(aResW2)
    ResW1slope = np.median(bResW1)
    ResW2slope = np.median(bResW2)
    
    fitResNLSW1 = ResW1intercept + ResW1slope*medNLSeddW1
    fitResNLSW2 = ResW2intercept + ResW2slope*medNLSeddW2
    fitResBLSW1 = ResW1intercept + ResW1slope*medBLSeddW1
    fitResBLSW2 = ResW2intercept + ResW2slope*medBLSeddW2
    
    print(fitResNLSW1, fitResBLSW1) #-0.10307872403861743 -0.08086723018889587 
    print(fitResNLSW2, fitResBLSW2) #-0.09030995063071218 -0.06970206199048677
    
    #Residuals/time lags at the BLSW1 median Edd ratio are systematically lower than NLSW1 median by around 5.25% (relative to NLSW1)
    #Residuals/time lags at the BLSW2 median Edd ratio are systematically lower than NLSW2 median by around 4.86% (relative to NLSW2)
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_xscale("log")
    ax2.set_xscale("log")

    #Get linear scale Eddington ratios
    linEddW1NLS_err = []
    linEddW1NLS_Err = []
    linEddW1BLS_err = []
    linEddW1BLS_Err = []
    linEddW2NLS_err = []
    linEddW2NLS_Err = []
    linEddW2BLS_err = []
    linEddW2BLS_Err = []
    
    for i in range(len(eddW1NLS_err)):
        linEddW1NLS_err.append(10**(eddW1NLS[i])-10**(eddW1NLS[i] - eddW1NLS_err[i]))
        linEddW1NLS_Err.append((10**(eddW1NLS[i]+eddW1NLS_err[i]))-10**eddW1NLS[i])
    for i in range(len(eddW1BLS_err)):
        linEddW1BLS_err.append(10**(eddW1BLS[i])-10**(eddW1BLS[i] - eddW1BLS_err[i]))
        linEddW1BLS_Err.append((10**(eddW1BLS[i]+eddW1BLS_err[i]))-10**eddW1BLS[i])
    for i in range(len(eddW2NLS_err)):
        linEddW2NLS_err.append(10**(eddW2NLS[i])-10**(eddW2NLS[i] - eddW2NLS_err[i]))
        linEddW2NLS_Err.append((10**(eddW2NLS[i]+eddW2NLS_err[i]))-10**eddW2NLS[i])
    for i in range(len(eddW2BLS_err)):
        linEddW2BLS_err.append(10**(eddW2BLS[i])-10**(eddW2BLS[i] - eddW2BLS_err[i]))
        linEddW2BLS_Err.append((10**(eddW2BLS[i]+eddW2BLS_err[i]))-10**eddW2BLS[i])
    
    ax1.errorbar([10**i for i in eddW1BLS], resW1BLS, xerr = [linEddW1BLS_err, linEddW1BLS_Err], yerr = resW1BLS_err, color = "blue", markersize = 1.5, elinewidth=0.3, fmt = "o")
    ax1.errorbar([10**i for i in eddW1NLS], resW1NLS, xerr = [linEddW1NLS_err, linEddW1NLS_Err], yerr = resW1NLS_err, color = "red", markersize = 1.5, elinewidth=0.3, fmt = "o")
    ax2.errorbar([10**i for i in eddW2BLS], resW2BLS, xerr = [linEddW2BLS_err, linEddW2BLS_Err], yerr = resW2BLS_err, color = "blue", markersize = 1.5, elinewidth=0.3, fmt = "o")
    ax2.errorbar([10**i for i in eddW2NLS], resW2NLS, xerr = [linEddW2NLS_err, linEddW2NLS_Err], yerr = resW2NLS_err, color = "red", markersize = 1.5, elinewidth=0.3, fmt = "o")    
    
    ax1.axvline(x = medNLSeddW1, color = "orange", ls = "--")
    ax1.axvline(x = medBLSeddW1, color = "green", ls = "--")
    ax2.axvline(x = medNLSeddW2, color = "orange", ls = "--")
    ax2.axvline(x = medBLSeddW2, color = "green", ls = "--")
    
    #Best fit line plotting
    xLine = np.logspace(-3,1,100)
    yW1 = ResW1intercept+ResW1slope*(np.array([math.log10(i) for i in xLine]))
    yW2 = ResW2intercept+ResW2slope*(np.array([math.log10(i) for i in xLine]))
    ax1.plot(xLine, yW1, color = "black")
    ax2.plot(xLine, yW2, color = "black")
    plt.show()
    
if __name__ == "__main__":
    main()
    