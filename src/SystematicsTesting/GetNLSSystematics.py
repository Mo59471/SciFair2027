import numpy as np
import linmix
import mpfit
import math
import statistics
import matplotlib.pyplot as plt
import random

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

    #Cleaned lists; "none" removed, used for fitting
    cleanLogTauW1 = []
    cleanLogTauW2 = []
    cleanLogTauW1_err = []
    cleanLogTauW2_err = []
    cleanLBolW1 =[]
    cleanLBolW2 = []
    cleanLBolW1_err = []
    cleanLBolW2_err = []

    #BLS residuals
    BLSresW1 = []
    BLSresW2 = []
    BLSresW1_err = []
    BLSresW2_err = []

    #NLS residuals
    NLSresW1 = []
    NLSresW2 = []
    NLSresW1_err = []
    NLSresW2_err = []

    toggle = "MPFIT"

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
            else:
                logLBol.append("none")
                logLBol_err.append("none")
                logTauW1.append("none")
                logTauW2.append("none")
                logTauW1_err.append("none")
                logTauW2_err.append("none")

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

    if toggle == "LINMIX":
        lmW1 = linmix.LinMix(np.array(cleanLBolW1, dtype = float), np.array(cleanLogTauW1, dtype = float), np.array(cleanLBolW1_err, dtype = float), np.array(cleanLogTauW1_err, dtype = float), K=2)
        lmW2 = linmix.LinMix(np.array(cleanLBolW2, dtype = float), np.array(cleanLogTauW2, dtype = float), np.array(cleanLBolW2_err, dtype = float), np.array(cleanLogTauW2_err, dtype = float), K=2)
        lmW1.run_mcmc(silent=True)
        lmW2.run_mcmc(silent=True)
        
        aChainW1 = lmW1.chain['alpha']
        aChainW2 = lmW2.chain['alpha']
        bChainW1 = lmW1.chain['beta']
        bChainW2 = lmW2.chain['beta']
        sigChainW1 = np.sqrt(lmW1.chain['sigsqr'])
        sigChainW2 = np.sqrt(lmW2.chain['sigsqr'])
        
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
        print(statistics.median(sigChainW1))
        print(statistics.median(sigChainW2))
        
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

    for i in range(len(logTauW1)):
        if logTauW1[i] != "none":
            if i in NLSindex:
                NLSresW1.append(logTauW1[i] - (W1intercept + logLBol[i]*W1slope))
                NLSresW1_err.append(math.sqrt((logTauW1_err[i]**2)+(W1slope**2 * logLBol_err[i]**2)))
            else: 
                BLSresW1.append(logTauW1[i] - (W1intercept + logLBol[i]*W1slope))
                BLSresW1_err.append(math.sqrt((logTauW1_err[i]**2)+(W1slope**2 * logLBol_err[i]**2)))
        if logTauW2[i] != "none":
            if i in NLSindex:
                NLSresW2.append(logTauW2[i] - (W2intercept + logLBol[i]*W2slope))
                NLSresW2_err.append(math.sqrt((logTauW2_err[i]**2)+(W2slope**2 * logLBol_err[i]**2)))
            else:
                BLSresW2.append(logTauW2[i] - (W2intercept + logLBol[i]*W2slope))
                BLSresW2_err.append(math.sqrt((logTauW2_err[i]**2)+(W2slope**2 * logLBol_err[i]**2)))

    togAvg = "MEDIAN"
    
    if togAvg == "MEAN":
        #Get error weighted average of the residuals
        num = 0
        denom = 0
        for i in range(len(NLSresW1)):
            num += (NLSresW1[i]/(NLSresW1_err[i]**2))
            denom += (1/(NLSresW1_err[i]**2))
        avgNLSW1 = num/denom
        num = 0
        denom = 0
        for i in range(len(NLSresW2)):
            num += (NLSresW2[i]/(NLSresW2_err[i]**2))
            denom += (1/(NLSresW2_err[i]**2))
        avgNLSW2 = num/denom
        num = 0
        denom = 0
        for i in range(len(BLSresW1)):
            num += (BLSresW1[i]/(BLSresW1_err[i]**2))
            denom +=  (1/(BLSresW1_err[i]**2))
        avgBLSW1 = num/denom
        num = 0
        denom = 0
        for i in range(len(BLSresW2)):
            num +=(BLSresW2[i]/(BLSresW2_err[i]**2))
            denom +=(1/(BLSresW2_err[i]**2))
        avgBLSW2 = num/denom
        
    elif togAvg == "MEDIAN":
        avgBLSW1 = []
        avgBLSW2 = []
        avgNLSW1 = []
        avgNLSW2 = []
        #Calculate bootstrapped median
        for i in range(10000):
            sBLSresW1 = []
            sBLSresW2 = []
            sNLSresW1 = []
            sNLSresW2 = []
            #Assumes that the uncertainty in the residuals (calculated from quadrature addition) follows a lognormal distribution
            while len(sBLSresW1) < len(BLSresW1):
                index = random.randint(0,len(BLSresW1)-1)
                sBLSresW1.append(sampleLogNormal(BLSresW1[index], BLSresW1_err[index]))
            while len(sBLSresW2) < len(BLSresW2):
                index = random.randint(0,len(BLSresW2)-1)
                sBLSresW2.append(sampleLogNormal(BLSresW2[index], BLSresW2_err[index]))
            while len(sNLSresW1) < len(NLSresW1):
                index = random.randint(0,len(NLSresW1)-1)
                sNLSresW1.append(sampleLogNormal(NLSresW1[index], NLSresW1_err[index]))
            while len(sNLSresW2) < len(NLSresW2):
                index = random.randint(0,len(NLSresW2)-1)
                sNLSresW2.append(sampleLogNormal(NLSresW2[index], NLSresW2_err[index]))
            avgBLSW1.append(statistics.median(sBLSresW1))
            avgBLSW2.append(statistics.median(sBLSresW2))
            avgNLSW1.append(statistics.median(sNLSresW1))
            avgNLSW2.append(statistics.median(sNLSresW2))
                
    togPlot = "MEDIANS"
    
    if togPlot == "RESIDUALS":
        fig, (ax1,ax2) = plt.subplots(2,1)

        #Plot residuals of W1 for NLS and bLS
        plotNLSres1,plotBLSres1 = np.split(np.arange(1,len(NLSresW1)+len(BLSresW1)+1,1),[len(NLSresW1)])
        ax1.errorbar(plotNLSres1, NLSresW1, yerr = NLSresW1_err, fmt = "o", color = "red", ecolor = "red", elinewidth= 0.3, markersize = 1)
        ax1.errorbar(plotBLSres1, BLSresW1, yerr = BLSresW1_err, fmt = "o", color = "blue", ecolor = "blue", elinewidth = 0.3, markersize = 1)
        ax1.plot(np.linspace(1,len(NLSresW1)+1, 10), np.ones(10) * statistics.median(avgNLSW1),color = "green", linewidth = 2, linestyle = "--")
        ax1.plot(np.linspace(len(NLSresW1)+1, len(BLSresW1)+len(NLSresW1)+1, 10), np.ones(10) * statistics.median(avgBLSW1), color = "green", linewidth = 2, linestyle = "--")
        
        #Plot residuals of W2 for NLS and BLS
        plotNLSres2,plotBLSres2 = np.split(np.arange(1,len(NLSresW2)+len(BLSresW2)+1,1),[len(NLSresW2)])
        ax2.errorbar(plotNLSres2, NLSresW2, yerr = NLSresW2_err, fmt = "o", color = "red", ecolor = "red", elinewidth= 0.3, markersize = 1)
        ax2.errorbar(plotBLSres2, BLSresW2, yerr = BLSresW2_err, fmt = "o", color = "blue", ecolor = "blue", elinewidth = 0.3, markersize = 1)
        ax2.plot(np.linspace(1,len(NLSresW2)+1, 10), np.ones(10) * statistics.median(avgNLSW2), color = "green", linewidth = 2, linestyle = "--")
        ax2.plot(np.linspace(len(NLSresW2)+1, len(BLSresW2)+len(NLSresW2)+1, 10), np.ones(10) * statistics.median(avgBLSW2), color = "green", linewidth = 2, linestyle = "--")
        
        plt.show()
        
    elif togPlot == "MEDIANS": #Plot distributions of W1 medians from bootstrap-MC for NLS and BLS
        fig, (ax1,ax2) = plt.subplots(1,2)
        histAvgBLSW1, binAvgBLSW1 = np.histogram(avgBLSW1, bins = "auto")
        histAvgBLSW2, binAvgBLSW2 = np.histogram(avgBLSW2, bins = "auto")
        histAvgNLSW1, binAvgNLSW1 = np.histogram(avgNLSW1, bins = "auto")
        histAvgNLSW2, binAvgNLSW2 = np.histogram(avgNLSW2, bins = "auto")
        ax1.stairs(histAvgBLSW1,binAvgBLSW1,edgecolor = "blue")
        ax2.stairs(histAvgBLSW2,binAvgBLSW2,edgecolor = "blue")
        ax1.stairs(histAvgNLSW1,binAvgNLSW1,edgecolor = "red")
        ax2.stairs(histAvgNLSW2,binAvgNLSW2, edgecolor = "red")
        ax1.axvline(x=statistics.median(avgBLSW1), color = "green", linestyle = "--")
        ax1.axvline(x=statistics.median(avgNLSW1), color = "orange", linestyle = "--")
        ax2.axvline(x=statistics.median(avgBLSW2), color = "green", linestyle = "--")
        ax2.axvline(x=statistics.median(avgNLSW2), color = "orange", linestyle = "--")
        
        # Standard (not calculated from sampled distribution of medians) median
        ax1.axvline(x=statistics.median(BLSresW1), color = "green")
        ax1.axvline(x=statistics.median(NLSresW1), color = "orange")
        ax2.axvline(x=statistics.median(BLSresW2), color = "green")
        ax2.axvline(x=statistics.median(NLSresW2), color = "orange")
        
        print(statistics.median(avgNLSW1),statistics.median(avgBLSW1)) #-0.0646182324041876 -0.006536063695465957
        print(statistics.median(avgNLSW2),statistics.median(avgBLSW2)) #-0.08654312491460525 0.0006075060984068434
        
        #Residuals/lags for BLSW1 are systematically higher than NLSW1 by around 14.3% (relative to NLSW1)
        #Residuals/lags for BLSW2 are systematically higher than NLSW2 by around 22.2% (relative to NLSW2)
        
        plt.show()
        
    elif togPlot == "R-L":
        cleanNLSTauW1 = []
        cleanNLSTauW1_err = []
        cleanNLSTauW1_Err = []
        
        cleanLbolW1NLS = []
        cleanLbolW1NLS_err = []
        cleanLbolW1NLS_Err = []
        
        cleanBLSTauW1 = []
        cleanBLSTauW1_err = []
        cleanBLSTauW1_Err = []
    
        cleanLbolW1BLS = []
        cleanLbolW1BLS_err = []
        cleanLbolW1BLS_Err = []
        
        cleanNLSTauW2 = []
        cleanNLSTauW2_err = []
        cleanNLSTauW2_Err = []
        
        cleanLbolW2NLS = []
        cleanLbolW2NLS_err = []
        cleanLbolW2NLS_Err = []
        
        cleanBLSTauW2 = []
        cleanBLSTauW2_err = []
        cleanBLSTauW2_Err = []
        
        cleanLbolW2BLS = []
        cleanLbolW2BLS_err = []
        cleanLbolW2BLS_Err = []

        with open("WISEData.txt", "r") as file:
            lines = file.readlines()
            for i in range(len(lines)):
                LBol = float(lines[i][34:39].strip())
                LBol_err = float(lines[i][40:44].strip())
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
                if i in NLSindex:
                    if W1 != "none":
                        cleanNLSTauW1.append(W1)
                        cleanNLSTauW1_err.append(W1_err)
                        cleanNLSTauW1_Err.append(W1_Err)
                        cleanLbolW1NLS.append(10**LBol)
                        cleanLbolW1NLS_Err.append(10**(LBol+LBol_err)-10**LBol)
                        cleanLbolW1NLS_err.append(10**LBol-10**(LBol-LBol_err))
                    if W2!= "none":
                        cleanNLSTauW2.append(W2)
                        cleanNLSTauW2_err.append(W2_err)
                        cleanNLSTauW2_Err.append(W2_Err)
                        cleanLbolW2NLS.append(10**LBol)
                        cleanLbolW2NLS_Err.append(10**(LBol+LBol_err)-10**LBol)
                        cleanLbolW2NLS_err.append(10**LBol-10**(LBol-LBol_err))
                else:
                    if W1 != "none":
                        cleanBLSTauW1.append(W1)
                        cleanBLSTauW1_err.append(W1_err)
                        cleanBLSTauW1_Err.append(W1_Err)
                        cleanLbolW1BLS.append(10**LBol)
                        cleanLbolW1BLS_Err.append(10**(LBol+LBol_err)-10**LBol)
                        cleanLbolW1BLS_err.append(10**LBol-10**(LBol-LBol_err))
                    if W2!= "none":
                        cleanBLSTauW2.append(W2)
                        cleanBLSTauW2_err.append(W2_err)
                        cleanBLSTauW2_Err.append(W2_Err)
                        cleanLbolW2BLS.append(10**LBol)
                        cleanLbolW2BLS_Err.append(10**(LBol+LBol_err)-10**LBol)
                        cleanLbolW2BLS_err.append(10**LBol-10**(LBol-LBol_err))
                        
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        
        ax1.set_xlim(float(10**42),float(10**48))
        ax1.set_ylim(float(10**0.8), float(10**4))
        ax2.set_xlim(float(10**42),float(10**48))
        ax2.set_ylim(float(10**0.8), float(10**4))
        
        #Set lines for best fit R-L with fixed slope and free slope
        xRL = np.logspace(42,48,100)
        yRLfreeW1= [-15.549830414892707+0.40117746132108323*math.log10(i) for i in xRL]
        yRLfixedW1 = [-20.07051847845333+0.5*math.log10(i) for i in xRL]
        yRLfreeW2= [-12.985627186880887+ 0.34705769626476324*math.log10(i) for i in xRL]
        yRLfixedW2 = [-19.98076311018178+0.5*math.log10(i) for i in xRL]
        
        ax1.errorbar(cleanLbolW1NLS,cleanNLSTauW1,xerr=[cleanLbolW1NLS_err,cleanLbolW1NLS_Err],yerr=[cleanNLSTauW1_err,cleanNLSTauW1_Err],fmt = "o", color = "red", markersize = 1, elinewidth = 0.3)
        ax1.errorbar(cleanLbolW1BLS,cleanBLSTauW1,xerr=[cleanLbolW1BLS_err,cleanLbolW1BLS_Err],yerr=[cleanBLSTauW1_err,cleanBLSTauW1_Err],fmt = "o", color = "blue", markersize = 1, elinewidth = 0.3)
        ax2.errorbar(cleanLbolW2NLS,cleanNLSTauW2,xerr=[cleanLbolW2NLS_err,cleanLbolW2NLS_Err],yerr=[cleanNLSTauW2_err,cleanNLSTauW2_Err],fmt = "o", color = "red", markersize = 1, elinewidth = 0.3)
        ax2.errorbar(cleanLbolW2BLS,cleanBLSTauW2,xerr=[cleanLbolW2BLS_err,cleanLbolW2BLS_Err],yerr=[cleanBLSTauW2_err,cleanBLSTauW2_Err],fmt = "o", color = "blue", markersize = 1, elinewidth = 0.3)
        ax1.plot(xRL, [10**i for i in yRLfreeW1], ls = "--", color = "orange")
        ax1.plot(xRL, [10**i for i in yRLfixedW1], ls = "--", color = "green")
        ax2.plot(xRL, [10**i for i in yRLfreeW2], ls = "--", color = "orange")
        ax2.plot(xRL, [10**i for i in yRLfixedW2], ls = "--", color = "green")
        plt.show()
        
if __name__ == "__main__": #prevents error with infinite recursion with LINMIX multiprocessing:
    main()