objID = []
SDSS = []
z = []
logLbol = []
logLbolErr = []
logEdd = []
logEddErr = []
rmaxW1 = []
pvalW1 = []
tauW1 = []
e_tauW1 = []
E_tauW1 = []
tauW1ADC = []
e_tauW1ADC = []
E_tauW1ADC = []
tauW1ADCJAV = []
e_tauW1ADCJAV = []
E_tauW1ADCJAV = []
rmaxW2 = []
pvalW2 = []
tauW2 = []
e_tauW2 = []
E_tauW2 = []
tauW2ADC = []
e_tauW2ADC = []
E_tauW2ADC = []
tauW2ADCJAV = []
e_tauW2ADCJAV = []
E_tauW2ADCJAV = []

excelData = []

with open("WISEDataOnly.txt", "r") as file:
    lines = file.readlines()
    for i in range(1, len(lines)):
        objID.append(lines[i][0:7])
        SDSS.append(lines[i][8:27])
        z.append(lines[i][28:33])
        logLbol.append(lines[i][34:39])
        logLbolErr.append(lines[i][40:44])
        logEdd.append(lines[i][46:50])
        logEddErr.append(lines[i][51:55])
        rmaxW1.append(lines[i][56:60])
        pvalW1.append(lines[i][61:68])
        tauW1.append(lines[i][69:73])
        e_tauW1.append(lines[i][74:78])
        E_tauW1.append(lines[i][79:82])
        tauW1ADC.append(lines[i][83:87])
        e_tauW1ADC.append(lines[i][88:91])
        E_tauW1ADC.append(lines[i][92:96])
        tauW1ADCJAV.append(lines[i][97:101])
        e_tauW1ADCJAV.append(lines[i][102:105])
        E_tauW1ADCJAV.append(lines[i][106:109])
        rmaxW2.append(lines[i][110:114])
        pvalW2.append(lines[i][115:121])
        tauW2.append(lines[i][122:126])
        e_tauW2.append(lines[i][127:130])
        E_tauW2.append(lines[i][131:134])
        tauW2ADC.append(lines[i][135:139])
        e_tauW2ADC.append(lines[i][140:143])
        E_tauW2ADC.append(lines[i][144:147])
        tauW2ADCJAV.append(lines[i][148:152])
        e_tauW2ADCJAV.append(lines[i][153:157])
        E_tauW2ADCJAV.append(lines[i][158:162])
    
    objID = [str(item).replace(" ", "") for item in objID]
    SDSS = [str(item).replace(" ", "") for item in SDSS]
    z = [str(item).replace(" ", "") for item in z]
    logLbol = [str(item).replace(" ", "") for item in logLbol]
    logLbolErr = [str(item).replace(" ", "") for item in logLbolErr]
    logEdd = [str(item).replace(" ", "") for item in logEdd]
    logEddErr = [str(item).replace(" ", "") for item in logEddErr]
    rmaxW1 = [str(item).replace(" ", "") for item in rmaxW1]
    pvalW1 = [str(item).replace(" ", "") for item in pvalW1]
    tauW1 = [str(item).replace(" ", "") for item in tauW1]
    e_tauW1 = [str(item).replace(" ", "") for item in e_tauW1]
    E_tauW1 = [str(item).replace(" ", "") for item in E_tauW1]
    tauW1ADC = [str(item).replace(" ", "") for item in tauW1ADC]
    e_tauW1ADC = [str(item).replace(" ", "") for item in e_tauW1ADC]
    E_tauW1ADC = [str(item).replace(" ", "") for item in E_tauW1ADC]
    tauW1ADCJAV = [str(item).replace(" ", "") for item in tauW1ADCJAV]
    e_tauW1ADCJAV = [str(item).replace(" ", "") for item in e_tauW1ADCJAV]
    E_tauW1ADCJAV = [str(item).replace(" ", "") for item in E_tauW1ADCJAV]
    rmaxW2 = [str(item).replace(" ", "") for item in rmaxW2]
    pvalW2 = [str(item).replace(" ", "") for item in pvalW2]
    tauW2 = [str(item).replace(" ", "") for item in tauW2]
    e_tauW2 = [str(item).replace(" ", "") for item in e_tauW2]
    E_tauW2 = [str(item).replace(" ", "") for item in E_tauW2]
    tauW2ADC = [str(item).replace(" ", "") for item in tauW2ADC]
    e_tauW2ADC = [str(item).replace(" ", "") for item in e_tauW2ADC]
    E_tauW2ADC = [str(item).replace(" ", "") for item in E_tauW2ADC]
    tauW2ADCJAV = [str(item).replace(" ", "") for item in tauW2ADCJAV]
    e_tauW2ADCJAV = [str(item).replace(" ", "") for item in e_tauW2ADCJAV]
    E_tauW2ADCJAV = [str(item).replace(" ", "") for item in E_tauW2ADCJAV]

for i in range(len(objID)-1):
    excelData.append(f"{objID[i]}|{SDSS[i]}|{z[i]}|{logLbol[i]}|{logLbolErr[i]}|{logEdd[i]}|{logEddErr[i]}|{rmaxW1[i]}|{pvalW1[i]}|{tauW1[i]}|{e_tauW1[i]}|{E_tauW1[i]}|{tauW1ADC[i]}|{e_tauW1ADC[i]}|{E_tauW1ADC[i]}|{tauW1ADCJAV[i]}|{e_tauW1ADCJAV[i]}|{E_tauW1ADCJAV[i]}|{rmaxW2[i]}|{pvalW2[i]}|{tauW2[i]}|{e_tauW2[i]}|{E_tauW2[i]}|{tauW2ADC[i]}|{e_tauW2ADC[i]}|{E_tauW2ADC[i]}|{tauW2ADCJAV[i]}|{e_tauW2ADCJAV[i]}|{E_tauW2ADCJAV[i]}\n")

with open("WISEDataExcel.txt", "w") as file:
    file.writelines(excelData)
