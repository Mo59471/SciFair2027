import math
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

LBol = []
LBol_err = []
Luv = []
Luv_err = []
Rsub1700 = []
Rsub1500 = []

tauW1corr = []
tauW1corr_Err = []
tauW1corr_err = []
tauW2corr = []
tauW2corr_Err = []
tauW2corr_err = []

cleanLuvW1 =[]
cleanLuvW1_err = []
cleanLuvW2 = []
cleanLuvW2_err = []

cleanTauW1 = []
cleanTauW1_err =[]
cleanTauW1_Err = []
cleanTauW2 = []
cleanTauW2_err = []
cleanTauW2_Err = []

offset1700W1 = []
offset1700W2 = []
offset1500W1 = []
offset1500W2 = []
writeOffset = []

with open("WISEData.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        LBol.append(float(line[34:39].strip()))
        LBol_err.append(float(line[40:44].strip()))

for i in range(len(LBol)):
    Luv.append(math.log10(3/5)+LBol[i])
    Luv_err.append(LBol_err[i])

for i in Luv:
    Rsub1500.append(1190.470779 * 0.13 * (1500/1500)**-2.8 * (0.05/0.05)**-0.5 * 10**(0.5 * (i-44)))
    Rsub1700.append(1190.470779 * 0.13 * (1700/1500)**-2.8 * (0.05/0.05)**-0.5 * 10**(0.5 * (i-44)))

W1lags = Path(r"C:\Users\winde\OneDrive\Documents\ISEF2027\KWProp\SampledTauW1.txt")
W2lags = Path(r"C:\Users\winde\OneDrive\Documents\ISEF2027\KWProp\SampledTauW2.txt")

with open(W1lags, "r") as file:
    lines = file.readlines()
    for line in lines:
        if "none" in line:
            tauW1corr.append("none")
            tauW1corr_Err.append("none")
            tauW1corr_err.append("none")
        else:
            tauW1corr.append(float(line[:line.index("|")]))
            tauW1corr_err.append(float(line[line.index("|")+1:line.index("|", line.index("|")+1)]))
            tauW1corr_Err.append(float(line[line.index("|", line.index("|")+1)+1:]))
            cleanTauW1.append(float(line[:line.index("|")]))
            cleanTauW1_err.append(float(line[line.index("|")+1:line.index("|", line.index("|")+1)]))
            cleanTauW1_Err.append(float(line[line.index("|", line.index("|")+1)+1:]))
            with open("WISEData.txt", "r") as file2:
                lines2 = file2.readlines()
                cleanLuvW1.append(float(lines2[lines.index(line)][34:39].strip()) + math.log10(3/5))
                cleanLuvW1_err.append(float(lines2[lines.index(line)][40:44].strip()))

with open(W2lags, "r") as file:
    lines = file.readlines()
    for line in lines:
        if "none" in line:
            tauW2corr.append("none")
            tauW2corr_Err.append("none")
            tauW2corr_err.append("none")
        else:
            tauW2corr.append(float(line[:line.index("|")]))
            tauW2corr_err.append(float(line[line.index("|")+1:line.index("|", line.index("|")+1)]))
            tauW2corr_Err.append(float(line[line.index("|", line.index("|")+1)+1:]))
            cleanTauW2.append(float(line[:line.index("|")]))
            cleanTauW2_err.append(float(line[line.index("|")+1:line.index("|", line.index("|")+1)]))
            cleanTauW2_Err.append(float(line[line.index("|", line.index("|")+1)+1:]))
            with open("WISEData.txt", "r") as file2:
                lines2 = file2.readlines()
                cleanLuvW2.append(float(lines2[lines.index(line)][34:39].strip()) + math.log10(3/5))
                cleanLuvW2_err.append(float(lines2[lines.index(line)][40:44].strip()))
        
for i in range(len(Rsub1700)):
    if tauW1corr[i] != "none":
        offset1700W1.append(tauW1corr[i]/Rsub1700[i])
        offset1500W1.append(tauW1corr[i]/Rsub1500[i])
    if tauW2corr[i] != "none":
        offset1700W2.append(tauW2corr[i]/Rsub1700[i])
        offset1500W2.append(tauW2corr[i]/Rsub1500[i])

with open("SystematicOffset.txt", "w") as file:
    for i in range(len(offset1700W1)):
        try:
            writeOffset.append(f"{offset1700W1[i]}|{offset1700W2[i]}\n")
        except IndexError:
            writeOffset.append(f"{offset1700W1[i]}\n")
    file.writelines(writeOffset)

errTauW1 = np.array([cleanTauW1_err,cleanTauW1_Err])
errTauW2 = np.array([cleanTauW2_err, cleanTauW2_Err]) 

tog = "W1"

plt.yscale('log')
plt.xscale('log')
plt.plot(Luv, Rsub1700, ls = "--", color = "red", linewidth = "3")    
plt.plot(Luv, Rsub1500, ls = "--", color = "red", linewidth = "3")

#Convert log space values to linear space:
errLuvW1 = []
ErrLuvW1 = []
errLuvW2 = []
ErrLuvW2 = []

for i in range(len(cleanLuvW1)):
    errLuvW1.append(10**cleanLuvW1[i]-10**(cleanLuvW1[i]-cleanLuvW1_err[i]))
    ErrLuvW1.append(10**(cleanLuvW1[i]+cleanLuvW1_err[i])-10**cleanLuvW1[i])

for i in range(len(cleanLuvW2)):
    errLuvW2.append(10**cleanLuvW2[i]-10**(cleanLuvW2[i]-cleanLuvW2_err[i]))
    ErrLuvW2.append(10**(cleanLuvW2[i]+cleanLuvW2_err[i])-10**cleanLuvW2[i])

cleanLuvW1 = [10**i for i in cleanLuvW1]
cleanLuvW2 = [10**i for i in cleanLuvW2]

LW1_e = [errLuvW1,ErrLuvW1]
LW2_e = [errLuvW2,ErrLuvW2]

#Get power laws for plotting
x = np.linspace(42.5, 48,100)
y1500 = 1190.470779 * 0.13 * (1500/1500)**-2.8 * (0.05/0.05)**-0.5 * 10**(0.5 * (x-44))
y1700 = 1190.470779 * 0.13 * (1700/1500)**-2.8 * (0.05/0.05)**-0.5 * 10**(0.5 * (x-44))

x = 10**x

plt.xlim(float(10**42.8),float(10**48))
plt.ylim(float(10**0.8), float(10**4))

plt.plot(x, y1500, linewidth = 1, ls = "--", color = "green")
plt.plot(x, y1700, linewidth = 1, ls = "--", color = "red")

if tog == "W1":
    plt.errorbar(cleanLuvW1, cleanTauW1, xerr=LW1_e, yerr =errTauW1, color = "blue", fmt="o",  elinewidth= 0.3, markersize = 1)
elif tog == "W2":
    plt.errorbar(cleanLuvW2, cleanTauW2, xerr=LW2_e, yerr =errTauW2, color = "blue", fmt="o",  elinewidth= 0.3, markersize = 1)

plt.show()

print(statistics.median(offset1700W1), statistics.median(offset1700W2)) #0.7142351169084717 0.7316193222926646
print(statistics.median(offset1500W1), statistics.median(offset1500W2)) #0.503083119142697 0.5153279668986464
