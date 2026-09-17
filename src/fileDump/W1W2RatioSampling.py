import random
import scipy
import statistics
import csv
import math

medianW2 = []
W2err = []
W2Err = []
modeW2 = []

medianW1 = []
W1err = []
W1Err= []
modeW1 = []

ratio = []
writeRatio = []
medDistr = []
gammas = []
medGammas = []

# Read CSV file into list
with open("RawWISELags.csv","r") as file:
    reader = csv.reader(file)
    data = list(reader)
    data = data[1:]

for i in data:
    if int(float(i[6])) != 0 and int(float(i[12])) != 0: #Check to see if there is a W1 AND W2 lag for an object
        medianW1.append(float(i[6]))
        W1err.append(float(i[7]))
        W1Err.append(float(i[8]))
        if float(i[7]) > float(i[8]):
            modeW1.append(float(i[6])-float(i[7]) * scipy.stats.norm.ppf((float(i[7])+float(i[8]))/(4*float(i[7])))) #Calculate mode from median based on skew
        elif float(i[7]) < float(i[8]):
            modeW1.append(float(i[6])-float(i[8])*scipy.stats.norm.ppf(1-((float(i[7])+float(i[8]))/(4*float(i[8])))))
        elif float(i[7]) == float(i[8]):
            modeW1.append(float(i[6]))         
            
        medianW2.append(float(i[12]))
        W2err.append(float(i[13]))
        W2Err.append(float(i[14]))
        if float(i[13]) > float(i[14]):     
            modeW2.append(float(i[12])-float(i[13]) * scipy.stats.norm.ppf((float(i[13])+float(i[14]))/(4*float(i[13]))))
        elif float(i[13]) < float(i[14]):
            modeW2.append(float(i[12])-float(i[14])*scipy.stats.norm.ppf(1-((float(i[13])+float(i[14]))/(4*float(i[14])))))
        elif float(i[13]) == float(i[14]):
            modeW2.append(float(i[12]))
    else: #Appends none if there is no value for one or more of the fields
        medianW1.append("none")
        W1err.append("none")
        W1Err.append("none")
        modeW1.append("none")
        medianW2.append("none")
        W2err.append("none")
        W2Err.append("none")
        modeW2.append("none")                

# Assumes W1 and W2 are independent, not strictly true but approximately so
def sampleSplitNormal(mode, sigmaL, sigmaR):
    pLeft = sigmaL/(sigmaL+sigmaR) #Define cumulative probability of LH side
    if random.uniform(0,1) <= pLeft: #random float between 0-1, inclusive; if less than prob of LH side -> falls in LH side
        return -1*abs(random.gauss(0, sigmaL))+mode #If on left hand side, sample from LH Gaussian
    else:
        return abs(random.gauss(0,sigmaR))+mode #If on right hand side, sample from RH Gaussian

for j in range(len(modeW1)):
    rat = []
    gam = []
    if modeW1[j] != "none":  
        for i in range(10000): #Sample 10,000 times from the split Gaussian for each object
            r = sampleSplitNormal(modeW2[j], W2err[j], W2Err[j])/sampleSplitNormal(modeW1[j],W1err[j],W1Err[j])
            while r <= 0:
                r = sampleSplitNormal(modeW2[j], W2err[j], W2Err[j])/sampleSplitNormal(modeW1[j],W1err[j],W1Err[j]) 
            rat.append(r)
            gam.append(math.log10(r)/math.log10(4.6/3.4)) #Find the power laws corresponding to the given ratio; uses WISE nominal wavelengths W1 = 3.4 and W2 = 4.6
    else:
        rat = ["none"]
        gam = ["none"]
    ratio.append(rat) #Append each object's ratio distribution lists to a list
    gammas.append(gam)

for i in ratio:
    if i[0] != "none":
        writeRatio.append(str(statistics.median(i))+"\n")
        medDistr.append(statistics.median(i)) #Calculate median ratios for each object
    else:
        writeRatio.append("none \n")

for i in gammas:
    if i[0] != "none":
        medGammas.append(statistics.median(i))

with open("W1W2Ratio.txt", "w") as file:
    file.writelines(writeRatio)

print(statistics.median(medDistr))
print(statistics.median(medGammas))
