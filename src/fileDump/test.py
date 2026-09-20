import csv
import random
import scipy
import statistics
import csv

medianW2 = []
W2err = []
W2Err = []
modeW2 = []

medianW1 = []
W1err = []
W1Err= []
modeW1 = []

rat = []
ratio = []
writeRatio = []
ratMedians = []

with open("RawWISELags.csv","r") as file:
    reader = csv.reader(file)
    data = list(reader)
    data = data[1:]

for i in data:
    if int(float(i[6])) != 0 and int(float(i[12])) != 0:
        medianW1.append(float(i[6]))
        W1err.append(float(i[7]))
        W1Err.append(float(i[8]))
        if float(i[7]) > float(i[8]):
            modeW1.append(float(i[6])-float(i[7]) * scipy.stats.norm.ppf((float(i[7])+float(i[8]))/(4*float(i[7]))))
        elif float(i[7]) < float(i[8]):
            modeW1.append(float(i[6])-float(i[8])*scipy.stats.norm.ppf(1-((float(i[7])+float(i[8]))/(4*float(i[8])))))
        elif float(i[7]) == float(i[8]):
            modeW1.append(float(i[6]))         
            
        medianW2.append(float(i[12]))
        W2err.append(float(i[13]))
        W2Err.append(float(i[14]))
        if float(i[13]) > float(i[14]):     
            modeW2.append(float(i[12])-float(i[13]) * scipy.stats.norm.ppf((float(i[13])+float(i[14]))/(4*float(i[13]))))
        elif float(i[13]) < float(i[8]):
            modeW2.append(float(i[12])-float(i[14])*scipy.stats.norm.ppf(1-((float(i[13])+float(i[14]))/(4*float(i[14])))))
        elif float(i[7]) == float(i[8]):
            modeW2.append(float(i[12]))
            
    