from astroquery.ipac.ned import Ned
import time

objID = []
redshifts = []
zOG = []
toggle = 0

if toggle == 0:
    with open("WISEDataOnly.txt", "r") as file:
        lines = file.readlines()
        for i in range(1, len(lines)):
            objID.append(lines[i][8:27])
            zOG.append(lines[i][28:33])
        
    for j in range(0, len(objID)):
        objID[j]="SDSS " + objID[j].strip()

    for obj in objID:
        try:
            result = Ned.query_object(obj)
            rshft = result["Redshift"][0]
            redshifts.append(rshft)
            time.sleep(1)
        except:
            redshifts.append(zOG[objID.index(obj)])

    for z in range(len(redshifts)):
        redshifts[z] = f"{redshifts[z]}\n"

    with open("WISERedshifts.txt", "w") as file:
        file.writelines(redshifts)
