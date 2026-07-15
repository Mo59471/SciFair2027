objID = []
coordData = []

with open("MAGNUMCoordsData.txt", "r") as file:
    objID = file.readlines()

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
    else:
        coordData.append(f"{15*(RAH + (RAM/60) + (RAS/3600))}|{DecD + (DecM/60) + (DecS/3600)}\n")

with open("MAGNUMCoordsDeg.txt", "w") as file:
    file.writelines(coordData)
