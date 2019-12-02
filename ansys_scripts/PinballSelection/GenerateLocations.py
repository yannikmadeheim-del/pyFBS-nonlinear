"""
Description:
Saves the information on location of input/output locations in a .csv file within a project directory.


Author: Tomaž Bregar
Date: 01.08.2019
"""

import shutil
import os
import csv

solution_dir = ExtAPI.DataModel.AnalysisList[1].WorkingDir
harmonic_dir = ExtAPI.DataModel.AnalysisList[1].WorkingDir.split("\\")
modal_dir = ExtAPI.DataModel.AnalysisList[0].WorkingDir.split("\\")
res_dir = ""
for i in range(len(harmonic_dir)-5):
    res_dir += harmonic_dir[i] + "/"




impact_csys = []
sensor_csys = []

ham = 1
sen = 1
for CSYS in ExtAPI.DataModel.Project.Model.CoordinateSystems.Children:
    if "H" in CSYS.Name:
        impact_csys.append(CSYS)
        ham += 1
    elif "S" in CSYS.Name:
        if "Global Coordinate System" in CSYS.Name:
            pass
        elif "VP" in CSYS.Name:
            pass
        else:
            sensor_csys.append(CSYS)
            sen += 1

VP_csys = []

VP = 1
for CSYS in ExtAPI.DataModel.Project.Model.CoordinateSystems.Children:
    if "VP" + str(VP) in CSYS.Name:
        VP_csys.append(CSYS)
        VP += 1

# Save location of impacts
impact_data = []
impact_data.append(["Description","X Global", "Unit","Y Global", "Unit","Z Global", "Unit", "Direction X","Direction Y","Direction Z"])
for impact in impact_csys:
    all = [impact.XAxis,impact.YAxis,impact.ZAxis]

    for one in all:
        _X = str(impact.OriginX).split(" ")
        _Y = str(impact.OriginY).split(" ")
        _Z = str(impact.OriginZ).split(" ")

        impact_data.append([impact.Name,_X[0],_X[1],_Y[0],_Y[1],_Z[0],_Z[1],one[0],one[1],one[2]])

try:
    os.remove(res_dir + "Impacts.csv")
except:
    pass

with open(res_dir + "Impacts.csv", "w") as f:
    wr = csv.writer(f, delimiter=";",lineterminator = "\n")
    wr.writerows(impact_data)

# Save location of sensors
sensor_data = []
sensor_data.append(["Description","X Global", "Unit","Y Global", "Unit","Z Global", "Unit", "Direction X","Direction Y","Direction Z"])
for sensor in sensor_csys:
    all = [sensor.XAxis,sensor.YAxis,sensor.ZAxis]

    for one in all:
        _X = str(sensor.OriginX).split(" ")
        _Y = str(sensor.OriginY).split(" ")
        _Z = str(sensor.OriginZ).split(" ")

        sensor_data.append([sensor.Name,_X[0],_X[1],_Y[0],_Y[1],_Z[0],_Z[1],one[0],one[1],one[2]])

try:
    os.remove(res_dir + "Sensors.csv")
except:
    pass

with open(res_dir + "Sensors.csv", "w") as f:
    wr = csv.writer(f, delimiter=";",lineterminator = "\n")
    wr.writerows(sensor_data)


# Save location of VPs
VP_data = []
VP_data.append(["Description","X Global", "Unit","Y Global", "Unit","Z Global", "Unit", "Direction X","Direction Y","Direction Z"])
for VP in VP_csys:
    all = [VP.XAxis,VP.YAxis,VP.ZAxis]

    for one in all:
        _X = str(VP.OriginX).split(" ")
        _Y = str(VP.OriginY).split(" ")
        _Z = str(VP.OriginZ).split(" ")

        VP_data.append([VP.Name,_X[0],_X[1],_Y[0],_Y[1],_Z[0],_Z[1],one[0],one[1],one[2]])

try:
    os.remove(res_dir + "VPs.csv")
except:
    pass

with open(res_dir + "VPs.csv", "w") as f:
    wr = csv.writer(f, delimiter=";",lineterminator = "\n")
    wr.writerows(VP_data)



