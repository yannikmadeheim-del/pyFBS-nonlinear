"""
Description:
Solves for each input and saves the FRFs from all output locations to a folder within project directory
(*/project_director/FRF/...).


Author: Tomaž Bregar
Date: 01.08.2019
"""

import os
import shutil

# Solves for all sensor locations and impact locations
# Directories
solution_dir = ExtAPI.DataModel.AnalysisList[1].WorkingDir
harmonic_dir = ExtAPI.DataModel.AnalysisList[1].WorkingDir.split("\\")
modal_dir = ExtAPI.DataModel.AnalysisList[0].WorkingDir.split("\\")
res_dir = ""
for i in range(len(harmonic_dir)-5):
    res_dir += harmonic_dir[i] + "/"
res_dir += "FRF"
try:
    shutil.rmtree(res_dir)
except:
    pass
os.mkdir(res_dir)
res_dir += "/"



arr_name = ["x","y","z"]
arr = [[1,0,0],[0,1,0],[0,0,1]]

for k,_impact in enumerate(impact_rp):
    _rForce.Location = _impact


    for i in range(3):

        workdir = res_dir + _impact.Name + "_" + arr_name[i]
        os.mkdir(workdir)

        #Set the Remote Force C-SYS
        _rForce.XComponent.Output.DiscreteValues = [Quantity(('%10.10f [N]' % arr[i][0]))]
        _rForce.YComponent.Output.DiscreteValues = [Quantity(('%10.10f [N]' % arr[i][1]))]
        _rForce.ZComponent.Output.DiscreteValues = [Quantity(('%10.10f [N]' % arr[i][2]))]

        #Solve Harmonic
        ExtAPI.DataModel.Project.Model.Analyses[1].Solve(True)

        #Move the results
        for p,sensor in enumerate(sensort_rp):
            os.rename(solution_dir + sensor.Name + ".dat", workdir + "/" + sensor.Name + ".dat")



