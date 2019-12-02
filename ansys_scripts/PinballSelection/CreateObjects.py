"""
Description:
Creates a Named Selection and Remote Point object for every output and input together with necessary APDL snippets for
FRF calculation. Inputs should be denoted with a CSYS containing a capital lettter "H" and outputs with a CSY containing
a capital letter "S". The CSYS can be imported from a CAD Plug-In or inserted directly in mechanical.


Author: Tomaž Bregar
Date: 01.08.2019
"""

import os
import shutil

try:
    for NS in ExtAPI.DataModel.Project.Model.NamedSelections.Children:
        if NS.Name == "allFaces":
            NS.Delete()
except:
    pass

# Select all Faces
model = ExtAPI.DataModel.Project.Model
allfaces = model.AddNamedSelection()
allfaces.Name = "allFaces"
selws = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.WorksheetSpecific)
allfaces.Location = selws
faces = allfaces.Location

faces.AddRow()
faces.SetEntityType(0,NamedSelectionWorksheetEntityType.Body)
faces.SetCriterion(0,NamedSelectionWorksheetCriterion.Size)
faces.SetOperator(0,NamedSelectionWorksheetOperator.GreaterThan)
faces.SetValue(0,3.8e-18)

faces.AddRow()
faces.SetAction(1,NamedSelectionWorksheetAction.Convert)
faces.SetEntityType(1,NamedSelectionWorksheetEntityType.Face)
faces.Generate()


# Get impact and sensor CSYS
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

# Impact remote points
n_RP_impact = len(impact_csys)
impact_rp = []
for i in range(n_RP_impact):
    impact_rp.append(ExtAPI.DataModel.Project.Model.RemotePoints.AddRemotePoint())

for _rp,impact in enumerate(impact_rp):
    # Rename each
    impact.Name = "H" + str(_rp + 1)
    # Set PinballRegion
    pinball = 1
    impact.PinballRegion = Quantity(('%10.10f [mm]' % pinball))
    #
    impact.Behavior = LoadBehavior()
    #
    impact.Location = allfaces
    #
    impact.CoordinateSystem = impact_csys[_rp]
    #
    impact.XCoordinate = Quantity(('%10.10f [mm]' % 0))
    impact.YCoordinate = Quantity(('%10.10f [mm]' % 0))
    impact.ZCoordinate = Quantity(('%10.10f [mm]' % 0))


# Sensor remote points + APDL scripts
n_RP_sensor = len(sensor_csys)
sensort_rp = []
for i in range(n_RP_sensor):
    sensort_rp.append(ExtAPI.DataModel.Project.Model.RemotePoints.AddRemotePoint())

for _rp,sensor in enumerate(sensort_rp):
    # Rename each
    sensor.Name = "S" + str(_rp + 1)
    # Set PinballRegion
    pinball = 1
    sensor.PinballRegion = Quantity(('%10.10f [mm]' % pinball))
    #
    sensor.Behavior = LoadBehavior()
    #
    sensor.Location = allfaces
    #
    sensor.CoordinateSystem = sensor_csys[_rp]
    #
    sensor.XCoordinate = Quantity(('%10.10f [mm]' % 0))
    sensor.YCoordinate = Quantity(('%10.10f [mm]' % 0))
    sensor.ZCoordinate = Quantity(('%10.10f [mm]' % 0))

    sensor.AddCommandSnippet()
    sensor.Children[0].Input = "S" + str(_rp + 1) + "= _npilot"


#Add output snippets on the sensors
for i,_sens in enumerate(sensort_rp):
    ExtAPI.DataModel.Project.Model.Analyses[1].Solution.AddCommandSnippet()
    ExtAPI.DataModel.Project.Model.Analyses[1].Solution.Children[i+1].Name = _sens.Name
    ExtAPI.DataModel.Project.Model.Analyses[1].Solution.Children[i+1].Input = "\
fini \n/post1 \n \
set,list \n \
*get ,my_modal_steps, active,0,set,nset \n \
my_modal_steps = my_modal_steps/2 \n \
/DELETE, %s.dat \n \
*CFOPEN ,%s,'dat',,append \n \
*do ,ii_,1,my_modal_steps \n \
set,1,ii_,,REAL \n \
*get,my_freq,ACTIVE,,SET,FREQ \n \
my_u_xR=UX(%s) \n \
my_u_yR=UY(%s) \n \
my_u_zR=UZ(%s) \n \
set,1,ii_,,IMAG \n \
*get,my_freq,ACTIVE,,SET,FREQ \n \
my_u_xI=UX(%s) \n \
my_u_yI=UY(%s) \n \
my_u_zI=UZ(%s) \n \
*VWRITE,my_freq,my_u_xR,my_u_yR,my_u_zR,my_u_xI,my_u_yI,my_u_zI \n \
(7F20.15) \n \
*enddo \n \
*CFCLOS \n" % (_sens.Name,_sens.Name,_sens.Name,_sens.Name,_sens.Name,_sens.Name,_sens.Name,_sens.Name)


# Add remote force
_rForce = ExtAPI.DataModel.Project.Model.Analyses[1].AddRemoteForce()
_rForce.DefineBy = _rForce.DefineBy.Components
_rForce.Behavior = _rForce.Behavior.Rigid

#_rForce.Location = RP
impact = []
impact_object = []

number_RP = ExtAPI.DataModel.Project.Model.RemotePoints.Children.Count

for i in range(number_RP):
    _name = ExtAPI.DataModel.Project.Model.RemotePoints.Children[i].Name
    if "H" in _name:
        impact.append([i, _name])
        impact_object.append(ExtAPI.DataModel.Project.Model.RemotePoints.Children[i])
        #impact_object[i].Suppressed = True

print("--------------------------------------------------------")
print("Created %s Remote Points for Impacts" % n_RP_impact)
print("Created %s Remote Points for Sensors" % n_RP_sensor)
print("Created Remote Force")
print("--------------------------------------------------------")
