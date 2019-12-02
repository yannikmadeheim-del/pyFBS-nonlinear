"""
Description: Deletes all Named Selections, Remote Points and APDL snippets.

Author: Tomaž Bregar
Date: 01.08.2019
"""

import os
import shutil

# Delete objects
# Remote Force
try:
    _list = ExtAPI.DataModel.Project.Model.Analyses[1].Children
    for boundary in _list:
        if "Remote" in boundary.Name:
            boundary.Delete()
except:
    print("Error: Can't delete Remote Force")

# Delete old APDL snippets

try:
    all = ExtAPI.DataModel.Project.Model.Analyses[1].Solution.Children.Count
    for i in range(all - 1):
        ExtAPI.DataModel.Project.Model.Analyses[1].Solution.Children[1].Delete()
except:
    print("Error: Can't delete APDL solution snippets")

# number of Remote Points
try:
    RP = ExtAPI.DataModel.Project.Model.RemotePoints.Children
    for remotepoint in RP:
        remotepoint.Delete()
except:
    print("Error: Can't delete Remote Points")
    import os
import shutil

# Delete objects
# Remote Force
try:
    _list = ExtAPI.DataModel.Project.Model.Analyses[1].Children
    for boundary in _list:
        if "Remote" in boundary.Name:
            boundary.Delete()
except:
    print("Error: Can't delete Remote Force")

# Delete old APDL snippets

try:
    all = ExtAPI.DataModel.Project.Model.Analyses[1].Solution.Children.Count
    for i in range(all - 1):
        ExtAPI.DataModel.Project.Model.Analyses[1].Solution.Children[1].Delete()
except:
    print("Error: Can't delete APDL solution snippets")

try:
    NS = ExtAPI.DataModel.Project.Model.NamedSelections.Children
    for namedsel in NS:
        namedsel.Delete()
except:
    print("Error: Can't delete Named selection")

# number of Remote Points
try:
    RP = ExtAPI.DataModel.Project.Model.RemotePoints.Children
    for remotepoint in RP:
        remotepoint.Delete()
except:
    print("Error: Can't delete Remote Points")


