"""
Description:
Sets the Remote Force at each input location iteratively. The whole iteration can be observed if the Remote Force is
selected in the project tree.


Author: Tomaž Bregar
Date: 01.08.2019
"""

arr_name = ["x","y","z"]
arr = [[1,0,0],[0,1,0],[0,0,1]]

for k,_impact in enumerate(impact_rp):
    _rForce.Location = _impact


    for i in range(3):

        _rForce.XComponent.Output.DiscreteValues = [Quantity(('%10.10f [N]' % arr[i][0]))]
        _rForce.YComponent.Output.DiscreteValues = [Quantity(('%10.10f [N]' % arr[i][1]))]
        _rForce.ZComponent.Output.DiscreteValues = [Quantity(('%10.10f [N]' % arr[i][2]))]
