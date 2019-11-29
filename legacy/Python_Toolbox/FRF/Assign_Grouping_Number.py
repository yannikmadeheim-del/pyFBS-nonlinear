# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

Accepts: 
    1. Read UFF data in structured form.
    2. Sensor and Impact data
    
Matches the node number between read uff data nd read excel data and then assigns the grouping number.

# =============================================================================

"""

def assign_grouping_number(sub_structure, sensors , impacts):
          
    for i in range (0,len(sub_structure)):
    
        for j in range (0,len(sensors)):
        
            if (sub_structure[i].DataSets.Y_Channels.Channel_Info.Node_Number == sensors[j].Node_Number):
                
                sub_structure[i].DataSets.Y_Channels.Channel_Info.Grouping = sensors[j].Grouping
                sub_structure[i].DataSets.Y_Channels.Channel_Info.Node.Grouping = sensors[j].Grouping
                    
        
    for i in range (0,len(sub_structure)):

        for j in range (0,len(impacts)):
        
            if (sub_structure[i].DataSets.Y_Channels.Ref_Channel_Info.Node_Number == impacts[j].Node_Number):
                            
                sub_structure[i].DataSets.Y_Channels.Ref_Channel_Info.Grouping = impacts[j].Grouping
                sub_structure[i].DataSets.Y_Channels.Ref_Channel_Info.Node.Grouping = impacts[j].Grouping
                    
    
    return sub_structure