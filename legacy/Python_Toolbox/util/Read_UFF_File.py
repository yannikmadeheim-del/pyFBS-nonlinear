
import pyuff
import re

from Python_Toolbox.Channel.Channel import Channel

def read_uff_file(file_name):
    
    uff_file = pyuff.UFF(file_name)
    data = uff_file.read_sets()
    
#    data = data[2:len(data)]   #Should uncomment this for subsA.unv and subsB.unv files. (Data for tutorial Main_Data_Structure file)
    
    Directions = {0:"None", 1:"+X", 2:"+Y", 3:"+Z", -1:"-X", -2:"-Y", -3:"-Z"}

    class Measurements(object):
        
        Name = None
        DataSets = None
        Geometries = None
        Measurement_Info = None
        
    class Measurement_Info(object):
        
        Date = None
        
    class DataSets(object):
            
        Measurement_Time = None
        Y_Channels = None
        X_Channels = None
    
    class Y_Channels(object):
    
        Name = None
        Data = None
        Description = None
        Quantity = None
        Unit = None
        Channel_Info = None
        Ref_Channel_Info = None
        
    class X_Channels(object):
    
        Name = None
        Data = None
        Description = None
        Quantity = None
        Unit = None
        
        
    Measurement_Data = list()
            
    for i in range (0,len(data)):
        
       #-------------- Class Definitions Start --------------------------------
        
        oMeasurements = Measurements()
        oMeasurement_Info = Measurement_Info()
        oDataSets = DataSets()
        oYChannels = Y_Channels()
        oXChannels = X_Channels()
        oChannelInfo = Channel()
        oRefChannelInfo = Channel()
        oNode_Channel = Channel()
        oNode_Ref_Channel = Channel()
        
        #-------------- Channel Node Info Start ------------------------------------
        
#        oNode_Channel.Node_Number = int(re.findall(r'\d+',data[i]['rsp_ent_name'])[0])   #Should uncomment this for subsA.unv and subsB.unv files. (Data for tutorial Main_Data_Structure file)
        oNode_Channel.Node_Number = data[i]['rsp_node']
        oNode_Channel.Name = 'S' + str(oNode_Channel.Node_Number) + Directions[data[i]['rsp_dir']]
        
        #-------------- Ref Channel Node Info Start ------------------------------------
        
#        oNode_Ref_Channel.Node_Number = int(re.findall(r'\d+',data[i]['ref_ent_name'])[0])  #Should uncomment this for subsA.unv and subsB.unv files. (Data for tutorial Main_Data_Structure file)
        oNode_Ref_Channel.Node_Number = data[i]['ref_node']
        oNode_Ref_Channel.Name = 'Impact' + str(oNode_Ref_Channel.Node_Number)
        
        
        
        #-------------- Channel Info Start ------------------------------------
        
        oChannelInfo.Direction_Number = data[i]['rsp_dir']
        oChannelInfo.Direction_Label = Directions[data[i]['rsp_dir']]       
        oChannelInfo.Name = oNode_Channel.Name
        oChannelInfo.Node_Number = oNode_Channel.Node_Number
        oChannelInfo.Quantity = data[i]['ordinate_axis_lab']
        oChannelInfo.Unit = data[i]['ordinate_axis_units_lab']
        oChannelInfo.Node = oNode_Channel
          
        
        #-------------- Ref Channel Info Start --------------------------------
        
        oRefChannelInfo.Direction_Number = data[i]['ref_dir']
        oRefChannelInfo.Direction_Label = Directions[data[i]['ref_dir']] 
        oRefChannelInfo.Name = oNode_Ref_Channel.Name
        oRefChannelInfo.Node_Number = oNode_Ref_Channel.Node_Number
        oRefChannelInfo.Quantity = data[i]['orddenom_axis_lab']  
        oRefChannelInfo.Unit = data[i]['orddenom_axis_units_lab']
        oRefChannelInfo.Node = oNode_Ref_Channel

        #-------------- Y Channel Start --------------------------------------                                      
        
        oYChannels.Name =  oRefChannelInfo.Name + " " + oChannelInfo.Name
        oYChannels.Data = data[i]['data']
        oYChannels.Quantity = data[i]['ordinate_axis_lab']
        oYChannels.Unit = data[i]['ordinate_axis_units_lab']
        
        oYChannels.Channel_Info = oChannelInfo
        oYChannels.Ref_Channel_Info = oRefChannelInfo
        
        #-------------- X Channel Start -------------------------------------- 
        
        oXChannels.Data = data[i]['x']
        oXChannels.Quantity = data[i]['abscissa_axis_lab']
        oXChannels.Unit = data[i]['abscissa_axis_units_lab']
        
        #Assigning objects to parents classes to created nested class structure
        
        oDataSets.Y_Channels = oYChannels
        oDataSets.X_Channels = oXChannels
        oDataSets.Measurement_Time = data[i]['id3']
        
        oMeasurement_Info.Date = data[i]['id3']
        
        oMeasurements.DataSets = oDataSets
        oMeasurements.Name = data[i]['id1']
        oMeasurements.Measurement_Info = oMeasurement_Info
        
        Measurement_Data.append(oMeasurements)
        
    return Measurement_Data
        