import pyuff
from pyfbs.io import Quantity

#TODO: Clean the code class instance within function?!
#TODO: introduce the same unit/quantity class from the io.py!!!

def read_uff_file(file_name):
    """
    Description

    :param file_name:
    :return:
    """
    uff_file = pyuff.UFF(file_name)
    data = uff_file.read_sets()

    #    data = data[2:len(data)]   #Should uncomment this for subsA.unv and subsB.unv files. (Data for tutorial Main_Data_Structure file)

    Directions = {0: "None", 1: "+X", 2: "+Y", 3: "+Z", -1: "-X", -2: "-Y", -3: "-Z"}

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
        Quantity = Quantity()
        Unit = None
        Channel_Info = None
        Ref_Channel_Info = None

    class X_Channels(object):
        Name = None
        Data = None
        Description = None
        Quantity = Quantity()
        Unit = None

    class Channel(object):
        """
        Description

        Attributes:
            Quantity
            Dim_Vector
            dBref
            Notes
            Type
            Direction
            Label
            DOF_Label
            Direction_Label
            Direction_Number
            Node
            Component
            Node_Number
            Grouping
            Position
            Name
            Description
        """
        def __init__(self):
            self.Quantity = Quantity()
            self.Dim_Vector = None
            self.dBref = None
            self.Notes = None
            self.Type = None
            self.Direction = None
            self.Label = None
            self.DOF_Label = None
            self.Direction_Label = None
            self.Direction_Number = None
            self.Node = None
            self.Component = None
            self.Node_Number = None
            self.Grouping = None
            self.Position = None
            self.Name = None
            self.Description = None



        def fromXls(self,excel_file, excel_sheet):
            """
            Description

            :param excel_file:
            :param excel_sheet:
            :return:
            """
            data = pd.read_excel(excel_file, excel_sheet)


        def select(self,Data, Grouping=None, Name=None, Quantity=None):
            """
            Description

            :param Data:
            :param Grouping:
            :param Name:
            :param Quantity:
            :return:
            """

            if Grouping is not None:
                Selected_Data = [d for d in Data for item in Grouping if d.Grouping == item]
            elif Name is not None:
                Selected_Data = [d for d in Data for item in Name if d.Name == item]
            elif Quantity is not None:
                Selected_Data = [d for d in Data if d.Quantity == Quantity]

            return Selected_Data

        def find(self,Data, Grouping=None, Name=None):
            """
            Description

            :param Data:
            :param Grouping:
            :param Name:
            :return:
            """
            if Grouping is not None:
                Name_List = [d.Grouping for d in Data]
                Selected_Data = [index for index, value in enumerate(Name_List) for item in Grouping if value == item]

            elif Name is not None:
                Name_List = [d.Name for d in Data]
                Selected_Data = Name_List.index(Name)

            return Selected_Data

    Measurement_Data = list()

    for i in range(0, len(data)):
        # -------------- Class Definitions Start --------------------------------

        oMeasurements = Measurements()
        oMeasurement_Info = Measurement_Info()
        oDataSets = DataSets()
        oYChannels = Y_Channels()
        oXChannels = X_Channels()
        oChannelInfo = Channel()
        oRefChannelInfo = Channel()
        oNode_Channel = Channel()
        oNode_Ref_Channel = Channel()

        # -------------- Channel Node Info Start ------------------------------------

        #        oNode_Channel.Node_Number = int(re.findall(r'\d+',data[i]['rsp_ent_name'])[0])   #Should uncomment this for subsA.unv and subsB.unv files. (Data for tutorial Main_Data_Structure file)
        oNode_Channel.Node_Number = data[i]['rsp_node']
        oNode_Channel.Name = 'S' + str(oNode_Channel.Node_Number) + Directions[data[i]['rsp_dir']]

        # -------------- Ref Channel Node Info Start ------------------------------------

        #        oNode_Ref_Channel.Node_Number = int(re.findall(r'\d+',data[i]['ref_ent_name'])[0])  #Should uncomment this for subsA.unv and subsB.unv files. (Data for tutorial Main_Data_Structure file)
        oNode_Ref_Channel.Node_Number = data[i]['ref_node']
        oNode_Ref_Channel.Name = 'Impact' + str(oNode_Ref_Channel.Node_Number)

        # -------------- Channel Info Start ------------------------------------

        oChannelInfo.Direction_Number = data[i]['rsp_dir']
        oChannelInfo.Direction_Label = Directions[data[i]['rsp_dir']]
        oChannelInfo.Name = oNode_Channel.Name
        oChannelInfo.Node_Number = oNode_Channel.Node_Number
        oChannelInfo.Quantity.Name = data[i]['ordinate_axis_lab']
        oChannelInfo.Quantity.Unit.Name = data[i]['ordinate_axis_units_lab']
        oChannelInfo.Node = oNode_Channel

        # -------------- Ref Channel Info Start --------------------------------

        oRefChannelInfo.Direction_Number = data[i]['ref_dir']
        oRefChannelInfo.Direction_Label = Directions[data[i]['ref_dir']]
        oRefChannelInfo.Name = oNode_Ref_Channel.Name
        oRefChannelInfo.Node_Number = oNode_Ref_Channel.Node_Number
        oRefChannelInfo.Quantity.Name = data[i]['orddenom_axis_lab']
        oRefChannelInfo.Quantity.Unit.Name = data[i]['orddenom_axis_units_lab']
        oRefChannelInfo.Node = oNode_Ref_Channel

        # -------------- Y Channel Start --------------------------------------

        oYChannels.Name = oRefChannelInfo.Name + " " + oChannelInfo.Name
        oYChannels.Data = data[i]['data']
        oYChannels.Quantity.Name = data[i]['ordinate_axis_lab']
        oYChannels.Quantity.Unit.Name = data[i]['ordinate_axis_units_lab']

        oYChannels.Channel_Info = oChannelInfo
        oYChannels.Ref_Channel_Info = oRefChannelInfo

        # -------------- X Channel Start --------------------------------------

        oXChannels.Data = data[i]['x']
        oXChannels.Quantity.Name = data[i]['abscissa_axis_lab']
        oXChannels.Quantity.Unit.Name = data[i]['abscissa_axis_units_lab']

        # Assigning objects to parents classes to created nested class structure

        oDataSets.Y_Channels = oYChannels
        oDataSets.X_Channels = oXChannels
        oDataSets.Measurement_Time = data[i]['id3']

        oMeasurement_Info.Date = data[i]['id3']

        oMeasurements.DataSets = oDataSets
        oMeasurements.Name = data[i]['id1']
        oMeasurements.Measurement_Info = oMeasurement_Info

        Measurement_Data.append(oMeasurements)

    return Measurement_Data


from numpy import ndarray
import matplotlib.pyplot as plt
import math as mt
import cmath as cmt
import numpy as np

def plot_FRF(Data, dofs_i, dofs_j, Freq=None, PlotType=None):
    if Freq is None:
        Freq = np.arange(0, Data.nFreq)

    if PlotType is None:
        if isinstance(Data, ndarray):
            db = 20 * np.log10(abs(Data[dofs_i, dofs_j, Freq[0]:len(Freq)]))
        else:
            db = 20 * np.log10(abs(Data.Data[dofs_i, dofs_j, Freq[0]:len(Freq)])).transpose()

        plt.plot(db)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Admittance (dB)")
        plt.grid()

    elif PlotType == "log":
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.semilogy(abs(Data.Data[dofs_i, dofs_j, Freq[0]:len(Freq)]))
        plt.xlabel("Frequency (Hz)")
        if Data.Channels[0].Unit is not None:
            if type(Data.Channels[0].Unit) == str:
                plt.ylabel("Admittance in " + Data.Channels[0].Unit)
            else:
                plt.ylabel("Admittance in " + Data.Channels[0].Unit.Name)
        else:
            plt.ylabel("Admittance")

        plt.grid(which="both")

    elif PlotType == "phase":
        plt.subplot(2, 1, 2)
        Data = Data.Data[dofs_i, dofs_j, Freq[0]:len(Freq)]
        Data = [mt.degrees(cmt.phase(d)) for d in Data]
        plt.plot(Data)
        plt.ylim(-180, +180)
        plt.yticks([-180, -90, 0, 90, 180])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase Angle in Degrees")
        plt.grid()

if __name__ == '__main__':
    print("Test: Cat!")