import pandas as pd
from numpy import around, zeros
from scipy.spatial.transform import Rotation
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as R


#TODO: Error handling for reading the .xlsx files!!!
#TODO: Double check the Channels class instance!!!
#TODO: Documentation on each class and class instance!!!




class Impacts(object):
    """
    Description

    Attributes:

    """
    def __init__(self,file_name,sheet_name):
        self.Description = None
        self.Data = None

        self.fromXls(file_name, sheet_name)

    def fromXls(self,file_name, sheet_name):
        """
        Description

        :param file_name:
        :param sheet_name:
        :return:
        """
        xls = pd.ExcelFile(file_name)
        Data = pd.read_excel(xls, sheet_name)


        Impact_List = list()

        for i in range(0, len(Data)):
            oImpact = self.Impact()

            oQuantity = Quantity()
            #oUnit = oQuantity.Unit()

            oQuantity.Unit.Name = Data['Unit'][i]

            oQuantity.Name = Data['Quantity'][i]
            #oQuantity.Unit = oUnit

            oImpact.Node_Number = Data['NodeNumber'][i]
            oImpact.Grouping = Data['Grouping'][i]

            oImpact.Position = [Data['Position_1'][i], Data['Position_2'][i], Data['Position_3'][i]]
            oImpact.Direction = [Data['Direction_1'][i], Data['Direction_2'][i], Data['Direction_3'][i]]
            oImpact.Name = Data['Name'][i]
            oImpact.Description = Data['Description'][i]

            try:
                oImpact.Direction_Label = Data['DirectionLabel'][i]
            except:
                pass

            oImpact.Quantity = oQuantity

            Impact_List.append(oImpact)

        self.Data = Impact_List

    class Impact(object):
        """
        Description

        Attributes:

        """

        def __init__(self):
            self.Name = None
            self.Description = None

            self.Quantity = None
            self.Position = None
            self.Direction = None
            self.Grouping = None

            #self.Node_Number = None
            #self.Alignment = None


class Sensors(object):
    """
    Description

    Attributes:
        Description ()
        Data        ()
    """
    def __init__(self,file_name, sheet_name_sensor, sheet_name_channel):
        self.Description = None
        self.Data = None

        self.fromXls(file_name, sheet_name_sensor, sheet_name_channel)


    def fromXls(self,file_name, sheet_name_sensor, sheet_name_channel):
        """

        :param file_name:
        :param sheet_name_sensor:
        :param sheet_name_channel:
        :return:
        """

        xls = pd.ExcelFile(file_name)
        Data_Sensor = pd.read_excel(xls, sheet_name_sensor)

        Data_Channel = pd.read_excel(xls, sheet_name_channel)
        if Data_Channel.empty:
            """
            Convert data to channel!
            """
            for index, row in Data_Sensor.iterrows():
                r = R.from_euler('zyx', [[row["Orientation_1"], row["Orientation_2"], row["Orientation_3"]]], degrees=True)

                for blah in r.as_dcm().reshape(3,3):
                    # should be dynamic iteration;
                    _dict = {'Name': row["Name"],
                                         'Direction_1': blah[0],
                                         'Direction_2': blah[1],
                                         'Direction_3': blah[2],
                                         'Position_1': row['Position_1'],
                                         'Position_2': row['Position_2'],
                                         'Position_3': row['Position_3'],
                                         'Quantity': row["Quantity"],
                                         'Unit': row["Unit"],
                                         'Grouping': row["Grouping"],
                                         'NodeNumber': row["NodeNumber"],
                                         'Type': row["Type"]}

                    Data_Channel = Data_Channel.append(_dict, ignore_index = True)
                    if row["Type"] == "8840RotSensor":
                        break

        self.DataChannelPD = Data_Channel
        Angle_List, Channel_List = self.assert_correctness(Data_Sensor, Data_Channel)



        Sensor_List = list()

        for i in range(0, len(Data_Sensor)):

            oSensor = self.Sensor()

            oQuantity = Quantity()
            #oUnit = oQuantity.Unit()
            oOrientation = self.Orientation()

            oQuantity.Unit.Name = Data_Sensor['Unit'][i]

            if "Orientation_4" in Data_Sensor:

                Local_X = np.asarray(
                    [Data_Sensor["Orientation_1"][i], Data_Sensor["Orientation_2"][i], Data_Sensor["Orientation_3"][i]])
                Local_Y = np.asarray(
                    [Data_Sensor["Orientation_4"][i], Data_Sensor["Orientation_5"][i], Data_Sensor["Orientation_6"][i]])
                Local_Z = np.asarray(
                    [Data_Sensor["Orientation_7"][i], Data_Sensor["Orientation_8"][i], Data_Sensor["Orientation_9"][i]])
                oOrientation.Matrix = np.asarray([Local_X, Local_Y, Local_Z])

            else:

                if Angle_List:
                    oOrientation.Angles = Angle_List[i]

                oOrientation.Matrix = Channel_List[i]

            oQuantity.Name = Data_Sensor['Quantity'][i]
            #oQuantity.Unit = oUnit

            oSensor.Type = Data_Sensor['Type'][i]
            oSensor.Size = Data_Sensor['Size'][i]
            oSensor.Node_Number = Data_Sensor['NodeNumber'][i]
            oSensor.Grouping = Data_Sensor['Grouping'][i]

            oSensor.Position = [Data_Sensor['Position_1'][i], Data_Sensor['Position_2'][i],
                                Data_Sensor['Position_3'][i]]
            oSensor.Name = Data_Sensor['Name'][i]
            oSensor.Description = Data_Sensor['Description'][i]

            oSensor.Orientation = oOrientation
            oSensor.Quantity = oQuantity

            Sensor_List.append(oSensor)

        self.Data = Sensor_List


    def assert_correctness(self,sensor_data, channel_data):
        """
        Description

        :param channel_data:
        :return:
        """
        Matrix_List = list()
        Channel_Direction_List = list()
        Channel_List = list()
        Angle_List = list()

        nodes = Counter(channel_data["NodeNumber"])
        total_row_idx = channel_data.index.values
        Matrix = zeros((3, 3), float)

        for item in total_row_idx:
            Channel_Direction_List.append([channel_data["Direction_1"][item], channel_data["Direction_2"][item],
                                           channel_data["Direction_3"][item]])

        for keys in nodes:
            for i in range(0, nodes[keys]):
                Matrix[:, i] = Channel_Direction_List[0]
                Channel_Direction_List.pop(0)
            Channel_List.append(Matrix)
            Matrix = zeros((3, 3), float)

            Angle_List = list()

        if not (sensor_data["Orientation_1"].isnull().values.all() or sensor_data["Orientation_2"].isnull().values.all() \
                or sensor_data["Orientation_3"].isnull().values.all()):

            for i in range(0, len(sensor_data)):
                oRot = Rotation.from_euler("xyz", [sensor_data["Orientation_1"][i], sensor_data["Orientation_2"][i],
                                                   sensor_data["Orientation_3"][i]], degrees=True)
                Rot_Matrix = around(oRot.as_dcm())
                Matrix_List.append(Rot_Matrix)

            for i in range(0, len(sensor_data)):

                if (Matrix_List[i] != Channel_List[i]).all():

                    print("Angles of Sensor: ", i + 1, " are not correct. Please check!")

                    oRot = Rotation.from_dcm(Channel_List[i])
                    Angles = oRot.as_euler('xyz', degrees=True)

                    print("The Euler angles (xyz) format should be: ", Angles)

                else:

                    Angles = [sensor_data["Orientation_1"][i], sensor_data["Orientation_2"][i],
                              sensor_data["Orientation_3"][i]]

                Angle_List.append(Angles)


        return Angle_List,Channel_List

    class Sensor(object):
        """
        Sensor Object

        Attributes
            Type
            Size
            Mass
            Node_Number
            Grouping
            Quantity
            Position
            Orientation
            Name
            Description
            Alignment
        """

        def __init__(self):
            self.Name = None
            self.Description = None

            self.Quantity = None
            self.Position = None
            self.Orientation = None
            self.Grouping = None

            #self.Type = None
            #self.Size = None
            #self.Mass = None
            #self.Alignment = None
            #self.Node_Number = None


    class Orientation(object):
        """
        Description

        Attributes:
            Angles      ():
            Quaternion  ():
            Matrix      ():
        """

        def __init__(self):
            self.Angles = None
            self.Quaternion = None
            self.Matrix = None

class Channels(object):
    """
    Description

    Attributes:

    """
    def __init__(self,list_channels,imp_sen,_val):
        self.Description = None
        self.Data = None

        self.from_list(list_channels,imp_sen,_val)

    def from_list(self,list_channels,imp_sen,_val):
        oChannel_list = list()
        Channel_Direction = list()

        for i in range(0, len(list_channels)):
            oChannel = self.Channel()

            oChannel.Name = list_channels[i].Name
            oChannel.Grouping = list_channels[i].Grouping
            #oChannel.Node = list_channels[i].Node
            #oChannel.Node_Number = list_channels[i].Node_Number
            oChannel.Quantity = list_channels[i].Quantity
            #oChannel.Quantity.Unit.Name = list_channels[i].Unit

            #try:
            #    oChannel.Direction_Label = list_channels[i].Direction_Label

            #oChannel.Direction_Number = list_channels[i].Direction_Number
            oChannel_list.append(oChannel)

        if _val == "sensors":
            Channel_Position = [s.Position for s in imp_sen.Data for c in list_channels if s.Node_Number == c.Node_Number]

            for s in imp_sen.Data:
                counter = 0
                for c in list_channels:
                    if s.Node_Number == c.Node_Number:
                        Channel_Direction.append(s.Orientation.Matrix[:, counter])
                        counter = counter + 1
        else:
            Channel_Direction = [i.Direction for i in imp_sen.Data for r in list_channels if i.Node_Number == r.Node_Number]
            Channel_Position = [i.Position for i in imp_sen.Data for r in list_channels if i.Node_Number == r.Node_Number]



        for i in range(0, len(list_channels)):
            try:
                #oChannel_list[i].Node.Position = Channel_Position[i]
                oChannel_list[i].Position = Channel_Position[i]
                oChannel_list[i].Direction = Channel_Direction[i]
            except:
                continue


        self.Data = oChannel_list


    class Channel(object):
        """
        Description

        Attributes:
        """
        def __init__(self):
            self.Name = None
            self.Description = None

            self.Quantity = Quantity()
            self.Position = None
            self.Direction = None
            self.Grouping = None

            #self.Dim_Vector = None
            #self.dBref = None
            #self.Notes = None
            #self.Type = None
            #self.Label = None
            #self.DOF_Label = None
            #self.Direction_Label = None
            #self.Direction_Number = None
            #self.Node = None
            #self.Component = None
            #self.Node_Number = None


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



class Quantity(object):
    """
    Description

    Attributes:
    """
    def __init__(self):
        self.Name = None
        self.Unit = self.Unit()

    class Unit(object):
        """
        Description

        Attributes:
        """

        def __init__(self):
            self.Name = None
            #self.Symbol = None
            #self.DimVector = None
            #self.SIFactor = None
            #self.dBref = None


if __name__ == '__main__':
    print("Test: Dog!")