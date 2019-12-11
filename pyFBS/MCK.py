import re
from pyFBS.IO import Channels

from numpy import ndarray
import matplotlib.pyplot as plt
import math as mt
import cmath as cmt
import numpy as np

#TODO: Documentation on each class and class instance!!!

class FRF(object):
    """
    FRF class for ...

    """
    def __init__(self):
        self.Freqs = None
        self.nFreq = None
        self.Channels = None
        self.nChannels = None
        self.RefChannels = None
        self.nRefChannels = None
        self.Data = None
        self.Name = None
        self.Coherence = None


    def from_series_to_matrix(self,frfs, sensors, impacts, _coh = None):
        """
        Returns the uncoupled FRF matrix object.

        :param sub_structure:
        :param sensors:
        :param impacts:
        :return:
        """

        sub_structure = self.assign_grouping_number(frfs, sensors, impacts)



        self.Freqs = sub_structure[0].DataSets.X_Channels.Data
        self.nFreq = len(self.Freqs)

        if _coh != None:
            _temp = self.assign_grouping_number(_coh, sensors, impacts)
            self.Coherence = self.create_matrix_data(_temp, self.nFreq)

        self.Data = self.create_matrix_data(sub_structure, self.nFreq)

        self.Measurement_Info = sub_structure[0].Measurement_Info.Date

        Sensor_Names = sorted(set([d.DataSets.Y_Channels.Channel_Info.Name for d in sub_structure]))

        Impact_Names = sorted(set([d.DataSets.Y_Channels.Ref_Channel_Info.Name for d in sub_structure]))

        _Channels = [d.DataSets.Y_Channels.Channel_Info for d in sub_structure if d.DataSets.Y_Channels.Channel_Info.Name \
                    in Sensor_Names and d.DataSets.Y_Channels.Ref_Channel_Info.Name == Impact_Names[0]]

        _Ref_Channels = [d.DataSets.Y_Channels.Ref_Channel_Info for d in sub_structure if
                        d.DataSets.Y_Channels.Ref_Channel_Info.Name \
                        in Impact_Names and d.DataSets.Y_Channels.Channel_Info.Name == Sensor_Names[0]]


        self.RefChannels = Channels(_Ref_Channels,impacts, _val = "impacts").Data
        self.nRefChannels = len(_Ref_Channels)


        self.Channels = Channels(_Channels,sensors,_val = "sensors").Data
        self.nChannels = len(_Channels)

    def assign_grouping_number(self,sub_structure, sensors, impacts):
        """
        Assigns the grouping number
        :param sub_structure:
        :param sensors:
        :param impacts:
        :return:
        """
        for i in range(0, len(sub_structure)):

            for j in range(0, len(sensors.Data)):

                if (sub_structure[i].DataSets.Y_Channels.Channel_Info.Node_Number == sensors.Data[j].Node_Number):
                    sub_structure[i].DataSets.Y_Channels.Channel_Info.Grouping = sensors.Data[j].Grouping
                    sub_structure[i].DataSets.Y_Channels.Channel_Info.Node.Grouping = sensors.Data[j].Grouping

        for i in range(0, len(sub_structure)):

            for j in range(0, len(impacts.Data)):

                if (sub_structure[i].DataSets.Y_Channels.Ref_Channel_Info.Node_Number == impacts.Data[j].Node_Number):
                    sub_structure[i].DataSets.Y_Channels.Ref_Channel_Info.Grouping = impacts.Data[j].Grouping
                    sub_structure[i].DataSets.Y_Channels.Ref_Channel_Info.Node.Grouping = impacts.Data[j].Grouping

        return sub_structure



    def create_matrix_data(self,sub_structure, nFreq):
        """

        :param sub_structure:
        :param nFreq:
        :return:
        """
        iterator = 0

        Names = set([d.DataSets.Y_Channels.Channel_Info.Name for d in sub_structure])
        Nodes_Sensor = sorted([int(re.findall('\d+', item)[0]) for item in Names])
        Nodes_Impact = sorted(set([d.DataSets.Y_Channels.Ref_Channel_Info.Node_Number for d in sub_structure]))

        FRF_Matrix = np.zeros((len(Nodes_Sensor), len(Nodes_Impact), nFreq), complex)

        Data = [d.DataSets.Y_Channels.Data for d in sub_structure]

        if sub_structure[0].DataSets.Y_Channels.Ref_Channel_Info.Name != sub_structure[
            1].DataSets.Y_Channels.Ref_Channel_Info.Name:

            for i in range(0, len(Nodes_Sensor)):
                for j in range(0, len(Nodes_Impact)):
                    FRF_Matrix[i, j, :] = Data[j + iterator]
                iterator = iterator + len(Nodes_Impact)

        else:

            for j in range(0, len(Nodes_Impact)):
                for i in range(0, len(Nodes_Sensor)):
                    FRF_Matrix[i, j, :] = Data[i + iterator]
                iterator = iterator + len(Nodes_Sensor)

        return FRF_Matrix


if __name__ == '__main__':
    print("Test: Dog!")