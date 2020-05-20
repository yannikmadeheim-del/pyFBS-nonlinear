import pyuff
from pyFBS.io import Quantity

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


def response_sync_lstsq(response_vec):
    _mode = response_vec.flatten()
    z = np.arctan(np.average(np.imag(_mode) / np.real(_mode), weights=np.abs(_mode) ** 2))

    response_vec_norm = response_vec * (np.cos(-1 * z) + 1j * np.sin(-1 * z))
    return response_vec_norm



def modeshape_sync_lstsq(mode_shape_vec):
    """
    Creates a straight line fit in the complex plane and substracts the phase from mode shapes.
    """
    _n = np.zeros_like(mode_shape_vec)
    for i in range(np.shape(mode_shape_vec)[1]):
        _mode = mode_shape_vec[:,i]
        z = np.arctan(np.average(np.imag(_mode)/np.real(_mode),weights = np.abs(_mode)**2))
            
        _n[:,i] = _mode*(np.cos(-1*z)+1j*np.sin(-1*z))
    return _n

def modeshape_scaling_DP(mode_shape_vec, driving_point,sync = True):
    """
    Scales modeshape from driving point measurement.
    """
    
    _mode = mode_shape_vec
    for i in range(np.shape(mode_shape_vec)[1]):
        _mode[:,i] = _mode[:,i]/np.sqrt(mode_shape_vec[driving_point,i])
    
    if sync:
        _mode = modeshape_sync_lstsq(_mode)
    return _mode        

def MCF(mod):
    """
    Calculate Mode Complexity Factor (MCF)
    """
    sxx = np.real(mod).T@np.real(mod)
    syy = np.imag(mod).T@np.imag(mod)
    sxy = np.real(mod).T@np.imag(mod)
    mcf = (1 - ((sxx-syy)**2+4*sxy**2)/((sxx+syy)**2))
    return mcf


import math
def eulerAnglesToRotationMatrix(theta):
    """
    Creates a rotational matrix based on the euler angles
    :param theta:
    :return:
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), math.sin(theta[0])],
                    [0, -math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, -math.sin(theta[1])],
                    [0, 1, 0],
                    [math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), math.sin(theta[2]), 0],
                    [-math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_x, np.dot(R_y, R_z))

    return R


def flatten_FRFs(Y):
    """
    :param Y:
    :return:
    Flattens FRF matrix Y from shape (out,in,freq) in (out x in,freq)
    """
    new = np.zeros((Y.shape[0] * Y.shape[1], Y.shape[2]), dtype=complex)

    _len = Y.shape[1]
    for i in range(Y.shape[0]):
        new[_len * i:_len * (i + 1), :] = Y[i, :, :]

    return new

def unflattenFRFs(_modes_acc,Y):
    """
    :param _modes_acc:
    :param Y:
    :return:
    Reconstructs modeshapes from pyEMA.get_constants in 3D space based on the structure of Y
    """
    new_mode = np.zeros((Y.shape[0],Y.shape[1],_modes_acc.shape[1]),dtype = complex)

    _len = Y.shape[1]
    for i in range(Y.shape[0]):
        new_mode[i,:,:] = _modes_acc[i*_len:(i+1)*_len,:]
    return new_mode

def complex_plot_3D(mode_shape):
    """
    Plots modeshape on a radial plot.
    :param mode_shape:
    :return:
    """
    fig = plt.figure(figsize = (3,3))
    ax1 = plt.subplot(111,projection = "polar")

    for i,color in enumerate(["tab:red","tab:green","tab:blue"]):
        for x in mode_shape[:,i]:
            ax1.plot([0,np.angle(x)],[0,np.abs(x)],marker='.',color = color,alpha = 0.5)

    plt.yticks([])


def mode_animation(mode_shape,scale, no_points=60):
    """
    Create animation sequence
    """
    ann = np.zeros((mode_shape.shape[0], mode_shape.shape[1], no_points))

    for g, _t in enumerate(np.linspace(0, 2, no_points)):
        ann[:, :, g] = (np.real(mode_shape) * np.cos(2 * np.pi * _t) - np.imag(mode_shape) * np.sin(
            2 * np.pi * _t))
    ann = ann / np.max(ann) *scale
    return ann

if __name__ == '__main__':
    print("Test: Cat!")