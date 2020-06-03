import pyuff
import numpy as np
import math
import pandas as pd
from pyFBS.utility import *


# TODO: are we missing anything?

def load_uff_file_PAK(uff_file_data,uff_file_output,uff_file_input):
    """
    Description

    :param uff_file:
    :return:
    """

    uff_file_out = pyuff.UFF(uff_file_output)
    data_output = uff_file_out.read_sets()

    uff_file_in = pyuff.UFF(uff_file_input)
    data_input = uff_file_in.read_sets()

    uff_file = pyuff.UFF(uff_file_data)
    data = uff_file.read_sets()

    chn_dof = len(data_output["x"]) * 3  # triax acc
    imp_dof = len(data_input["x"])

    Directions = {0: "None", 1: "+X", 2: "+Y", 3: "+Z", -1: "-X", -2: "-Y", -3: "-Z"}
    Directions_array = {0: "None", 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1], -1: [-1, 0, 0], -2: [0, -1, 0],
                        -3: [0, 0, -1]}

    freq = data[0]["x"]
    FRF = np.zeros((len(freq), chn_dof, imp_dof), dtype=complex)

    out_resp = np.zeros((chn_dof, 3))
    in_resp = np.zeros((imp_dof, 3))

    out_N = [""] * chn_dof
    in_N = [""] * imp_dof

    i = 0
    for _out in range(chn_dof):
        for _in in range(imp_dof):
            Node_Number = data[i]['rsp_node']
            Name = 'S' + str(math.ceil(Node_Number / 3)) + " " + Directions[data[i]['rsp_dir']]

            out_resp[_out, :] = Directions_array[data[i]['rsp_dir']]
            in_resp[_in, :] = Directions_array[data[i]['ref_dir']]

            Node_Number = data[i]['ref_node']
            RefName = 'H' + str(Node_Number) + " " + Directions[data[i]['ref_dir']]

            FRF[:, _out, _in] = data[i]["data"]

            out_N[_out] = Name
            in_N[_in] = RefName

            i += 1

    # parse channel data
    columns_chann = ["Name", "Description", "Type", "DirectionLabel", "Quantity", "Unit", "Component", "NodeNumber",
                     "Grouping", "Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2",
                     "Direction_3"]

    df = pd.DataFrame(columns=columns_chann)

    for _out in range(int(chn_dof / 3)):
        for i in range(3):
            out_dir = out_resp[_out * 3 + i]
            out_pos = [data_output["x"][_out], data_output["y"][_out], data_output["z"][_out]]

            data_chn = np.asarray([[out_N[_out * 3 + i], None, None, out_N[_out * 3 + i].split(" ")[1], None, None,
                                    None, None, None, out_pos[0],
                                    out_pos[1], out_pos[2], out_dir[0], out_dir[1], out_dir[2]]])

            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df = df.append(df_row, ignore_index=True)

    df_chn = df

    # parse impact data
    columns_chann = ["Name", "Description", "Type", "DirectionLabel", "Quantity", "Unit", "Component", "NodeNumber",
                     "Grouping", "Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2",
                     "Direction_3"]

    df = pd.DataFrame(columns=columns_chann)

    for _in in range(imp_dof):
        in_pos = [data_input["x"][_in], data_input["y"][_in], data_input["z"][_in]]
        in_dir = in_resp[_in]

        data_chn = np.asarray(
            [[in_N[_in].split(" ")[0], None, None, in_N[_in].split(" ")[1], None, None, None, None, None, in_pos[0],
              in_pos[1], in_pos[2], in_dir[0], in_dir[1], in_dir[2]]])

        df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
        df = df.append(df_row, ignore_index=True)

    df_imp = df

    df_acc = generate_sensors_from_channels(df_chn)

    return freq,FRF,df_chn,df_imp,df_acc

def load_lvm_files(directory):
    """
    Description

    :param directory:
    :return:
    """
    #TODO: port the function
    return None


