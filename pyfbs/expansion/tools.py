import numpy as np

DEFAULT_COLUMNS = [
    "Position_1",
    "Position_2",
    "Position_3",
    "Direction_1",
    "Direction_2",
    "Direction_3",
]


def find_locations_in_data_frames(df_1, df_2, additional_columns=[]):
    """Find matching locations of data frames ``df_1`` and ``df_2``.

    :param df_1: Data frame 1
    :type df_1: pandas.DataFrame
    :param df_2: Data frame 2
    :type df_2: pandas.DataFrame
    :return: Vector of matching locations of both data frames.
    :rtype: array(float)
    """

    columns = DEFAULT_COLUMNS + additional_columns
    df_1_val = np.array(df_1[columns].values, dtype=float)
    df_2_val = np.array(df_2[columns].values, dtype=float)

    # to prevent numerical errors
    df_1_val = np.round(df_1_val, 6)
    df_2_val = np.round(df_2_val, 6)

    return np.array(
        np.all(
            (df_1_val[:, None, :] == df_2_val[None, :, :]), axis=-1
        ).nonzero()
    ).T
