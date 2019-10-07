from pyfbs.io import Sensor
import os


# get path to data
path_list = os.path.realpath('__file__').split(os.sep)
script_directory = path_list[0:len(path_list)-2]
example_xlsx = "/".join(script_directory) + "/" + "data/test_geometry.xlsx"


ASensor = Sensor().fromXls(example_xlsx,"Sensors", "Channels")

#ASensor.fromXls(example_xlsx,"Sensors", "Channels")