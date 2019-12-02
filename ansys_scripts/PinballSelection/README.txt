A collection of python scripts for Ansys ACTConsole to create a FRF matrix with multiple outputs/inputs. Coordinate CSYS
should be placed at input locations with capital letter H in the name and at the output location the CSYS with a capital
S in the name. The workbench project schematic should contain: Geometry -> Modal -> Harmonic Response.

The location of output/input remains at the CSYS and the mesh nodes are selected within the selected PinBall region.


The scripts should then be executed in the following order:
    1. CreateObjects        (create named selections and remote points according to output/input locations)
    2. CheckSolve           (iterates through all inputs; the locations can be double checked)
    3. GenerateLocations    (creates a .csv file containing the information on output/input locations)
    4. SolveFRF             (solves FRFs for each input and save the result in the project directory in ./FRF folder)

    5. DeleteObjects        (deletes all created objects from previous scripts)



Ansys version: Release 17.2
Author: Tomaž Bregar
Date: 02.09.2019