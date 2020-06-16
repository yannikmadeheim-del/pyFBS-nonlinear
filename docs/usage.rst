=====
Usage
=====

To use :mod:`pyFBS` within a project simply import the package:

.. code-block:: python

	import pyFBS
	
	
********
Features
********

3D display
==========
Short description 

:mod:`pyFBS.view3D`

FRF synthetization
==================
Short description 

:mod:`pyFBS.MK_model`

Virtual Point Transformation
============================
Short description 

:mod:`pyFBS.VPT`

System Equivalent Model Mixing
==============================
Short description 

:mod:`pyFBS.SEMM`



	
************
Example data
************
To test out the capabilities of :mod:`pyFBS` also two example datasets are available directly with the package. 
For each testbench structure a dictionary is available containing relative path to the predefined datasets.

Predefined datasets are also used directly in basic and application examples.

Academic testbench
==================
The first testbench is an academic example ``pyFBS.example_lab_testbench``. 
The testbench is used to evaluate and compare different dynamic substructuring methodologies. 

.. figure:: ./examples/data/3D_view.png
   :width: 800px
   
   An example of a academic substructuring testbench depicted in the pyFBS 3D display.


Example datasets for academic testbench contain:

* STL files of the testbench (e.g. ``pyFBS.example_lab_testbench['STL']['A']``),

* FEM of each substructure (e.g. ``pyFBS.example_lab_testbench['FEM']``),

* Excel files of positional data for sensors and impacts (e.g. ``pyFBS.example_lab_testbench['meas']['xlsx']``),

* Experimental FRF measurements (e.g. ``pyFBS.example_lab_testbench['meas']['Y_A']``).

Automotive testbench
====================

The second testbench is an automotive example ``pyFBS.example_auto_testbench``. 
This testbench s designed to represent an engine-transmission unit’s suspension from a real car.


Add a picture of example.

Example datasets for automotive testbench contain:

* STL files of the testbench (e.g. ``pyFBS.example_auto_testbench['STL']['receiver']``),

* Excel files of positional data for sensors and impacts (e.g. ``pyFBS.example_auto_testbench['meas']['xlsx_modal']``),

* Experimental FRF measurements (e.g. ``pyFBS.example_auto_testbench['meas']['Y_m_1']``).

