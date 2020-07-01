=====
Usage
=====

To use :mod:`pyFBS` within a project simply import the package:

.. code-block:: python

	import pyFBS

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
The automotive testbench was designed to represent an engine-transmission unit’s suspension from a real car.


.. figure:: ./examples/data/nine_one.png
   :width: 800px
   
   An example of a automotive testbench depicted in the pyFBS 3D display.
   
Example datasets for automotive testbench contain:

* STL files of the testbench (e.g. ``pyFBS.example_auto_testbench['STL']['receiver']``),

* Excel files of positional data for sensors and impacts (e.g. ``pyFBS.example_auto_testbench['meas']['xlsx_modal']``),

* Experimental FRF measurements (e.g. ``pyFBS.example_auto_testbench['meas']['Y_m_1']``).

********
Features
********

3D display
==========
With the pyFBS substructures and positions of impacts, sensors and channels can be visualized in 3D display :mod:`pyFBS.view3D`. 
The 3D display uses PyVista :cite:`usage-sullivan2019pyvista` for the visualization and enables an intuitive way to display relevant data. 
Sensors and impacts can be interactively positioned on the substructures and the updated positions can be directly used within pyFBS. 
Furthermore, various animations can be performed directly in the 3D display, such as the animation of mode shapes or operational deflection shapes.

One of the main features of the pyFBS is also the ability to synthetize FRFs directly from the predefined positions of channels and impacts. 
Currently, mode superposition FRF synthetization is supported, where mass and stiffness matrices are imported from FEM software. 
Damping can be introduced as modal damping for each mode shape. Additionally, noise can be introduced to the response so a realistic set of FRFs, representing experimental measurements, can be obtained.


FRF synthetization
==================
One of the main features of the pyFBS is also the ability to synthetize FRFs directly from the predefined positions of channels and impacts :mod:`pyFBS.MK_model`. 
Currently, mode superposition FRF synthetization is supported, where mass and stiffness matrices are imported from FEM software. 
Damping can be introduced as modal damping for each mode shape. 
Additionally, noise can be introduced to the response so a realistic set of FRFs, representing experimental measurements, can be obtained.


Virtual Point Transformation
============================
Within the pyFBS Virtual Point Transformation (VPT) :mod:`pyFBS.VPT` is implemented :cite:`usage-solvingRDOF`. 
VPT projects measured dynamics on the predefined interface displacement modes (IDMs). 
The interface is usually considered to be rigid; therefore, only 6 rigid IDMs are used in the transformation. 
After applying the transformation, a collocated set of FRFs is obtained, which can afterwards directly be used in DS. 
Expanded VPT is also supported, where directly measured rotational response is included in the transformation :cite:`usage-Bregar2020`.


System Equivalent Model Mixing
==============================
The pyFBS supports System Equivalent Model Mixing (SEMM) :mod:`pyFBS.SEMM` :cite:`usage-Klaassen2018`. 
SEMM enables mixing of two equivalent frequency-based models into a hybrid model. 
The models used can either be of numerical or experimental nature. 
One of the models provides the dynamic properties (overlay model) and the second model provides a set of degrees of freedom. 
A numerical model is commonly used as a parent model and an experimental model is used as an overlay model. 


Singular Vector Transformation
==============================
With the pyFBS the Singular Vector Transformation is supported  (SVT) :mod:`pyFBS.SVT`. 
SVT projects measured dynamics into subspaces composed by dominant singular vectors. 
The singular vectors are extracted directly from the measured interface dynamics by using Singular Value Decomposition (SVD). 
Since the reduction space is defined directly from the measured dynamics, no analytical or geometrical model is required.



.. rubric:: References

.. bibliography:: ..\joss\paper.bib
   :style: unsrt
   :filter: docname in docnames
   :keyprefix: usage-
