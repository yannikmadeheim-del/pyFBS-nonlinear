#############
Identification algorithm for inconsistent measurements
#############


This example introduces a comprehensive experimental method to check the consistency of individual measurements based on comparisons with the complete experimental response model [1]_.  
The numerical model is introduced only to enable the experimental model to be expanded using the System Equivalent Model Mixing method. 
The entire formulation is developed in the frequency domain.

.. note:: 
   Download example showing the basic use of method: :download:`18_SEMM_consistency_measurements.ipynb <../../../examples/18_SEMM_consistency_measurements.ipynb>`

Example data import
*******************

In the beginning, it necessary to define numerical and experimental data (response models). Datasets used in this example are from a laboratory testbench and are available directly within the :mod:`pyFBS`.

Experimental model
------------------
The experimental model must be properly aranged so that the FRFs are of correct shape. 
The first dimension represents the frequency depth, second the response points, and the third excitation points. 
The experimental model is used as the overlay model.

.. code-block:: python

   exp_file = r"./lab_testbench/Measurements/Y_AB.p"

   freq, Y_exp = np.load(exp_file, allow_pickle = True)
   Y_exp = np.transpose(Y_exp, (2, 0, 1))

   df_chn_exp = df_chn[0:15]
   df_imp_exp = df_imp[5:20]


Numerical model
---------------
FRFs used for the parent model numerical model can be imported to Python or generated with the mass and stiffness matrix using :mod:`pyFBS.MK_model.FRF_synth`.
Locations and directions for which FRFs are generated are defined in an Excel file and can be parsed into a :mod:`pandas.DataFrame`.

.. code-block:: python

   stl = r"./lab_testbench/STL/AB.stl"
   xlsx = r"./lab_testbench/Measurements/AM_measurements.xlsx"

   full_file = r"./lab_testbench/FEM/AB.full"
   rst_file = r"./lab_testbench/FEM/AB.rst"

   MK = pyfbs.mck.Model.from_ansys(rst_file,full_file,no_modes = 100,recalculate = False, mesh_scale=1000)

   df_chn = pd.read_excel(xlsx, sheet_name='Channels_AB')
   df_imp = pd.read_excel(xlsx, sheet_name='Impacts_AB')

   MK.frf_synth(df_chn,df_imp, 
               f_start=0,
               f_end=2002.5,
               f_resolution=2.5,
               modal_damping = 0.003,
               frf_type = "accelerance")

   Y_num = MK.frf

   df_chn_num = df_chn
   df_imp_num = df_imp

As experimental, also the numerical model must be properly arranged. The first dimension represents the frequency depth, the second the response points, and the third excitation points. 
Additionally, the frequency resolution of the numerical and experimental model has to be the same.

The numerical model is not necessarily a square matrix, as it can also be rectangular. 
But the numerical model must contain at least all the DoFs contained in the experimental model.

Application of identification algorithm
***************************************

The basic idea consists of removing a measurement from the entire data set and then reconstructing it from the remaining measurement set. 
After the expansion process, the comparison between all the reconstructed FRFs and their original experimental counterparts is made. 
If the identification criterion shows a reasonable correlation between the experimental and the associated reconstructed FRF, 
it can be considered as a consistent measurement and should remain in the experimental set. 
The comparison between the reconstructed and the original measurements can be performed using a variety of criteria, 
for instance, the coherence criterion compares two different FRFs for the same input-output position. 

.. code-block:: python

   FRF_rec, coh = pyfbs.expansion.identification_algorithm(Y_num, Y_exp, 
                        df_chn_num, df_imp_num, df_chn_exp, df_imp_exp)

Finally, the results of the method are inspected. Using coherence criteria, quick assessement of the measurement consistency can be obtained:

.. code-block:: python

   pyfbs.display.imshow(np.average(coh, axis=0))

.. raw:: html

   <iframe src="../../_static/id_algorithm.html" height="280px" width="750px" frameborder="0"></iframe>

Let us now compare two examples of identified consistent and inconsistent measurements, respectivelly:

.. raw:: html

   <iframe src="../../_static/id_algorithm_good.html" height="500px" width="750px" frameborder="0"></iframe>

.. raw:: html

   <iframe src="../../_static/id_algorithm_bad.html" height="500px" width="750px" frameborder="0"></iframe>

.. card:: That's a wrap!

    Want to know more, see a potential application? Contact us at info.pyfbs@gmail.com!
   
.. rubric:: References

.. [1] Kodrič M, Čepon G, Boltežar M. Experimental framework for identifying inconsistent measurements in frequency-based substructuring. Mechanical Systems and Signal Processing. 2021 Jun 1;154:107562.