############
VPT Coupling
############

The frequency-based substructure coupling is embedded in ``pyFBS``. 
In particular, the admittance-based dual formulation named Lagrange-Multiplier Frequency-Based Substructuring (LM-FBS) is implemented [1]_. 
In the following, a basic coupling of two numerically-generated substructures is presented. 
The virtual point transformation [2]_ is applied to impose collocated matching DoFs at the interface. 
This can also be performed analogously with experimentally acquired data.

.. note:: 
   Download example showing a substructure coupling application: :download:`07_coupling.ipynb <../../../examples/07_FBS_coupling.ipynb>`

.. tip::
    Why use virtual point when coupling substructures?
    
    * Complex interfaces are very often inaccessible for the measurement equipment. Therefore measurements are often performed away from the actual interface. Also, the collocation of DoFs at the neighboring interfaces is almost impossible to ensure. Virtual point, on the other hand, can be defined in an arbitrary location at the interface that coincides for all substructures.
    * Virtual point accounts for rotational degrees of freedom, which are mandatory for the successful coupling of the substructures. 
    * By the reduction of measurements to the virtual point the interface problem is weakened. That means that compatibility and equilibrium conditions on the measured DoFs are a bit more relaxed. In this manner, unwanted stiffening effects can be avoided. 
    * Due to the reduction, measurement errors such as bias from sensor positioning or uncorrelated measurement noise are filtered out to some extend.
         
    
Example Datasets and 3D view
****************************
Load the required predefined datasets and open the 3D viewer in the background as already shown in `3D Display <../../../html/examples/basic_examples/01_static_display.html>`_. 
Especially for the illustration of different substructures and the assembly, the 3D viewer subplot capabilities of `PyVista <https://docs.pyvista.org/index.html>`_ can be used.

.. code-block:: python

    view3D = pyFBS.view3D(show_origin = False, show_axes = False,shape =  (1,3),title = "Overview")
    
Add an STL file of substructure A to the 1-1 subplot and show the corresponding accelerometers, channels and impacts.

.. code-block:: python

    view3D.plot.subplot(0,0)
    view3D.plot.isometric_view()
    view3D.plot.add_text("A structure", position='upper_left', font_size=10, color="k", font="times", name="A_structure")

    view3D.add_stl(stl_dir_A,color = "#83afd2",name = "A");
    view3D.show_acc(df_acc_A)
    view3D.show_imp(df_imp_A)
    view3D.show_chn(df_chn_A)
    
.. figure:: ./../data/seven_one.png
   :width: 500px
    
    
Add an STL file of substructure B to the 1-2 subplot and show the corresponding accelerometers, channels and impacts.

.. code-block:: python

    view3D.plot.subplot(0,1)
    view3D.plot.isometric_view()
    view3D.plot.add_text("B structure", position='upper_left', font_size=10, color="k", font="times", name="B_structure")

    view3D.add_stl(stl_dir_B,color = "#83afd2",name = "B");
    view3D.show_acc(df_acc_B,overwrite = False)
    view3D.show_imp(df_imp_B,overwrite = False)
    view3D.show_chn(df_chn_B,overwrite = False)
 
.. figure:: ./../data/seven_two.png
   :width: 500px
   
    
Add an STL file of the assembly AB to the 1-2 subplot and show the corresponding reference accelerometers, channels and impacts.
 
.. code-block:: python

    view3D.plot.subplot(0,2)
    view3D.plot.isometric_view()
    view3D.plot.add_text("AB structure", position='upper_left', font_size=10, color="k", font="times", name="AB_structure");

    view3D.add_stl(stl_dir_AB,color = "#83afd2",name = "AB");
    view3D.show_acc(df_acc_AB,overwrite = False)
    view3D.show_imp(df_imp_AB,overwrite = False)
    view3D.show_chn(df_chn_AB,overwrite = False)
    
.. figure:: ./../data/seven_three.png
   :width: 500px
        
Each separate subplot view can also be linked or unlinked:

.. code-block:: python

    view3D.plot.link_views()
    #view3D.plot.unlink_views()

.. tip::
    With ``pyFBS`` you can simply prepare the experiments before hand! 
    Position your virtual accelerometers and sensors on 3D model, generate numerical FRFs and make sure that everything is in order. 
    Then, perform your experiment, following the sensor setup you prepared in your virtual example. 
    Simply replace numerical FRFs with experimental ones, and results are only few clicks away!

.. 
    Numerical model
    ***************
    Load the corresponding .full and .rst file from the example datasets. For more information on .full and .rst files refer to the :download:`03_FRF_synthetization.ipynb <../../../examples/03_FRF_synthetization.ipynb>` example.

    .. code-block:: python

        full_file_AB = r"./lab_testbench/FEM/AB.full"
        ress_file_AB = r"./lab_testbench/FEM/AB.rst"

        full_file_B = r"./lab_testbench/FEM/B.full"
        ress_file_B = r"./lab_testbench/FEM/B.rst"

        full_file_A = r"./lab_testbench/FEM/A.full"
        ress_file_A = r"./lab_testbench/FEM/A.rst"
        
    Create an MK model for each component:

    .. code-block:: python

        MK_A = pyFBS.MK_model(ress_file_A,full_file_A,no_modes = 100,allow_pickle= True,recalculate = False)
        MK_B = pyFBS.MK_model(ress_file_B,full_file_B,no_modes = 100,allow_pickle= True,recalculate = False)
        MK_AB = pyFBS.MK_model(ress_file_AB,full_file_AB,no_modes = 100,allow_pickle= True,recalculate = False)
        
    Update locations of channels and impacts to snap to the nearest FE node.

    .. code-block:: python

        df_chn_A_up = MK_A.update_locations_df(df_chn_A)
        df_imp_A_up = MK_A.update_locations_df(df_imp_A)

        df_chn_B_up = MK_B.update_locations_df(df_chn_B)
        df_imp_B_up = MK_B.update_locations_df(df_imp_B)

        df_chn_AB_up = MK_AB.update_locations_df(df_chn_AB)
        df_imp_AB_up = MK_AB.update_locations_df(df_imp_AB)
        
    Perform the FRF sythetization for each component based on the updated locations.

    .. code-block:: python

        MK_A.FRF_synth(df_chn_A_up,df_imp_A_up,f_start = 0,modal_damping = 0.003)
        MK_B.FRF_synth(df_chn_B_up,df_imp_B_up,f_start = 0,modal_damping = 0.003)
        MK_AB.FRF_synth(df_chn_AB_up,df_imp_AB_up,f_start = 0,modal_damping = 0.003)
    
    
Virtual point transformation
****************************

.. tip::
    It would be impractical to measure interface admittance for both substructures in multiple DoFs at the interface and furthermore ensure, 
    that these DoFs are perfectly collocated. Therefore we adopt VPT in order to obtain a collocated full-DoF interface admittance matrix for each substructure.

The VPT can be performed directly on the measured/generated FRFs. See the :download:`04_VPT.ipynb <../../../examples/04_VPT.ipynb>` example for more options and details.

.. code-block:: python

    df_vp = pd.read_excel(pos_xlsx, sheet_name='VP_Channels')
    df_vpref = pd.read_excel(pos_xlsx, sheet_name='VP_RefChannels')

    vpt_A = pyFBS.VPT(df_chn_A_up,df_imp_A_up,df_vp,df_vpref)
    vpt_B = pyFBS.VPT(df_chn_B_up,df_imp_B_up,df_vp,df_vpref)
    
Apply the defined VP transformation on the FRFs:

.. code-block:: python

    vpt_A.apply_VPT(MK_A.freq,MK_A.FRF)
    vpt_B.apply_VPT(MK_B.freq,MK_B.FRF)
    
Extract the requried FRFs and the frequency vector:

.. code-block:: python

    freq = MK_A.freq
    Y_A = vpt_A.vptData
    Y_B = vpt_B.vptData
    
LM-FBS Coupling
***************

First, construct an admittance matrix for the uncoupled system, containing substructure admittances:

.. math::
    \mathbf{Y}^\text{A|B} = \begin{bmatrix} 
    \mathbf{Y}^\text{A} & \mathbf{0} \\
    \mathbf{0} & \mathbf{Y}^\text{B}
    \end{bmatrix}

.. code-block:: python

    Y_AnB = np.zeros((Y_A.shape[0],Y_A.shape[1]+Y_B.shape[1],Y_A.shape[2]+Y_B.shape[2]), dtype=complex)

    Y_AnB[:,:Y_A.shape[1],:Y_A.shape[2]] = Y_A
    Y_AnB[:,Y_A.shape[1]:,Y_A.shape[2]:] = Y_B

Next the compatibility and the equilibrium conditions has to be defined through the signed Boolean matrices ``Bu`` and ``Bf``. 

.. math::
    \mathbf{B}_\text{u}\,\boldsymbol{u} = \mathbf{0}

.. math::
    \boldsymbol{g} = - \mathbf{B}_\text{f}^\text{T} \boldsymbol{\lambda}

Make sure that the correct DoFs are selected for the coupling. In the following example the 6 virtual/generalized DoFs at the interface are matched, 
so the size of the Boolean matrix should be 6  ×  30 (30 is the sum of all DoFs from both substructures A and B).

.. code-block:: python

    k = 6

    Bu = np.zeros((k,Y_A.shape[1]+Y_B.shape[1]))
    Bu[:k,6:6+k] = 1*np.eye(k)
    Bu[:k,12:12+k] = -1*np.eye(k)

    Bf = np.zeros((k,Y_A.shape[2]+Y_B.shape[2]))
    Bf[:k,6:6+k] = 1*np.eye(k)
    Bf[:k,12:12+k] = -1*np.eye(k)

.. figure:: ./../data/Bu.png
   :width: 500px

.. figure:: ./../data/Bf.png
   :width: 500px
    
    
For the LM FBS method, having defined :math:`\mathbf{Y^{\text{A|B}}}`, :math:`\mathbf{B}_\text{u}` and :math:`\mathbf{B}_\text{f}` is already sufficient to perform coupling:

.. math::
    \mathbf Y^{\text{AB}} = \mathbf Y^{\text{A|B}} - \mathbf Y^{\text{A|B}}\,\mathbf B^\mathrm{T} \left( \mathbf B \mathbf Y^{\mathrm{A|B}} \mathbf{B}^\mathrm{T} \right)^{-1} \mathbf B \mathbf Y^\text{A|B}

.. code-block:: python

    Y_ABn = np.zeros_like(Y_AnB,dtype = complex)

    Y_int = Bu @ Y_AnB @ Bf.T
    Y_ABn = Y_AnB - Y_AnB @ Bf.T @ np.linalg.pinv(Y_int) @ Bu @ Y_AnB
    
Results
*************

First extract the FRFs at the reference DoFs:

.. code-block:: python

    arr_coup = [0,1,2,3,4,5,18,19,20,21,22,23,24,25,26,27,28,29]
    Y_AB_coupled = Y_ABn[:,arr_coup,:][:,:,arr_coup]
    Y_AB_ref = MK_AB.FRF
    
The coupled and the reference results can then be compared and evaluated:
    
.. raw:: html

   <iframe src="../../_static/VPT_coupling.html" height="500px" width="750px" frameborder="0"></iframe>

.. panels::
    :column: col-12 p-3

    **That's a wrap!**
    ^^^^^^^^^^^^

    Want to know more, see a potential application? Contact us at info.pyfbs@gmail.com!

.. rubric:: References

.. [1] de Klerk D, Rixen DJ, Voormeeren SN. General framework for dynamic substructuring: history, review and classification of techniques. AIAA journal. 2008 May;46(5):1169-81.
.. [2] van der Seijs MV, van den Bosch DD, Rixen DJ, de Klerk D. An improved methodology for the virtual point transformation of measured frequency response functions in dynamic substructuring. In4th ECCOMAS thematic conference on computational methods in structural dynamics and earthquake engineering 2013 Jun (No. 4).

    