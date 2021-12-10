System Equivalent Model Mixing
==============================

System Equivalent Model Mixing (SEMM) [1]_ enables the mixing of equivalent models into a hybrid model in the frequency domain. 
The models used can either be of numerical or experimental nature. 
The overlay model provides the dynamic properties which are expanded to the DoFs of the parent model. 
Therefore the overlay model is usually represented by the experimental model and parent model with the numerical model.

.. note:: 
   Download example showing the basic use of SEMM: :download:`05_SEMM.ipynb <../../../examples/05_SEMM.ipynb>`

..
   DoF-set of parent model is contained from internal (i) and boundary (b) DoFs. 
   Boundary DoFs must overlap with the overlay model so the dynamic coupling can be performed, while the internal DoFs of the parent model can be unique to its own. 
   The equivalent models, appearing in the SEMM method, are arranged by separating internal and boundary DoFs in the admittance matrices:

   .. math::

      \begin{equation}\label{parent_overlay}
      \mathbf{Y}^{\text{par}}=
      \begin{bmatrix}
      \mathbf{Y}_{\text{ii}}&\mathbf{Y}_{\text{ib}}\\
      \mathbf{Y}_{\text{bi}}&\mathbf{Y}_{\text{bb}}
      \end{bmatrix}^{\text{par}},\quad
      \mathbf{Y}^{\text{ov}}=
      \begin{bmatrix}
      \mathbf{Y}_{\text{bb}}
      \end{bmatrix}^{\text{ov}},\quad
      \mathbf{Y}^{\text{rem}}=
      \begin{bmatrix}
      \mathbf{Y}_{\text{bb}}
      \end{bmatrix}^{\text{rem}}.
      \end{equation}

   After satisfying compatibility and equilibrium conditions between equivalent models, the basic form of the SEMM method is defined using the equation:

   .. math::

      \mathbf{Y}^{\text{SEMM}}=
      \begin{bmatrix}
      \mathbf{Y}
      \end{bmatrix}^{\text{par}}
      -
      \begin{bmatrix}
      \mathbf{Y}_{\text{ib}}\\
      \mathbf{Y}_{\text{bb}}
      \end{bmatrix}^{\text{par}}
      %\,
      \left( \mathbf{Y}^{\text{rem}}\right) ^{-1}
      %\,
      \left( \mathbf{Y}^{\text{rem}}-\mathbf{Y}^{\text{ov}}\right)
      %\,
      \left( \mathbf{Y}^{\text{rem}}\right)^{-1}
      %\,
      \begin{bmatrix}
      \mathbf{Y}_{\text{bi}}&\mathbf{Y}_{\text{bb}}
      \end{bmatrix}^{\text{par}}

   By extending the removed model to all DoFs of numerical model, the fully extend formulation of SEMM method follows equation:

   .. math::

      \mathbf{Y}^{\text{SEMM}}=
      \mathbf{Y}^{\text{par}}
      -
      \mathbf{Y}^{\text{par}}
      %\,
      \left( \begin{bmatrix} \mathbf{Y}_{\text{bi}}&\mathbf{Y}_{\text{bb}}\end{bmatrix}^{\text{rem}}\right ) ^{+}
      %\,
      \left( \mathbf{Y}^{\text{rem}}_{\text{bb}}-\mathbf{Y}^{\text{ov}}\right)
      %\,
      \left( \begin{bmatrix} \mathbf{Y}_{\text{ib}}\\ \mathbf{Y}_{\text{bb}}\end{bmatrix}^{\text{rem}}\right ) ^{+}
      %\,
      \mathbf{Y}^{\text{par}}

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


Numerical model
---------------
FRFs used for the parent model numerical model can be imported to Python or generated with the mass and stiffness matrix using :mod:`pyFBS.MK_model.FRF_synth`.
Locations and directions for which FRFs are generated are defined in an Excel file and can be parsed into a :mod:`pandas.DataFrame`.

.. code-block:: python

   stl = r"./lab_testbench/STL/AB.stl"
   xlsx = r"./lab_testbench/Measurements/AM_measurements.xlsx"

   full_file = r"./lab_testbench/FEM/AB.full"
   rst_file = r"./lab_testbench/FEM/AB.rst"

   MK = pyFBS.MK_model(rst_file, full_file, no_modes = 100, recalculate = False)

   df_chn = pd.read_excel(xlsx, sheet_name='Channels_AB')
   df_imp = pd.read_excel(xlsx, sheet_name='Impacts_AB')

   MK.FRF_synth(df_chn,df_imp, 
                f_start=0,
                f_end=2002.5,
                f_resolution=2.5,
                modal_damping = 0.003,
                frf_type = "accelerance")

As experimental, also the numerical model must be properly arranged. The first dimension represents the frequency depth, the second the response points, and the third excitation points. 
Additionally, the frequency resolution of the numerical and experimental model has to be the same.

The numerical model is not necessarily a square matrix, as it can also be rectangular. 
But the numerical model must contain at least all the DoFs contained in the experimental model.

Application of SEMM
*******************

The function enables the implementation of three SEMM method formulations: ``basic``, ``fully-extend`` and ``fully-extend-svd``, the choice of which is defined with the ``SEMM_type`` parameter.
The ``red_comp`` and ``red_eq`` parameters can be used to influence the number of eigenvalues used to ensure equilibrium and compatibility conditions when the ``fully-extend-svd`` formulation is used [2]_.

The result is a hybrid model that contains the DoFs represented in the numerical model.

In the example below, only a part of the experimental response matrix is included in SEMM. The remaining DoFs are used to evaluate SEMM.
It is essential that the order of measurements in the experimental model ``Y_exp`` coincides with the order of measurements in the parameters ``df_chn_exp`` and ``df_imp_exp`` and the same must be valid for the numerical model. 
Function :mod:`pyFBS.SEMM` will automatically match corresponding DoFs.

.. code-block:: python

   Y_AB_SEMM = pyFBS.SEMM(MK.FRF, Y_exp[:, 0:15, 5:20],
                          df_chn_num = df_chn, 
                          df_imp_num = df_imp, 
                          df_chn_exp = df_chn[0:15], 
                          df_imp_exp = df_imp[5:20], 
                          SEMM_type='fully-extend-svd', red_comp=10, red_eq=10)

Finally, the results of the hybrid model can be compared with the reference experimental and the numerical model.

..
   .. code-block:: python

      s1 = 24
      s2 = 24

      display(df_chn.iloc[[s1]])
      display(df_imp.iloc[[s2]])

      plt.figure(figsize = (12,8))

      plt.subplot(211)
      plt.semilogy(MK.freq,np.abs(MK.FRF[:,s1,s2]), label = "Num.")
      plt.semilogy(freq,np.abs(Y_exp[:, s1,s2]), label = "Exp.")
      plt.semilogy(freq,np.abs(Y_AB_SEMM[:, s1,s2]), label = "SEMM")
      plt.ylabel("Accelerance [m/s$^2$/N]")
      plt.legend()

      plt.subplot(413)
      plt.plot(MK.freq,np.angle(MK.FRF[:,s1,s2]))
      plt.plot(freq,np.angle(Y_exp[:, s1,s2]))
      plt.plot(MK.freq,np.angle(Y_AB_SEMM[:,s1,s2]))
      plt.xlabel("f [Hz]")
      plt.ylabel("Angle [rad]")
   
.. raw:: html

   <iframe src="../../_static/SEMM_plot.html" height="500px" width="750px" frameborder="0"></iframe>

.. tip::
	That's a wrap! Want to know more, see a potential application? Contact us at info.pyfbs@gmail.com!
   
.. rubric:: References

.. [1] Steven WB Klaassen, Maarten V. van der Seijs, and Dennis de Klerk. System equivalent model mixing. Mechanical Systems and Signal Processing, 105:90–112, 2018.
.. [2] Steven WB Klaassen and D. J. Rixen. The Inclusion of a Singular-value Based Filter in SEMM. in: Proceedings of the 38th International Modal Analysis Conference, A Conference on Structural Dynamics, (2020), 2020.
