System Equivalent Reduction Expansion Process
=============================================

`Short intro of SEREP`. 

.. note:: 
   Download example showing the basic use of SEREP: :download:`21_SEREP.ipynb <../../../examples/21_expansion_methods.ipynb>`

`basic theory.` [1]_

Example data import
*******************

In the beginning, it necessary to define numerical and experimental data (response models). 
Datasets used in this example are from a laboratory testbench and are available directly within the :mod:`pyFBS`.

.. code-block:: python

   # number of modes
   # limited by m_par <= n_b, where n_b = 10
   m_par_SEREP = 9 
   # overlay - 2 kHz limit
   m_ov = 15

Experimental model
------------------
`Short description`

.. code-block:: python

   eig_val_ov, xi_ov, eig_vec_ov = MK_O.transform_modal_parameters(df_chn_O, 
                                                                limit_modes = m_ov, 
                                                                modal_damping = dam_ov, 
                                                                return_channel_only = True)


Numerical model
---------------
`Short description`

.. code-block:: python

   eig_val_par_SEREP, xi_par_SEREP, eig_vec_par_SEREP = MK_P.transform_modal_parameters(df_chn_P,
                                                                   limit_modes = m_par_SEREP, 
                                                                   modal_damping = dam_par, 
                                                                   return_channel_only = True)

.. figure:: ./../data/serep.png
   :width: 800px

Application of SEREP
********************

The result is a hybrid model that contains the DoFs represented in the numerical model.

.. code-block:: python

   eig_vec_serep = pyFBS.SEREP(eig_vec_par_SEREP, eig_vec_ov, df_chn_P, df_chn_O)

.. tip::
   Condition number!

Finally, the results of the hybrid model can be compared with the reference using MAC.

.. raw:: html

   <iframe src="../../_static/serep_mac.html" height="270px" width="750px" frameborder="0"></iframe>

.. panels::
    :column: col-12 p-3

    **That's a wrap!**
    ^^^^^^^^^^^^

    Want to know more, see a potential application? Contact us at info.pyfbs@gmail.com!
   
.. rubric:: References

.. [1] O'CALLAHAN JC. System equivalent reduction expansion process. InProc. of the 7th Inter. Modal Analysis Conf., 1989.
