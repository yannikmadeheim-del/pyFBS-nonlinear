#############
Pseudo-forces
#############

Pseudo-forces coming soon!

.. note:: 
   Download example showing a numerical example of the pseudo-forces: :download:`17_pseudo-forces.ipynb <../../../examples/16_TPA_pseudo_forces.ipynb>`

..
   What is OPTA?
   ******************************

   Consider a system of substructures A and B, coupled at the interface, as depicted below.
   Substructure A is treated as an active component with operational excitation acting in :math:`\boldsymbol{u}_1`. 
   Meanwhile, no excitation force is acting on passive substructure B. 
   Responses in :math:`\boldsymbol{u}_3`, :math:`\boldsymbol{u}_4`, and also in interface DoFs :math:`\boldsymbol{u}_2` are hence a consequence of active force :math:`\boldsymbol{f}_1` only. 

   .. figure:: ./../data/in_situ.png
      :width: 250px
      :align: center

   Using LM-FBS notation, responses at the indicator sensors :math:`\boldsymbol{u}_4` can be expressed in terms of subsystem admittances [1]_:

   .. math::

      \boldsymbol{u}_4 = \textbf{Y}_{41}^{\text{AB}} \boldsymbol{f}_1 = \textbf{Y}_{42}^{\text{B}} \underbrace{ \Big(\textbf{Y}_{22}^{\text{A}} + \textbf{Y}_{22}^{\text{B}}\Big)^{-1} \textbf{Y}_{21}^{\text{A}} \boldsymbol{f}_1 }_{\boldsymbol{g}_2^{\text{B}}}.

   Expressing :math:`\boldsymbol{g}_2^{\mathrm{B}}` yields:

   .. math::

      \boldsymbol{g}_2^{\text{B}} = \Big( \textbf{Y}_{42}^{\text{B}} \Big)^+ \boldsymbol{u}_4.

   Number of indicator responses :math:`\boldsymbol{u}_4` should preferably exceed number of interface forces :math:`\boldsymbol{g}_2^{\mathrm{B}}` (or be at least equal).

   How to calculate equivalent forces?
   ***********************************

   In order to determine equivalent forces, the following steps should be performed:

   1. Measurement of admittance matrices :math:`\textbf{Y}_{42}^{\text{B}}` and :math:`\textbf{Y}_{32}^{\text{B}}` (note that for this assembly must be taken apart and only passive side is considered).
   2. Measurement of responses :math:`\boldsymbol{u}_4` on an assembly subjected to the operational excitation.

   Virtual Point Transformation
   ============================

   To simplify the measurement of the :math:`\textbf{Y}_{42}^{\text{B}}` and :math:`\textbf{Y}_{32}^{\text{B}}` the VPT can be applied on the interface excitation to transform forces at the interface into virtual DoFs (from :math:`\textbf{Y}_{\mathrm{uf}}` to :math:`\textbf{Y}_{\mathrm{um}}`): 

   .. math::

      \textbf{Y}_{\text{um}} = \textbf{Y}_{\text{uf}} \, \textbf{T}_{\text{f}}.

   For the VPT, positional data is required for channels (``df_chn_up``), impacts (``df_imp_up``) and for virtual points  (``df_vp`` and ``df_vpref``):

   .. code-block:: python

      df_imp_B = pd.read_excel(xlsx_pos, sheet_name='Impacts_B')
      df_chn_B = pd.read_excel(xlsx_pos, sheet_name='Channels_B')
      df_vp = pd.read_excel(xlsx_pos, sheet_name='VP_Channels')
      df_vpref = pd.read_excel(xlsx_pos, sheet_name='VP_RefChannels')

      vpt_B = pyFBS.VPT(df_chn_B, df_imp_B, df_vp, df_vpref)

   Defined force transformation is then applied on the FRFs and requried admittance matrices :math:`\textbf{Y}_{42}^{\text{B}}` and :math:`\textbf{Y}_{32}^{\text{B}}` are extracted as follows:

   .. code-block:: python

      Y42_B = MK_B.FRF[:,:9,:9] @ vpt_B.Tf
      Y32_B = MK_B.FRF[:,9:12,:9] @ vpt_B.Tf

   For more options and details about :mod:`pyFBS.VPT` see the :download:`04_VPT.ipynb <../../../examples/04_VPT.ipynb>` example.

   Calculation of interface forces
   ================================

   Interface forces are calculated in the following manner:

   .. code-block:: python

      g2_B = np.linalg.pinv(Y42_B) @ u4

   On-board validation
   ===================

   Finally, interface forces are applied to build up predicted response at 
   Completeness of the interface forces is then evaluated via comparison of predicted and actual response using on-board validation:

   .. code-block:: python

      u3_tpa = Y32_B @ g2_B

      o = 0

      u3 = plot_frequency_response(freq, np.hstack((u3_tpa[:,o:o+1], u3_op[:,o:o+1])))

   .. raw:: html

      <iframe src="../../_static/on_board_matrix_inverse.html" height="460px" width="100%" frameborder="0"></iframe>

   .. [1] Van der Seijs, M. V. "Experimental dynamic substructuring: Analysis and design strategies for vehicle development." (2016).