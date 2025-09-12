Examples
====================

Here you can find a gallery of examples demonstrating the capabilities of pyFBS. Have fun exploring FBS field!

Basic examples
**************

This examples show how to use basic features of pyFBS. Explore this basic examples to get familiar with the pyFBS workflow.

.. only:: html

     .. figure:: ./data/interaction.gif
         :height: 150px
         :target: ./basic_examples/01_static_display.html

         Static display

.. toctree::
   :hidden:

   ./basic_examples/01_static_display




.. only:: html

     .. figure:: ./data/snapping.gif
         :height: 150px
         :target: ./basic_examples/02_interactive_display.html

         Interactive positioning

.. toctree::
   :hidden:

   ./basic_examples/02_interactive_display




.. only:: html

     .. figure:: ./data/FRF_syn-FRF-visualization.svg
         :height: 150px
         :target: ./basic_examples/03_FRF_synthetization.html

         FRF synthetization

.. toctree::
   :hidden:

   ./basic_examples/03_FRF_synthetization




.. only:: html

     .. figure:: ./data/plot.png
         :height: 150px
         :target: ./basic_examples/plotting.html

         Plotting with wrapper functions

.. toctree::
   :hidden:

   ./basic_examples/plotting



Interface modelling
*******************

In experimental dynamic substructuring, coupling of substructures sharing a line- or surface-like interface proves to be a challenge due to the 
difficulties in interface modelling. Modelling a high number of degrees of freedom at the common interface can be too stringent when imposing 
compatibility and equilibrium conditions, thereby causing redundancy and ill-conditioning. These examples show state-of-the-art techniques to 
establish an interface model composed of significant degrees of freedom.


.. only:: html

     .. figure:: ./data/pic_vpt.png
         :height: 150px
         :target: ./fbs/04_VPT.html

         VPT example

.. toctree::
   :hidden:

   ./fbs/04_VPT



.. only:: html

     .. figure:: ./data/SVT.png
         :height: 150px
         :target: ./fbs/12_SVT.html

         SVT example

.. toctree::
   :hidden:

   ./fbs/12_SVT



Frequency Based Substructuring
******************************

Structural dynamic analyses can be carried out more efficiently if complex systems are divided into smaller subsystems, analysed separately, 
and later coupled using dynamic substructuring (DS) methods. In terms of the modeling domain, a frequency-based substructuring (FBS) is often 
preferred by experimentalists due to its ease of use and implementation with directly measured Frequency Response Functions (FRFs). 
These examples show state-of-the-art techniques to successfully couple or decouple substructures using FBS framework.

.. only:: html

     .. figure:: ./data/vp_a_b_ab.png
         :height: 150px
         :target: ./fbs/07_coupling.html

         FBS Coupling with VPT

.. toctree::
   :hidden:

   ./fbs/07_coupling


.. only:: html

     .. figure:: ./data/decoupling.png
         :height: 150px
         :target: ./fbs/08_decoupling.html

         FBS Decoupling with VPT

.. toctree::
   :hidden:

   ./fbs/08_decoupling


.. only:: html

     .. figure:: ./data/SVT.png
         :height: 150px
         :target: ./fbs/13_SVT_decoupling.html

         FBS Decoupling with SVT

.. toctree::
   :hidden:

   ./fbs/13_SVT_decoupling



Transfer Path Analysis
**********************

Transfer-path analysis (TPA) is a reliable and effective diagnostic tool for the characterization of actively vibrating components and the 
propagation of noise and vibrations to the connected passive substructures. TPA offers the ability to analyse the vibration transfer between 
the individual components of the assembly, distinguish the partial transfer-path contribution and predict the receiver's response.

.. only:: html

     .. figure:: ./data/classical_tpa.svg
         :height: 100px
         :target: ./tpa/classical_tpa.html

         Classical TPA

.. toctree::
   :hidden:

   ./tpa/classical_tpa



.. only:: html

     .. figure:: ./data/component-based_tpa.svg
         :height: 100px
         :target: ./tpa/component-based_tpa.html

         Component-based TPA

.. toctree::
   :hidden:

   ./tpa/component-based_tpa




.. only:: html

     .. figure:: ./data/otpa.svg
         :height: 100px
         :target: ./tpa/transmissibility-based_tpa.html

         Transmissibility-based TPA

.. toctree::
   :hidden:

   ./tpa/transmissibility-based_tpa



Dynamic expansion
*****************

A high-resolution dynamic response is important for characterizing a system's dynamic properties. Measurements involving a limited number of 
points on the structure can be expanded to unmeasured points through approximation or model-based expansion techniques that rely on the introduction 
of a numerical model.


.. only:: html

     .. figure:: ./data/semm_scheme.svg
         :height: 150px
         :target: ./dynamic_expansion/system_equivalent_model_mixing.html

         System Equivalent Model Mixing in Frequency Domain (SEMM)

.. toctree::
   :hidden:

   ./dynamic_expansion/system_equivalent_model_mixing



.. only:: html

     .. figure:: ./data/serep_scheme.svg
         :height: 150px
         :target: ./dynamic_expansion/21_SEREP.html

         System Equivalent Reduction Expansion Process (SEREP)

.. toctree::
   :hidden:

   ./dynamic_expansion/21_SEREP



Modal identification
********************

The response of a system can often be represented by much fewer variables in the modal domain. A feature of pyFBS, multi-reference modal identification method, 
enables you to perform experimental or operational modal analysis and identify modal parameters from experimental dynamic models.


.. only:: html

     .. figure:: ./data/modal_id_tn.jpg
         :height: 150px
         :target: ./modal_id/20_EMA.html

         Experimental Modal Analysis

.. toctree::
   :hidden:

   ./modal_id/20_EMA



Case studies
************

These examples show applications of the pyFBS on more complex problems. Explore this application examples to see how pyFBS can be used on more complex dynamic problems.


.. only:: html

     .. figure:: ./data/ods.gif
         :height: 150px
         :target: ./case_studies/06_ODS.html

         Operational Deflection Shapes

.. toctree::
   :hidden:

   ./case_studies/06_ODS



.. only:: html

     .. figure:: ./data/ten_display_four.png
         :height: 150px
         :target: ./case_studies/10_TS.html

         Transmission Simulator

.. toctree::
   :hidden:

   ./case_studies/10_TS




.. only:: html

     .. figure:: ./data/AJB_.png
         :height: 150px
         :target: ./case_studies/23_rubber_mount.html

         Rubber Mount Characterization

.. toctree::
   :hidden:

   ./case_studies/23_rubber_mount
