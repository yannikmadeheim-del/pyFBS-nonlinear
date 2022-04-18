Methods and examples
====================

Here you can find a gallery of examples demonstrating the capabilities of pyFBS. 
Besides the functionalities, already incorporated in pyFBS, you can also find basic theoretical principles of the methods used for these examples. 
If any of the topics interests you, all methods are accompanied with several references, that lead you to more detailed theoretical backgrounds or more advanced applications. 
Have fun exploring FBS field!

Basic examples
**************

This examples show how to use basic features of pyFBS. Explore this basic examples to get familiar with the pyFBS workflow.

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Static display">

.. only:: html

    .. figure:: ./data/interaction.gif	   
       :target: ./basic_examples/01_static_display.html

       Static display

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./basic_examples/01_static_display


  
 
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Interactive positioning">

.. only:: html

    .. figure:: ./data/snapping.gif	   
       :target: ./basic_examples/02_interactive_display.html

       Interactive positioning

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./basic_examples/02_interactive_display
   


   
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="FRF synthetization">

.. only:: html

    .. figure:: ./data/FRF_syn-FRF-visualization.svg 
       :target: ./basic_examples/03_FRF_synthetization.html

       FRF synthetization

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./basic_examples/03_FRF_synthetization




Frequency Based Substructuring
******************************
Structural dynamic analyses can be carried out more efficiently if complex systems are divided into smaller subsystems, analysed separately, and later coupled using dynamic substructuring (DS) methods.
In terms of the modeling domain, a frequency-based substructuring (FBS) is often preferred by experimentalists due to its ease of use and implementation with directly measured Frequency Response Functions (FRFs). 
These examples show state-of-the-art techniques to successfully couple or decouple substructures using FBS framework.


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="General Theory">

.. only:: html

    .. figure:: ./data/assembly_tn.svg  
       :target: ./fbs/frequency_based_substructuring.html

       General theory

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./fbs/frequency_based_substructuring





.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Virtual Point Transformation">

.. only:: html

    .. figure:: ./data/vpt_scheme.svg   
       :target: ./fbs/virtual_point_transformation.html

       Virtual Point Transformation

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./fbs/virtual_point_transformation

   


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Singular Vector Transformation">

.. only:: html

    .. figure:: ./data/SVT.png   
       :target: ./fbs/singular_vector_transformation.html

       Singular Vector Transformation

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./fbs/singular_vector_transformation



Transfer Path Analysis
**********************
Transfer-path analysis (TPA) is a reliable and effective diagnostic tool for the characterization of actively vibrating components and the propagation of noise and vibrations to the connected passive substructures. 
TPA offers the ability to analyse the vibration transfer between the individual components of the assembly, distinguish the partial transfer-path contribution and predict the receiver's response.

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Clasiccal TPA">

.. only:: html

    .. figure:: ./data/classical_tpa.svg   
       :target: ./tpa/classical_tpa.html

       Clasiccal TPA

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./tpa/classical_tpa





.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Component-based TPA">

.. only:: html

    .. figure:: ./data/component-based_tpa.svg   
       :target: ./tpa/component-based_tpa.html

       Component-based TPA

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./tpa/component-based_tpa





.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Transmssibility-based TPA">

.. only:: html

    .. figure:: ./data/otpa.svg   
       :target: ./tpa/transmissibility-based_tpa.html

       Transmissibility-based TPA

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./tpa/transmissibility-based_tpa






Dynamic expansion
*****************
A high-resolution dynamic response is important for characterizing a system's dynamic properties. 
Measurements involving a limited number of points on the structure can be expanded to unmeasured points through approximation or model-based expansion techniques that rely on the introduction of a numerical model.

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="System Equivalent Model Mixing">

.. only:: html

    .. figure:: ./data/semm_scheme.svg   
       :target: ./dynamic_expansion/system_equivalent_model_mixing.html

       System Equivalent Model Mixing

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./dynamic_expansion/system_equivalent_model_mixing

|
|
|
|
|
|
|
|



Modal identification
********************
The response of a system can often be represented by much fewer variables in the modal domain. 
A feature of pyFBS, multi-reference modal identification method, enables you to perform experimental or operational modal analysis 
and identify modal parameters from experimental dynamic models.



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Experimental Modal Analysis">

.. only:: html

    .. figure:: ./data/modal_id_tn.jpg 
       :target: ./modal_id/20_EMA.html

       Experimental Modal Analysis

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./modal_id/20_EMA


|
|
|
|
|
|
|
|




Case studies
************
These examples show applications of the pyFBS on more complex problems. Explore this application examples to see how pyFBS can be used on more complex dynamic problems.

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Operational Deflection Shapes">

.. only:: html

    .. figure:: ./data/ods.gif	   
       :target: ./case_studies/06_ODS.html

       Operational Deflection Shapes

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./case_studies/06_ODS


  
 
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Transmission Simulator">

.. only:: html

    .. figure:: ./data/ten_display_four.png   
       :target: ./case_studies/10_TS.html

       Transmission Simulator

.. raw:: html

    </div>

.. toctree::
   :hidden:

   ./case_studies/10_TS

|
|
|
|
|
|
|
|
|