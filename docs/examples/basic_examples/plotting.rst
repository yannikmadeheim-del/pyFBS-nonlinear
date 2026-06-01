========
Plotting
========

The :mod:`pyFBS` package offers wrapper functions for creating interactive plots aimed at a user-friendly display of quantities common for
structural dynamic analyses. Wrapper functions utilize the `Altair <https://altair-viz.github.io/>`_ python package and are designed to distinctly present 
important analysis results to the reader.

.. tip::
    All plots presented within this tutorial are interactive for you to try them out! Use mouse scroll to zoom in/out on figures, hoover mouse tip over plots 
    to see the interactive tool-tip messages, and explore capabilities of such plotting.

Plotting FRFs
*************

For plotting frequency response functions (also known as a holy grail in pyFBS) several types of plots can be exploited, depending on the size of the FRF matrix. 
Commonly, FRFs are measured for several excitation points, while the output is measured using multiple transducers. Full FRF matrix can be neatly displayed using the 
following function by passing frequency vector and 3D FRF matrix:

.. code-block:: python

    pyfbs.display.plot_frf(freq, FRF_matrix)

.. raw:: html

   <iframe src="../../_static/FRF_VP.html" height="480px" width="750px" frameborder="0"></iframe>

.. tip::
    To many FRFs? Click on corresponding circles to display individual FRF only. By holding ``shift`` while clicking, multiple FRFs can be displayed.

When directly comparing two or more FRFs (or other quantities such as responses etc.), ``plot_frequency_response`` can be used. Required arguments are frequency vector 
and 3D FRF matrix, composed of FRFs you want to display. You can manually insert labels to easily differentiate between curves:

.. code-block:: python

    o = 3
    i = 0

    pyfbs.display.plot_frequency_response(freq, 
                np.hstack((MK.FRF_noise[:,o:o+1,i:i+1], MK.FRF[:,o:o+1,i:i+1], Y_B_exp[:,o:o+1,i:i+1])),
                labels=('Num. FRF + noise', 'Num. FRF', 'Exp. FRF'))

.. note::
    Even if you are plotting a single FRF, FRF data must be in a form of a 3D matrix. This is due to the optimization of the wrappers for plotting multiple functions.
	
.. raw:: html

   <iframe src="../../_static/FRF_synth.html" height="460px" width="100%" frameborder="0"></iframe>

.. tip::
    Click on labels in the legend to show/hide individual curves.

Plotting coherence
******************

You can use ``plot_coh`` to neatly plot frequency dependant coherence criterion between two functions:

.. code-block:: python

    pyfbs.display.plot_coh(freq, vpt.overall_impact, color='firebrick', title='Overall Impact Consistency')

.. raw:: html

    <iframe src="../../_static/overall_impact_consistency.html" height="290px" width="600px" frameborder="0"></iframe>

For cases where you compare several pairs of functions, you can use ``plot_coh_group``. On the left, you can see averaged coherence value for each pair, 
while on the right, frequency dependant coherence is plotted:

.. code-block:: python

    pyfbs.display.plot_coh_group(freq, coh_data)
	
.. raw:: html

    <iframe src="../../_static/on_board_coherence.html" height="340px" width="750px" frameborder="0"></iframe>

.. tip::
    Circle sizes are proportional to the averaged coherence value to quickly differentiate between the DoFs with high and low coherence.

Commonly used plots
*******************

For simple plotting of curves on a linear scale, the following wrapper can be used:

.. code-block:: python

    pyfbs.display.comparison_plot(time_data, np.expand_dims(acc_data, axis=(1,2)),
                         x_label='Time', y_label='Acceleration', labels='acc1')
	
.. raw:: html

   <iframe src="../../_static/comparison_plot.html" height="320px" width="100%" frameborder="0"></iframe>

All of us use barcharts for multiple purposes quite frequently. 
How about a barchart plot with an interactive mouse tool-tip to quickly assess bar values? We got you there:

.. code-block:: python

    pyfbs.display.barchart(np.arange(1,10,1), vpt_AB.specific_impact, 
                    color='firebrick', title='Specific Impact Consistency')

.. raw:: html

    <iframe src="../../_static/specific_impact_consistency.html" height="280px" width="295px" frameborder="0"></iframe>

.. tip::
    Barchart plot is very useful for assessing channel/impact consistency in VPT.

Another commonly used plot is an image-show plot. Pass a 2D matrix with real entries and define a colormap of your own (we prefer jet):

.. code-block:: python

    pyfbs.display.imshow(coh_crit)

.. raw:: html

    <iframe src="../../_static/VP_rec.html" height="260px" width="350px" frameborder="0"></iframe>

.. tip::
    Using ``imshow``, you can quickly estimate FRF matrix reciprocity.

Similar to ``imshow`` is a contour plot. Required arguments in this case are ``x`` and ``y`` coordinates (1D array with real entries) 
and z values in form of a 2D array with real entries. 

.. code-block:: python

    pyfbs.display.contour_plot(time, freq, np.abs(Acc_f_dB))

.. raw:: html

    <iframe src="../../_static/run_up_wm.html" height="270px" width="380px" frameborder="0"></iframe>

.. tip::
    Contour plot wrapper is perfect to use for run-up diagrams, as you can see from the example above.

.. card:: That's a wrap!

    Want to know more, see a potential application? Contact us at info.pyfbs@gmail.com!