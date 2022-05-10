========
Plotting
========

The :mod:`pyFBS` package offers wrapper functions for creating interactive plots aimed at user-friendly display of quantities common for
structural dynamic analyses. Wrapper functions utilize `Altair <https://altair-viz.github.io/>`_ python package. 

.. tip::
    All plots above are interactive for you to try them out! Use mouse scroll to zoom in/out on figures.

Plotting FRFs
*************

For plotting frequency response function (also known as holy grail in pyFBS) several types of plots can be exploited, depending on the size of the FRF matrix. 
Commonly, FRFs are measured at several excitation points, while the output is measured using multiple transducers. Full FRF matrix can be neatly displayed using the 
following function passing frequency vector and 3D FRF matrix:


.. code-block:: python

    pyFBS.plot_FRF(freq, FRF_matrix)

.. raw:: html

   <iframe src="../../_static/FRF_VP.html" height="500px" width="750px" frameborder="0"></iframe>

.. tip::
    To many FRFs? Click on corresponding circles to display only individual ones. By holding ``shift`` while clicking, multiple FRFs can be displayed.

When directly comparing two or more FRFs (or other quantities such as responses etc.), ``plot_frequency_response`` can be used. Required arguments are frequency vector 
and 3D FRF matrix, composed of FRFs you want to display. You can manually insert labels to easiliy differentiate between curves:

.. code-block:: python

    o = 3
    i = 0

    pyFBS.plot_frequency_response(freq, 
                np.hstack((MK.FRF_noise[:,o:o+1,i:i+1], MK.FRF[:,o:o+1,i:i+1], Y_B_exp[:,o:o+1,i:i+1])),
                labels=('Num. FRF + noise', 'Num. FRF', 'Exp. FRF'))
	
.. raw:: html

   <iframe src="../../_static/FRF_synth.html" height="460px" width="100%" frameborder="0"></iframe>

.. tip::
    Click on labels in legend to show/hide individual curves.

Plotting coherence
******************

You can use ``plot_coh`` to neatly plot frequency dependant coherence criterion between two functions:

.. code-block:: python

    pyFBS.plot_coh(vpt.freq, vpt.overall_sensor, color='steelblue', title='Overall Channel Consistency')

.. raw:: html

   <iframe src="../../_static/VP_spec_cons.html" height="290px" width="600px" frameborder="0"></iframe>

For cases where you compare several pairs of functios, you can use ``plot_coh_group``. On the left, you can see averaged coherence value for each pair, 
while on the right, frequency dependant coherence are plotted:

.. code-block:: python

    pyFBS.plot_coh_group(freq, coh_data)
	
.. raw:: html

   <iframe src="../../_static/on_board_coherence.html" height="340px" width="750px" frameborder="0"></iframe>

.. tip::
    Circle sizes is proportional to the averaged coherence value.

Commonly used plots
*******************

For simple plotting of curves on a linear scale, the following wrapper can be used:

.. code-block:: python

    pyFBS.comparison_plot(time_data, np.expand_dims(acc_data, axis=(1,2)),
                         x_label='Time', y_label='Acceleration', labels='acc1')
	
.. raw:: html

   <iframe src="../../_static/comparison_plot.html" height="320px" width="100%" frameborder="0"></iframe>

.. note::
    Even if you are plotting single curves, y data must be in a form of a 3D matrix. This is due to the optimization of the wrappers for plotting multiple functions.

All of us use barchart plots for multiple purposes quite frequently. 
How about a barchart with interactive mouse tool-tip to quickly assess bar values? We got you there:

.. code-block:: python

    pyFBS.barchart(np.arange(1,10,1), vpt_AB.specific_impact, 
                    color='firebrick', title='Specific Impact Consistency')

.. raw:: html

    <iframe src="../../_static/specific_impact_consistency.html" height="300px" width="295px" frameborder="0"></iframe>

.. tip::
    Barchart plot is very useful for assessing channel/impact consistency in VPT.

Another commonly used plot is an image-show plot. Pass a 2D matrix with real entries and define colormap of your own (we prefer jet):

.. code-block:: python

    pyFBS.imshow(coh_crit)

.. raw:: html

    <iframe src="../../_static/VP_rec.html" height="270px" width="350px" frameborder="0"></iframe>

.. tip::
    Using imshow, you can quickly estimate FRF matrix reciprocity.

Similar to imshow is a contour plot. Required arguments in this case are x and y coordinates (1D array with real entries) 
and z values in form of 2D array with real entries. 

.. code-block:: python

    pyFBS.contour_plot(time, freq, np.abs(Acc_f_dB))

.. raw:: html

    <iframe src="../../_static/run_up_wm.html" height="300px" width="380px" frameborder="0"></iframe>

.. tip::
    Contour plot wrapper is perfect to use for run-up diagrams, as you can see from example above.

.. panels::
    :column: col-12 p-3

    **That's a wrap!**
    ^^^^^^^^^^^^

    Want to know more, see a potential application? Contact us at info.pyfbs@gmail.com!