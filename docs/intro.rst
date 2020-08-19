==========
Why pyFBS?
==========
:mod:`pyFBS` is a Python package for Frequency Based Substructuring and Transfer Path Analysis. 
The package implements an object-oriented approach for dynamic substructuring. 
Current state-of-the-art methodologies in frequency based substructuring are available in pyFBS. 
Each method can be used as a standalone or interchangeably with others. 
Furthermore, basic and application examples are provided with the package together with real experimental and numerical data. 
The pyFBS has been designed to be used for scientific research in the field of dynamic substructuring. 
It is currently being used by a number of undergraduate students and postgraduate researchers. 

.. figure:: ./logo/pyFBS_logo_presenttion.gif
   :width: 800px

**********************
Dynamic Substructuring
**********************
In science, engineering and technology complex problems are often decomposed into smaller, simpler subsystems. 
Each subsystem can then be analyzed and evaluated separately. 
This approach can often reduce the complexity of the overall problem and provide invaluable insight into the optimization and troubleshooting of each individual component. 
The subsystems can also be assembled back together and with that the system can be analyzed as a whole.

Dynamic Substructuring (DS) is an engineering concept where dynamic systems are modeled and analyzed in terms of their components or so-called substructures. 
There are several ways of formulating the dynamics of substructures. One of them is with Frequency Response Functions (FRFs), which describe the response as the result of a unit harmonic force. 
The method is well suited for experimental approaches where FRFs are obtained from measurement of components. Such approaches were already investigated in the 70s [1]_  
and 80s [2]_ [3]_ [4]_  [5]_. 
Due to complicated formulations and difficulties in obtaining good measurements, the method was hardly applicable. 
Thanks to better measurement hardware and proper formulation of the problem,  Frequency Based Substructuring (FBS) has gained popularity in recent years [6]_ [7]_ [8]_.  
With this approach, it is also possible to build hybrid models in which experimentally characterized and numerically modelled parts are combined.

.. rubric:: References

.. [1] Albert L. Klosterman. A combined experimental and analytical procedure for improving automotive system dynamics. PhD thesis, University of Cincinnati, Department of Mechanical Engineering, 1971.
.. [2] David R. Martinez, Thomas G. Carrie, Dan L. Gregory, and A. Keith Miller. Combined Experimental/Analytical Modelling using component modes synthesis. In 25th Structures, Structural Dynamics and Materials Conference, 140–152. Palm Springs, CA, USA, 1984.
.. [3] John R. Crowley, Albert L. Klosterman, G. Thomas Rocklin, and Havard Vold. Direct structural modification using frequency response functions. Proceedings of IMAC II, February 1984.
.. [4] Bjorn Jetmundsen, Richard L. Bielawa, and William G. Flannelly. Generalized frequency domain substructure synthesis. Jnl. American Helicopter Society, 33(1):55–64, 1988.
.. [5] Antonio Paulo Vale Urgueira. Dynamic analysis of coupled structures using experimental data. PhD thesis, Imperial College, London, 1989.
.. [6] D.J. Rixen, T. Godeby, and E. Pagnacco. Dual assembly of substructures and the fbs method: application to the dynamic testing of a guitar. International Conference on Noise and Vibration Engineering, ISMA, September 18-20 2006.
.. [7] D. de Klerk, D. J. Rixen, and S. N. Voormeeren. General framework for dynamic substructuring: history, review and classification of techniques. AIAA Journal, 46(5):1169–1181, May 2008.
.. [8] Maarten V. van der Seijs, Dennis de Klerk, and Daniel J. Rixen. General framework for transfer path analysis: history, theory and classification of techniques. Mechanical Systems and Signal Processing, 68-69:217–244, February 2016.
