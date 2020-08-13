==========
Why pyFBS?
==========
pyFBS is a Python package for Frequency Based Substructuring. 
The package implements an object-oriented approach for dynamic substructuring. 
Current state-of-the-art methodologies in frequency based substructuring are available in pyFBS. 
Each method can be used as a standalone or interchangeably with others. 
Furthermore, basic and application examples are provided with the package together with real experimental and numerical data. 
The pyFBS has been designed to be used for scientific research in the field of dynamic substructuring. 
It is currently being used by a number of undergraduate students and postgraduate researchers. 

.. figure:: ../data/pyFBS_logo_presenttion.gif
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
The method is well suited for experimental approaches where FRFs are obtained from measurement of components. Such approaches were already investigated in the 70s :cite:`intro-KLOSTERMAN_1971_PHD`  
and 80s (e.g. :cite:`intro-MARTINEZ_1984_COMBINEDEXPANALYTICAL,intro-KLOSTERMAN_1984_SMURF,intro-JETMUNDSEN_1988_FBS,intro-URGUEIRA_1989_DYNAMIC`). 
Due to complicated formulations and difficulties in obtaining good measurements, the method was hardly applicable. 
Thanks to better measurement hardware and proper formulation of the problem,  Frequency Based Substructuring (FBS) has gained popularity in recent years :cite:`intro-deKlerk2008,intro-vanderSeijs2016,intro-RIXEN_2006_GUITAR`.  
With this approach, it is also possible to build hybrid models in which experimentally characterized and numerically modelled parts are combined.


.. rubric:: References

.. bibliography:: ..\joss\paper.bib
   :style: unsrt
   :filter: docname in docnames
   :keyprefix: intro-
