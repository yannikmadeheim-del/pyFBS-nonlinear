!   Commands inserted into this file will be executed immediately after the ANSYS /POST1 command.

!   Active UNIT system in Workbench when this object was created:  Metric (m, kg, N, s, V, A)
!   NOTE:  Any data that requires units (such as mass) is assumed to be in the consistent solver unit system.
!                See Solving Units in the help system for more information.

!! Output mass and stiffness
/aux2
file,file,full
hbmat,stiff_HB,txt,,ascii,stiff,no,yes
hbmat,mass_HB,txt,,ascii,mass,no,yes

!! Output node

/HEADER,OFF,OFF,OFF,OFF,OFF
/OUTPUT,nodes.txt
NLIST,,,,COORD
/OUTPUT

/OUTPUT,units.txt
/STATUS,UNITS
/OUTPUT