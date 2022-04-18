==============================
Singular Vector Transformation
==============================
Singular Vector Transformation (SVT) consists in projecting the acquired data into subspaces composed by dominant singular vectors, which are extracted directly from the available FRF datasets. 
No geometrical and/or analytical model is required. 
If some basic requirements are met, the reduced orthonormal frequency dependent basis would be able to control and observe most of the rigid and flexible vibration modes of interest over a broad frequency range. 
The SVT can tackle challenging scenarios with flexible behaving interfaces and lightly damped systems.



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="SVT example">

.. only:: html

    .. figure:: ./../data/SVT.png 
       :target: 12_SVT.html

       SVT example

.. raw:: html

    </div>

.. toctree::
   :hidden:

   12_SVT





.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="FBS Decoupling with SVT">

.. only:: html

    .. figure:: ./../data/seven_three.png   
       :target: 13_SVT_decoupling.html

       FBS Decoupling with SVT

.. raw:: html

    </div>

.. toctree::
   :hidden:

   13_SVT_decoupling


|
|
|
|
|
|
|
|

Singular value decomposition
****************************

The singular value decomposition can be used to decompose and sort the dynamical information of the measured FRF dataset:

.. math::
    \mathbf{Y}(\omega)=\mathbf{U}(\omega)\mathbf{\Sigma}(\omega)\mathbf{V}(\omega)^\text{H}=\mathbf{U}(\omega)\textbf{diag}(\sigma_i(\omega))\mathbf{V}(\omega)^\text{H}

where :math:`\mathbf{U}(\omega)` and :math:`\mathbf{V}(\omega)` are the orthonormal frequency-dependent left and right singular vectors and :math:`\mathbf{\Sigma}(\omega)` 
is the diagonal matrix containing the non-negative singular values of the matrix.

The column vectors of :math:`\mathbf{U}(\omega)` and :math:`\mathbf{V}(\omega)` can be considered as approximate mode shapes and approximate modal participation 
factors at the frequency :math:`\omega`. The associated singular values are the scaling factors representing how dominant the contribution (**dynamically**) 
of a singular state is at that frequency.

.. note::
    Note how the singular 'motions' are frequency dependent and span an orthogonal space (mathematical orthogonality), 
    while the characteristic modal vectors are frequency independent and are :math:`\mathbf{K}`-orthogonal and :math:`\mathbf{M}`-orthogonal (physical orthogonality).

.. tip::
    **Idea!** Why don't we use the singular vectors as reduction bases to weaken the interface problem?

    A subset of dominant singular vectors in :math:`\mathbf{U}(\omega)` and :math:`\mathbf{V}(\omega)` could be used to construct meaningful reduction 
    spaces that naturally contain rigid as well as flexible local interface deformations.

Methodology
***********

An extended (direct) decoupling problem is considered, where the goal is finding the interface forces that suppress the influence of A on AB, thus isolating the uncoupled response of subsystem B.

.. figure:: ./../data/disassembly.svg
   :width: 400px
   :align: center

Let's assume collocation between the inputs between AB and A and outputs between AB and A. 
Let's also assume that inputs and outputs do not share the same location and are distributed all over the common area between AB and A (interface and internal) 
such that the interface deformation of interest is controlled and observed in a similar manner.

The proposed procedure is summarized [1]_:

1.  Place inputs and outputs in a collocated manner between AB and A. Distribute them homogeneously over the full area (interface and internal), 
    such that all the vibration modes of interest are controlled and observed through similar dynamic spaces.
2.  Perform a SVD on :math:`\mathbf{Y}^\text{A}`, such that :math:`\mathbf{Y}^\text{A}=\mathbf{U}^\text{A}\mathbf{\Sigma}^\text{A}(\mathbf{V}^\text{A})^\text{H}`.
3.  Select truncated basis of left- and right SVs :math:`\mathbf{U}_r^\text{A}`, :math:`\mathbf{V}_r^\text{A}` for the reduction 
    of measured displacements and forces. The subspaces' vectors are complex, orthonormal and frequency-dependent.
4.  Use the same reduction bases for :math:`\mathbf{Y}^\text{A}` and :math:`\mathbf{Y}^\text{AB}`. This is necessary for imposing collocated compatibility and equilibrium 
    between the substructures.
5.  Decouple the reduced components according to the LM-FBS algorithm.

.. note::
    The SVD can be applied analogously on :math:`\mathbf{Y}^\text{AB}`. For simplicity, the derivation will be performed with the singular bases of :math:`\mathbf{Y}^\text{A}`.

Mathematical derivation
***********************

Let's apply an SVD on the measured dataset :math:`\mathbf{Y}^\text{A}`:

.. math::
    \mathbf{Y}^\text{A}(\omega)=\mathbf{U}^\text{A}(\omega)\mathbf{\Sigma}^\text{A}(\omega)(\mathbf{V}^\text{A}(\omega))^\text{H}

The left and right singular subspaces are complex orthonormal frequency-dependent bases.
Proper reduced bases :math:`\mathbf{U}_r^\text{A}` and :math:`\mathbf{V}_r^\text{A}` are chosen per frequency line according to a user-defined criteria. 
The frequency dependence will be omitted for simplicity from here on.

Displacement reduction
======================

The same reduction basis :math:`\mathbf{U}_r^\text{A}` is chosen for A and AB to preserve a collocated compatibility in the reduced domain:

.. math::
    \begin{array}{lcc}
    \boldsymbol{u}^\text{A}=\mathbf{U}_r^\text{A}\boldsymbol{\zeta}^\text{A}+\boldsymbol{\mu}^\text{A} \\
    \boldsymbol{u}^\text{AB}=\mathbf{U}_r^\text{A}\boldsymbol{\zeta}^\text{AB}+\boldsymbol{\mu}^\text{AB}
    \end{array}

The residual :math:`\boldsymbol{\mu}^\text{A}` represent the low-value singular states not included in :math:`\mathbf{U}_r^\text{A}`. 
The residual :math:`\boldsymbol{\mu}^\text{AB}` indicates the motion of AB that could not be described by the reduced singular subspace of A.
The transformation from the measured to the reduced space is obtained as:

.. math::
    \boldsymbol{\zeta}=\mathbf{U}_r^+\boldsymbol{u}=\mathbf{U}_r^\text{H}\boldsymbol{u}

where

.. math::
    \boldsymbol{u}=\begin{bmatrix}
    \boldsymbol{u}^\text{AB}  \\ \boldsymbol{u}^\text{A}  
    \end{bmatrix}, \quad \boldsymbol{\zeta}=\begin{bmatrix}
    \boldsymbol{\zeta}^\text{AB}  \\ \boldsymbol{\zeta}^\text{A}  
    \end{bmatrix} \quad \text{and}\quad \mathbf{U}_r=\begin{bmatrix}
    \mathbf{U}_r^\text{A} & \mathbf{0} \\ \mathbf{0} & \mathbf{U}_r^\text{A} 
    \end{bmatrix}

The superscripts :math:`(\star)^+` and :math:`(\star)^\text{H}` denote the Moore-Penrose pseudo-inverse and Hermitian operators respectively.

.. note::
    Note that, due to the orthogonality of :math:`\mathbf{U}_r`, no mathematical inversion has to be computed. 

To check the quality of the transformation, the filtered and measured displacements can be compared:

.. math::
    \tilde{\boldsymbol{u}}=\mathbf{U}_r\mathbf{U}_r^\text{H}\boldsymbol{u}=\mathbf{F}_\text{u}\boldsymbol{u}.

Force reduction
===============

The same reduction basis :math:`\mathbf{V}_r^\text{A}` is chosen for A and AB to preserve a collocated equilibrium in the reduced domain:

.. math::
    \begin{array}{lcc}
    \boldsymbol{\eta}^\text{A}=(\mathbf{V}_r^\text{A})^\text{H}\boldsymbol{g}^\text{A}\\
    \boldsymbol{\eta}^\text{AB}=(\mathbf{V}_r^\text{A})^\text{H}\boldsymbol{g}^\text{AB}
    \end{array}

The transformation from the reduced to the measured space is obtained as:

.. math::
    \boldsymbol{g}=\begin{bmatrix}
    \boldsymbol{g}^\text{AB}  \\ \boldsymbol{g}^\text{A}  
    \end{bmatrix}, \quad \boldsymbol{\eta}=\begin{bmatrix}
    \boldsymbol{\eta}^\text{AB}  \\ \boldsymbol{\eta}^\text{A}  
    \end{bmatrix}\quad \text{and}\quad \mathbf{V}_r=\begin{bmatrix}
    \mathbf{V}_r^\text{A} & \mathbf{0} \\ \mathbf{0} & \mathbf{V}_r^\text{A} 
    \end{bmatrix}

The superscripts :math:`(\star)^+` and :math:`(\star)^\text{H}` denote the Moore-Penrose pseudo-inverse and Hermitian operators respectively.
The filtered forces can be compared to the measured ones:

.. math::
    \tilde{\boldsymbol{g}}=\mathbf{V}_r\mathbf{V}_r^\text{H}\boldsymbol{g}=\mathbf{F}_g\boldsymbol{g}

Finally, the compatibility and equilibrium conditions are imposed in the reduced space observed by :math:`\mathbf{U}_r` and controlled by :math:`\mathbf{V}_r` respectively:

.. math::
    \begin{cases} 
    \boldsymbol{u}=\mathbf{Y}^\text{AB|A}(\mathbf{f}-\mathbf{V}_r\mathbf{B}^\text{T}_{\eta}\boldsymbol{\lambda}_{\eta}) \\
    \mathbf{B}_{\zeta}\mathbf{U}_r^\text{H}\boldsymbol{u}=\mathbf{0}  
    \end{cases}

Thus, a weakening of the interface problem is obtained. The solution of the equations follows the LM-FBS formulation:

.. math::
    \boldsymbol{\zeta}=\left[\mathbf{I}-\mathbf{Y}^\text{AB|A}_{\zeta\eta}\mathbf{B}_{\eta}^\text{T}\left(\mathbf{B}_{\zeta}\mathbf{Y}^\text{AB|A}_{\zeta\eta}\mathbf{B}_{\eta}^\text{T}\right)^{-1}\mathbf{B}_{\zeta}\right]\mathbf{Y}^\text{AB|A}_{\zeta\eta}\boldsymbol{\eta}=\mathbf{Y}^\text{B}_{\zeta\eta}\boldsymbol{\eta}

where the reduced uncoupled admittance :math:`\mathbf{Y}^\text{AB|A}_{\zeta\eta}` can be written as:

.. math::
    \mathbf{Y}^\text{AB|A}_{\zeta\eta}=\mathbf{U}_r^\text{H}\mathbf{Y}^\text{AB|A}\mathbf{V}_r=\begin{bmatrix} \mathbf{Y}^\text{AB}_{\zeta\eta}  & \mathbf{0} \\ \mathbf{0} & -\mathbf{Y}^\text{A}_{\zeta\eta}
    \end{bmatrix}=\begin{bmatrix} \mathbf{Y}^\text{AB}_{\zeta\eta}  & \mathbf{0} \\ \mathbf{0} & -\mathbf{\Sigma}_r^\text{A}
    \end{bmatrix}

Filtering :math:`\mathbf{Y}^\text{AB}` by reduction and back-projection (see filter matrices :math:`\mathbf{F}_\text{u}` and :math:`\mathbf{F}_\text{f}`) using the singular subspaces of :math:`\mathbf{Y}^\text{A}` 
can be useful to determine how well the local dynamics of AB is described through the reduced dominant controllability and observability spaces of A.

Requirements
************

Let's remind ourselves of the main assumptions for a successful SVT:

- Inputs in A and in the A part of AB must be collocated and so must the output in A and in the A part of AB. This is necessary to since no geometrical reduction is applied afterwards. In experimental practice, this requirement should not be too challenging,

- Inputs and output spaces should contain the same dynamical information. The balance between controllability and observability spaces is needed to minimize the physical inconsistency of the reduced model. In experimental practice, an homogeneous distribution of inputs and outputs throughout the available common area (A and the A part of AB) should be ensured,

- The reduced orthonormal subspaces of A should be able to map the observed and controlled dynamics in AB.

Benefits
********

The SVT relies on a simple concept: the measured dynamics is projected into a subspace composed by dominant singular states extracted from the measurement itself.
What are the main advantages and features of the SVT?

- No need for geometrical/analytical information,
- Flexible behaviour potentially included (if properly observed/controlled),
- Frequency-dependent transformation,
- Efficient least square smoothing of random error and outliers,
- Better conditioning of the interface problem,
- Reduce sensitivity to measurement location/direction bias.

.. rubric:: References

.. [1] F.Trainotti, T.Bregar, S.W.B.Klaassen and D.J.Rixen. Experimental Decoupling of Substructures by Singular Vector Transformation. in: Mechanical System and Signal Processing ('Under Review'), 2021