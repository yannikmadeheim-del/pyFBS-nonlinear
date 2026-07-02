# Ansys rod models for the bars-with-gap AFT example

Each rod is a 3D slender solid (SOLID186) built in Ansys Mechanical, modal-solved with
the **shared-memory (SMP)** solver and **"Delete Unneeded Files = No"** so the assembled
matrices are kept. Two files per rod, read by `pyfbs.mck.Model.from_ansys`:

| file | what it is |
|---|---|
| `rodI.full`  | substructure I mass + stiffness matrices (M, K) |
| `rodI.rst`   | substructure I mesh + node components (incl. `interface_I`) |
| `rodII.full` | substructure II M, K |
| `rodII.rst`  | substructure II mesh + node components (incl. `interface_II`) |

Provenance (so this is reproducible):
- Geometry from SolidWorks → STEP (AP214), imported into standalone Mechanical.
- Material: Structural Steel (default). Rod I clamped-free (far end fixed); Rod II free-free.
- Interface = the bolt-hole faces of the mount, captured as a **named selection / Komponente**
  (`interface_I` / `interface_II`) with *Send to Solver = Yes*; the virtual point / RBE2 pilot
  is the hole center on the rod axis.
- Source solver files are Mechanical's `file.full` / `file.rst`; copy + rename them here.

The interface reduction (VPT for the HBM solver, RBE2 for the time-integration reference)
is done in Python — these Ansys files carry the **physical** DoFs only.
