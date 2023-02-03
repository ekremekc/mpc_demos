import gmsh
from mpi4py import MPI
from dolfinx_mpc.utils import (create_point_to_point_constraint,
                               gmsh_model_to_mesh, rigid_motions_nullspace)
# Mesh parameters for creating a mesh consisting of two disjoint rectangles
right_tag = 1
left_tag = 2


gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    gmsh.clear()

    left_rectangle = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    right_rectangle = gmsh.model.occ.addRectangle(1.1, 0, 0, 1, 1)

    gmsh.model.occ.synchronize()

    # Add physical tags for volumes
    gmsh.model.addPhysicalGroup(1, [8], tag=2)
    gmsh.model.setPhysicalName(1, right_tag, "Downstream")
    gmsh.model.addPhysicalGroup(1, [2], tag=1)
    gmsh.model.setPhysicalName(1, left_tag, "Upstream")

    gmsh.model.addPhysicalGroup(2, [right_rectangle], tag=right_tag)
    gmsh.model.setPhysicalName(2, right_tag, "Right square")
    gmsh.model.addPhysicalGroup(2, [left_rectangle], tag=left_tag)
    gmsh.model.setPhysicalName(2, left_tag, "Left square")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1)
    # Generate mesh
    gmsh.model.mesh.generate(2)

    # gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.fltk.run()

mesh, ct = gmsh_model_to_mesh(gmsh.model, cell_data=True, gdim=2)
gmsh.clear()
gmsh.finalize()
MPI.COMM_WORLD.barrier()