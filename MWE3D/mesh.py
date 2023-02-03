import gmsh
from mpi4py import MPI
from dolfinx_mpc.utils import (create_point_to_point_constraint,
                               gmsh_model_to_mesh, rigid_motions_nullspace)
# Mesh parameters for creating a mesh consisting of two disjoint rectangles
right_tag = 2
left_tag = 1


gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    gmsh.clear()


    left_cube = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    right_cube = gmsh.model.occ.addBox(1.1, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    # Add physical tags for Surfaces
    gmsh.model.addPhysicalGroup(2, [2], tag=left_tag)
    gmsh.model.setPhysicalName(2, left_tag, "Downstream")
    gmsh.model.addPhysicalGroup(2, [7], tag=right_tag)
    gmsh.model.setPhysicalName(2, right_tag, "Upstream")

    gmsh.model.addPhysicalGroup(3, [right_cube], tag=right_tag)
    gmsh.model.setPhysicalName(3, right_tag, "Right cube")
    gmsh.model.addPhysicalGroup(3, [left_cube], tag=left_tag)
    gmsh.model.setPhysicalName(3, left_tag, "Left cube")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1)
    # Generate mesh
    gmsh.model.mesh.generate(3)

    # gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.fltk.run()

mesh, ct,ft = gmsh_model_to_mesh(gmsh.model, cell_data=True, facet_data=True, gdim=3)
gmsh.clear()
gmsh.finalize()
MPI.COMM_WORLD.barrier()