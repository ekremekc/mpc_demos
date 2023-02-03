import gmsh
import numpy as np
from dolfinx.fem import (Function, FunctionSpace,
                         locate_dofs_geometrical,
                         locate_dofs_topological,form)
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from dolfinx_mpc import  MultiPointConstraint, assemble_matrix
from dolfinx_mpc.utils import (create_point_to_point_constraint,
                               gmsh_model_to_mesh)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import ( Measure,  TestFunction,
                 TrialFunction, grad, inner)
from slepc4py import SLEPc

from helmholtz_x.mpc_utils import  EPS_get_spectrum, solve_GEP_shiftinvert, print0, gather_dof_coordinates

# Mesh parameters for creating a mesh consisting of two disjoint cubes
right_tag = 2
left_tag = 1


gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)

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
    # gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
    gmsh.model.mesh.generate(3)

    
    gmsh.model.mesh.optimize("Netgen")
    # gmsh.fltk.run()

mesh, ct,ft = gmsh_model_to_mesh(gmsh.model, cell_data=True, facet_data=True, gdim=3)
gmsh.clear()
gmsh.finalize()
MPI.COMM_WORLD.barrier()

with XDMFFile(mesh.comm, "test3D.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)

V = FunctionSpace(mesh, ("Lagrange", 1))
tdim = mesh.topology.dim
fdim = tdim - 1

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
dx = Measure("dx", domain=mesh, subdomain_data=ct)
ds = Measure("ds", domain=mesh, subdomain_data=ft)

facets_r = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1))
dofs_r = locate_dofs_topological(V, fdim, facets_r)
facets_l = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1.1))
dofs_l = locate_dofs_topological(V, fdim, facets_l)

# Given the local coordinates of the dofs, distribute them on all processors
nodes = [gather_dof_coordinates(V, dofs_r), gather_dof_coordinates(V, dofs_l)]
# print(nodes)
pairs = []
for left_node in nodes[0]:
    for right_node in nodes[1]:
        if np.isclose(left_node[1], right_node[1]):
            if np.isclose(left_node[2], right_node[2]):
                pairs.append([left_node, right_node])
                break

# print(pairs)
mpc = MultiPointConstraint(V)
for i, pair in enumerate(pairs):
    sl, ms, co, ow, off = create_point_to_point_constraint(
        V, pair[0], pair[1])
    mpc.add_constraint(V, sl, ms, co, ow, off)
mpc.finalize()


s = 343
a_form = -s**2 * inner(grad(u), grad(v))*dx

e_form =  s**2 * (inner(u, v)*ds(2)-inner(u, v)*ds(1)) # THIS MIGHT BE WRONG

mass_form = form(a_form+e_form)

A = assemble_matrix(mass_form, mpc)
A.assemble()

stiffnes_form = form(inner(u , v) * dx)

C = assemble_matrix(stiffnes_form, mpc)
C.assemble()

target_eig = (200*2*np.pi)**2

EPS = solve_GEP_shiftinvert(A, C, problem_type=SLEPc.EPS.ProblemType.GHEP,
                            solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                            nev=10, tol=1e-7, max_it=10,
                            target=target_eig, shift=1.5)
(eigval, eigvec_r, eigvec_i) = EPS_get_spectrum(EPS, mpc)

for i in range(len(eigval)):
    mpc.backsubstitution(eigvec_r[i].vector)
    mpc.backsubstitution(eigvec_i[i].vector)

print0(f"Computed eigenvalues:\n {np.around(eigval,decimals=2)}")
print0(f"Computed eigenfrequencies:\n {np.sqrt(eigval)/(2*np.pi)}")


# Save first eigenvector
with XDMFFile(MPI.COMM_WORLD, "results/eigenvector.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(eigvec_r[6])