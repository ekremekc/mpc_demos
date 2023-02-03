# from helmholtz_x.eigensolvers_x import pep_solver,eps_solver
# from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, XDMFReader, cart2cyl

# from helmholtz_x.parameters_utils import c_step

# from passive_flame_x import PassiveFlame

from dolfinx.fem import (Function, FunctionSpace,
                         locate_dofs_geometrical,
                         locate_dofs_topological,form)
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from dolfinx_mpc import  MultiPointConstraint, assemble_matrix
from dolfinx_mpc.utils import (create_point_to_point_constraint)
from ufl import ( Measure,  TestFunction,
                 TrialFunction, grad, inner)
from slepc4py import SLEPc
from mpi4py import MPI
from helmholtz_x.solver_utils import start_time, execution_time
from helmholtz_x.mpc_utils import  EPS_get_spectrum, solve_GEP_shiftinvert, print0, gather_dof_coordinates
start = start_time()
import numpy as np
import params

# approximation space polynomial degree
degree = 1

# number of elements in each direction of mesh
MP = XDMFReader("MeshDir/Disconnected")
mesh, subdomains, facet_tags = MP.getAll()
MP.getInfo()

# Define the boundary conditions

boundary_conditions = {}

# Define Speed of sound

c = params.c_u

outer_tag = 9
inner_tag = 3

V = FunctionSpace(mesh, ("Lagrange", 1))
tdim = mesh.topology.dim
fdim = tdim - 1

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)

r_inner = 0.199
r_outer = 0.201

dofs_inner = facet_tags.indices[facet_tags.values == inner_tag]
dofs_outer = facet_tags.indices[facet_tags.values == outer_tag]

dofs_inner = locate_dofs_topological(V, fdim, dofs_inner)
dofs_outer = locate_dofs_topological(V, fdim, dofs_outer)
# print(dofs_outer)
# Given the local coordinates of the dofs, distribute them on all processors
nodes = [gather_dof_coordinates(V, dofs_outer), gather_dof_coordinates(V, dofs_inner)]
# print(nodes[0])

nodes_cyc_outer = np.zeros((len(dofs_outer),3))
for i,node in enumerate(nodes[0]):
    nodes_cyc_outer[i] = cart2cyl(node[0],node[1],node[2])

nodes_cyc_inner = np.zeros((len(dofs_inner),3))
for j,node in enumerate(nodes[1]):
    nodes_cyc_inner[j] = cart2cyl(node[0],node[1],node[2])
# print(nodes_cyc_inner)

nodes_cyc = [nodes_cyc_outer,nodes_cyc_inner]
# print(nodes)
print("here")
pairs = []
for inner_node in nodes_cyc[0]:
    for outer_node in nodes_cyc[1]:
        if np.isclose(inner_node[1], outer_node[1]):
            if np.isclose(inner_node[2], outer_node[2]):
                pairs.append([inner_node, outer_node])
                break
print("here finish")
# print(pairs)
mpc = MultiPointConstraint(V)
# THIS LOOP IS TAKING SO LONG
for i, pair in enumerate(pairs):
    sl, ms, co, ow, off = create_point_to_point_constraint(
        V, pair[0], pair[1])
    mpc.add_constraint(V, sl, ms, co, ow, off)
mpc.finalize()

print("here finish2")

s = 347
a_form = -s**2 * inner(grad(u), grad(v))*dx

e_form =  s**2 * (inner(u, v)*ds(2)-inner(u, v)*ds(1)) # THIS MIGHT BE WRONG

mass_form = form(a_form)

A = assemble_matrix(mass_form, mpc)
A.assemble()

stiffnes_form = form(inner(u , v) * dx)

C = assemble_matrix(stiffnes_form, mpc)
C.assemble()

target_eig = (400.5*2*np.pi)**2


EPS = solve_GEP_shiftinvert(A, C, problem_type=SLEPc.EPS.ProblemType.GHEP,
                            solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                            nev=2, tol=1e-7, max_it=10,
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
    xdmf.write_function(eigvec_r[1])

execution_time(start)