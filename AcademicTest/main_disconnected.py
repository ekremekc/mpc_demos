from helmholtz_x.eigensolvers_x import pep_solver,eps_solver
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, XDMFReader
from helmholtz_x.solver_utils import start_time, execution_time
from helmholtz_x.parameters_utils import c_step
start = start_time()
import numpy as np
import params
from passive_flame_x import PassiveFlame

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

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

target = 380 * 2 * np.pi 
# E = pep_solver(matrices.A, matrices.B, matrices.C, target, nev=10, print_results= True)
E = eps_solver(matrices.A, matrices.C, target**2, 10, print_results= True)

omega, uh = normalize_eigenvector(mesh, E, 0, absolute=True, degree=degree, which='right')
xdmf_writer("Results/P", mesh, uh)

# omega2, uh2 = normalize_eigenvector(mesh, E, 2, absolute=True, degree=degree, which='right')
# xdmf_writer("Results/Disconnected", mesh, uh2)

execution_time(start)