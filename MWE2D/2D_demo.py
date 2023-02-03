import gmsh
import numpy as np
from typing import List, Tuple
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
    gmsh.model.setPhysicalName(1, 2, "Downstream")
    gmsh.model.addPhysicalGroup(1, [2], tag=1)
    gmsh.model.setPhysicalName(1, 1, "Upstream")

    gmsh.model.addPhysicalGroup(2, [right_rectangle], tag=right_tag)
    gmsh.model.setPhysicalName(2, right_tag, "Right square")
    gmsh.model.addPhysicalGroup(2, [left_rectangle], tag=left_tag)
    gmsh.model.setPhysicalName(2, left_tag, "Left square")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
    # Generate mesh
    gmsh.model.mesh.generate(2)

    # gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.mesh.optimize("Netgen")
    # gmsh.fltk.run()

mesh, ct,ft = gmsh_model_to_mesh(gmsh.model, cell_data=True, facet_data=True, gdim=2)
gmsh.clear()
gmsh.finalize()
MPI.COMM_WORLD.barrier()

with XDMFFile(mesh.comm, "test.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)

V = FunctionSpace(mesh, ("Lagrange", 1))
tdim = mesh.topology.dim
fdim = tdim - 1

# DG0 = FunctionSpace(mesh, ("DG", 0))
# left_cells = ct.indices[ct.values == left_tag]
# right_cells = ct.indices[ct.values == right_tag]
# left_dofs = locate_dofs_topological(DG0, tdim, left_cells)
# right_dofs = locate_dofs_topological(DG0, tdim, right_cells)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
dx = Measure("dx", domain=mesh, subdomain_data=ct)
ds = Measure("ds", domain=mesh, subdomain_data=ft)


def gather_dof_coordinates(V, dofs):
    """
    Distributes the dof coordinates of this subset of dofs to all processors
    """
    x = V.tabulate_dof_coordinates()
    local_dofs = dofs[dofs < V.dofmap.index_map.size_local * V.dofmap.index_map_bs]
    coords = x[local_dofs]
    num_nodes = len(coords)
    glob_num_nodes = MPI.COMM_WORLD.allreduce(num_nodes, op=MPI.SUM)
    recvbuf = None
    if MPI.COMM_WORLD.rank == 0:
        recvbuf = np.zeros(3 * glob_num_nodes, dtype=np.float64)
    sendbuf = coords.reshape(-1)
    sendcounts = np.array(MPI.COMM_WORLD.gather(len(sendbuf), 0))
    MPI.COMM_WORLD.Gatherv(sendbuf, (recvbuf, sendcounts), root=0)
    glob_coords = MPI.COMM_WORLD.bcast(recvbuf, root=0).reshape((-1, 3))
    return glob_coords



facets_r = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1))
dofs_r = locate_dofs_topological(V, fdim, facets_r)
facets_l = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1.1))
dofs_l = locate_dofs_topological(V, fdim, facets_l)

# Given the local coordinates of the dofs, distribute them on all processors
nodes = [gather_dof_coordinates(V, dofs_r), gather_dof_coordinates(V, dofs_l)]
pairs = []
for x in nodes[0]:
    for y in nodes[1]:
        if np.isclose(x[1], y[1]):
            pairs.append([x, y])
            break

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

def print0(string: str):
    """Print on rank 0 only"""
    if MPI.COMM_WORLD.rank == 0:
        print(string)

def solve_GEP_shiftinvert(A: PETSc.Mat, B: PETSc.Mat,
                          problem_type: SLEPc.EPS.ProblemType = SLEPc.EPS.ProblemType.GNHEP,
                          solver: SLEPc.EPS.Type = SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev: int = 10, tol: float = 1e-7, max_it: int = 10,
                          target: float = 0.0, shift: float = 0.0) -> SLEPc.EPS:


    # Build an Eigenvalue Problem Solver object
    EPS = SLEPc.EPS()
    EPS.create(comm=MPI.COMM_WORLD)
    A=-A
    EPS.setOperators(A, B)
    EPS.setProblemType(problem_type)
    # set the number of eigenvalues requested
    EPS.setDimensions(nev=nev)
    # Set solver
    EPS.setType(solver)
    # set eigenvalues of interest
    EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    EPS.setTarget(target)  # sorting
    # set tolerance and max iterations
    EPS.setTolerances(tol=tol, max_it=max_it)
    # Set up shift-and-invert
    # Only work if 'whichEigenpairs' is 'TARGET_XX'
    ST = EPS.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    ST.setShift(shift)
    EPS.setST(ST)
    # set monitor
    it_skip = 1
    EPS.setMonitor(lambda eps, it, nconv, eig, err:
                   monitor_EPS_short(eps, it, nconv, eig, err, it_skip))
    # parse command line options
    EPS.setFromOptions()
    # Display all options (including those of ST object)
    # EPS.view()
    EPS.solve()
    EPS_print_results(EPS)
    return EPS

def EPS_get_spectrum(EPS: SLEPc.EPS,
                     mpc: MultiPointConstraint) -> Tuple[List[complex], List[PETSc.Vec], List[PETSc.Vec]]:
    """ Retrieve eigenvalues and eigenfunctions from SLEPc EPS object.
        Parameters
        ----------
        EPS
           The SLEPc solver
        mpc
           The multipoint constraint

        Returns
        -------
            Tuple consisting of: List of complex converted eigenvalues,
             lists of converted eigenvectors (real part) and (imaginary part)
        """
    # Get results in lists
    eigval = [EPS.getEigenvalue(i) for i in range(EPS.getConverged())]
    eigvec_r = list()
    eigvec_i = list()
    V = mpc.function_space
    for i in range(EPS.getConverged()):
        vr = Function(V)
        vi = Function(V)

        EPS.getEigenvector(i, vr.vector, vi.vector)
        eigvec_r.append(vr)
        eigvec_i.append(vi)    # Sort by increasing real parts
    idx = np.argsort(np.real(np.array(eigval)), axis=0)
    eigval = [eigval[i] for i in idx]
    eigvec_r = [eigvec_r[i] for i in idx]
    eigvec_i = [eigvec_i[i] for i in idx]
    return (eigval, eigvec_r, eigvec_i)

def monitor_EPS_short(EPS: SLEPc.EPS, it: int, nconv: int, eig: list, err: list, it_skip: int):
    """
    Concise monitor for EPS.solve().

    Parameters
    ----------
    eps
        Eigenvalue Problem Solver class.
    it
       Current iteration number.
    nconv
       Number of converged eigenvalue.
    eig
       Eigenvalues
    err
       Computed errors.
    it_skip
        Iteration skip.

    """
    if (it == 1):
        print0('******************************')
        print0('***  SLEPc Iterations...   ***')
        print0('******************************')
        print0("Iter. | Conv. | Max. error")
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")
    elif not it % it_skip:
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")


def EPS_print_results(EPS: SLEPc.EPS):
    """ Print summary of solution results. """
    print0("\n******************************")
    print0("*** SLEPc Solution Results ***")
    print0("******************************")
    its = EPS.getIterationNumber()
    print0(f"Iteration number: {its}")
    nconv = EPS.getConverged()
    print0(f"Converged eigenpairs: {nconv}")

    if nconv > 0:
        # Create the results vectors
        vr, vi = EPS.getOperators()[0].createVecs()
        print0("\nConverged eigval.  Error ")
        print0("----------------- -------")
        for i in range(nconv):
            k = EPS.getEigenpair(i, vr, vi)
            error = EPS.computeError(i)
            if k.imag != 0.0:
                print0(f" {k.real:2.2e} + {k.imag:2.2e}j {error:1.1e}")
            else:
                pad = " " * 11
                print0(f" {k.real:2.2e} {pad} {error:1.1e}")

target_eig = (170*2*np.pi)**2
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
    xdmf.write_function(eigvec_r[12])