from dolfinx.fem import Function, FunctionSpace, dirichletbc, form, locate_dofs_topological, Constant
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem.assemble import assemble_scalar
from helmholtz_x.solver_utils import info
from helmholtz_x.parameters_utils import sound_speed_variable_gamma, gamma_function
from ufl import Coefficient, Measure, TestFunction, TrialFunction, grad, inner
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from scipy.special import j1,y1,jv,yv

class PassiveFlame:

    def __init__(self, mesh, subdomains, facet_tags, boundary_conditions,
                 parameter, holes=False,degree=1):

        self.mesh = mesh
        self.facet_tags = facet_tags
        self.subdomains = subdomains
        self.fdim = self.mesh.topology.dim - 1
        self.boundary_conditions = boundary_conditions
        self.parameter = parameter
        self.degree = degree

        self.dimension = self.mesh.topology.dim
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.subdomains)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags)

        self.V = FunctionSpace(self.mesh, ("Lagrange", degree))

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        
        self.AreaConstant = Constant(self.mesh, PETSc.ScalarType(1))

        self.bcs = []
        self.integrals_R = []
        self.a_form = None
        self.b_form = None
        self.c_form = None

        # if self.parameter.name =="temperature":
        #     self.c = sound_speed_variable_gamma(self.mesh, parameter)
        #     self.gamma = gamma_function(self.mesh, self.parameter)
        #     info("/\ Temperature function is used for passive flame matrices.")
        # else:
        self.c = parameter
        info("\/ Speed of sound function is used for passive flame matrices.")

        for i in boundary_conditions:
            if 'Dirichlet' in boundary_conditions[i]:
                u_bc = Function(self.V)
                facets = np.array(self.facet_tag.indices[self.facet_tag.values == i])
                dofs = locate_dofs_topological(self.V, self.fdim, facets)
                bc = dirichletbc(u_bc, dofs)
                self.bcs.append(bc)

            if 'Robin' in boundary_conditions[i]:
                R = boundary_conditions[i]['Robin']
                Z = (1+R)/(1-R)
                integrals_Impedance = 1j * self.c / Z * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integrals_Impedance)

            if 'MP' in boundary_conditions[i]:
                rho = boundary_conditions[i]['MP']
                # Z = rho*(inner(self.u, self.v)*self.ds(i)-inner(self.u, self.v)*self.ds(i+1))
                integrals_MP1 = 1j * self.c**2 / rho*inner(self.u, self.v)*self.ds(i)
                integrals_MP2 = -1j * self.c**2 / rho*inner(self.u, self.v)*self.ds(i+1)

                self.integrals_R.append(integrals_MP1)
                self.integrals_R.append(integrals_MP2)

            if 'ChokedInlet' in boundary_conditions[i]:
                # https://www.oscilos.com/download/OSCILOS_Long_Tech_report.pdf
                A_inlet = MPI.COMM_WORLD.allreduce(assemble_scalar(form(self.AreaConstant * self.ds(i))), op=MPI.SUM)
                gamma_inlet_form = form(self.gamma/A_inlet* self.ds(i))
                gamma_inlet = MPI.COMM_WORLD.allreduce(assemble_scalar(gamma_inlet_form), op=MPI.SUM)

                Mach = boundary_conditions[i]['ChokedInlet']
                R = (1-gamma_inlet*Mach/(1+(gamma_inlet-1)*Mach**2))/(1+gamma_inlet*Mach/(1+(gamma_inlet-1)*Mach**2))
                Z = (1+R)/(1-R)
                integral_C_i = 1j * self.c / Z * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integral_C_i)

            if 'ChokedOutlet' in boundary_conditions[i]:
                # https://www.oscilos.com/download/OSCILOS_Long_Tech_report.pdf
                A_outlet = MPI.COMM_WORLD.allreduce(assemble_scalar(form(self.AreaConstant * self.ds(i))), op=MPI.SUM)
                gamma_outlet_form = form(self.gamma/A_outlet* self.ds(i))
                gamma_outlet = MPI.COMM_WORLD.allreduce(assemble_scalar(gamma_outlet_form), op=MPI.SUM)

                Mach = boundary_conditions[i]['ChokedOutlet']
                R = (1-0.5*(gamma_outlet-1)*Mach)/(1+0.5*(gamma_outlet-1)*Mach)
                Z = (1+R)/(1-R)
                integral_C_o = 1j * self.c / Z * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integral_C_o)

        if holes:
            
            self.a_form = form(-self.c**2 * inner(grad(self.u), grad(self.v))*self.dx - self.c**2 * inner(grad(self.u), grad(self.v))*self.dx(holes['tag']))

            if self.integrals_R: # B matrix with Robin BC and cooling holes
                self.b_form = form(sum(self.integrals_R)+self.c**2 * 1j*holes['L']/holes['U']*inner(grad(self.u),grad(self.v))*self.dx(holes['tag']))
                info("- Cooling holes and Robin boundaries are modelled.")
            else:# B matrix with cooling holes
                self.b_form = form(+self.c**2 * 1j*holes['L']/holes['U']*inner(grad(self.u),grad(self.v))*self.dx(holes['tag']))
                info("- Only cooling holes are modelled, Robin boundary condition is not exist.")

        else:
            omega = (247.5)*2*np.pi # U=120
            # omega = (382.9)*2*np.pi # U=120
            
            a = 3*1E-3
            d = 35*1E-3
            U = 120
            St = omega*a/U
            print("St number: ", St)
            num =       (np.pi/2)*jv(1,St)*np.exp(-St)-1j*yv(1,St)*np.sinh(St)
            denom = St*((np.pi/2)*jv(1,St)*np.exp(-St)+1j*yv(1,St)*np.cosh(St)) 
            expression = 1+num/denom
            K_r = 2*a*expression
            print("num: ", num)
            print("denom: ", denom)
            print("K_r is: ", K_r)
            # self.a_form = form(-self.c**2 * inner(grad(self.u), grad(self.v))*self.dx \
            #     +2*self.c**2 * K_r/d**2 * (inner(self.u, self.v)*self.ds(9)-inner(self.u, self.v)*self.ds(3))\
            #     )
            self.a_form = -self.c**2 * inner(grad(self.u), grad(self.v))*self.dx
            # self.e_form = -self.c**2 * K_r/d**2 * (inner(self.u, self.v)*self.ds(9)-inner(self.u, self.v)*self.ds(3))\
            #               +self.c**2 * K_r/d**2 * (inner(self.u, self.v)*self.ds(3)-inner(self.u, self.v)*self.ds(9))
            # self.e_form =  self.c**2 * K_r/d**2 * (inner(self.u, self.v)*self.ds(9)-inner(self.u, self.v)*self.ds(3)) # CORRECT
            self.e_form =  -self.c**2 * K_r/d**2 * (inner(self.u, self.v)*self.ds(3)-inner(self.u, self.v)*self.ds(9)) # CORRECT

            self.a_form = form(self.a_form+self.e_form)
            # self.a_form = form(self.a_form)
            if self.integrals_R:
                self.b_form = form(sum(self.integrals_R))
              
        self.c_form = form(inner(self.u , self.v) * self.dx)
        
        self._A = None
        self._B = None
        self._B_adj = None
        self._C = None

        info("- Passive matrices are assembling..")

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def B_adj(self):
        return self._B_adj

    @property
    def C(self):
        return self._C

    def assemble_A(self):
        
        A = assemble_matrix(self.a_form, bcs=self.bcs)
        A.assemble()
        info("- Matrix A is assembled.")
        self._A = A

    def assemble_B(self):

        if self.b_form:
            B = assemble_matrix(self.b_form)
            
        else:
            N = self.V.dofmap.index_map.size_global
            n = self.V.dofmap.index_map.size_local
            B = PETSc.Mat().create()
            B.setSizes([(n, N), (n, N)])
            B.setFromOptions()
            B.setUp()
            info("! Note: It can be faster to use EPS solver.")
        
        B.assemble()
            
        B_adj = B.copy()
        B_adj.transpose()
        B_adj.conjugate()

        info("- Matrix B is assembled.")

        self._B = B
        self._B_adj = B_adj

    def assemble_C(self):

        C = assemble_matrix(self.c_form, self.bcs)
        C.assemble()
        info("- Matrix C is assembled.\n")
        self._C = C
