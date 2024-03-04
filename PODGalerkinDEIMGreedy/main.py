from dolfin import *
from rbnics import *
import numpy as np
from time import time

@DEIM("online",basis_generation="Greedy")
@ExactParametrizedFunctions("offline")
class NonlinearElliptic(NonlinearEllipticProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        NonlinearEllipticProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.du = TrialFunction(V)
        self.u = self._solution
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # Store the forcing term expression
        self.f = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", element=self.V.ufl_element())
        # Customize nonlinear solver parameters
        self._nonlinear_solver_parameters.update({
            "linear_solver": "mumps",
            "maximum_iterations": 20,
            "report": True
        })

    # Return custom problem name
    def name(self):
        return "NonLinElliptic"

    # Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_derivatives
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = 1.
            return (theta_a0,)
        elif term == "c":
            theta_c0 = 1
            return (theta_c0,)
        elif term == "f":
            theta_f0 = 100.
            return (theta_f0,)
        elif term == "s":
            theta_s0 = 1.0
            return (theta_s0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    @assemble_operator_for_derivatives
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            du = self.du
            a0 = inner(grad(du), grad(v)) * dx
            return (a0,)
        elif term == "c":
            u = self.u
            mu = self.mu
            c0 = (exp(mu[0] * u) - 1) / mu[0] * v * dx
            return (c0,)
        elif term == "f":
            f = self.f
            f0 = f * v * dx
            return (f0,)
        elif term == "s":
            s0 = v * dx
            return (s0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1)]
            return (bc0,)
        elif term == "inner_product":
            du = self.du
            x0 = inner(grad(du), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# Customize the resulting reduced problem
@CustomizeReducedProblemFor(NonlinearEllipticProblem)
def CustomizeReducedNonlinearElliptic(ReducedNonlinearElliptic_Base):
    class ReducedNonlinearElliptic(ReducedNonlinearElliptic_Base):
        def __init__(self, truth_problem, **kwargs):
            ReducedNonlinearElliptic_Base.__init__(self, truth_problem, **kwargs)
            self._nonlinear_solver_parameters.update({
                "report": True,
                "line_search": "wolfe"
            })

    return ReducedNonlinearElliptic
    

mesh = Mesh("../data/square.xml")
subdomains = MeshFunction("size_t", mesh, "../data/square_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "../data/square_facet_region.xml")

V = FunctionSpace(mesh, "Lagrange", 1)

problem = NonlinearElliptic(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.01, 10.0),]
problem.set_mu_range(mu_range)

reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(20,DEIM=21)
reduction_method.set_tolerance(1e-8,DEIM=1e-4)

reduction_method.initialize_training_set(50, DEIM=60)
reduced_problem = reduction_method.offline()


#training_param=np.array([7.467415654021864,0.04778149657650545,2.174174499423764,4.103298951726431,2.349450432186314,7.492338244077934,3.8743042593266934,1.6674382889903223,5.739308761422573,6.595866899181118,1.0303248153791154,3.97062140080252,8.078846195415274,6.477471930419845,2.4630677369796947,8.005140562704984,7.304897982493075,0.3142776142136902,1.7685995484654005,0.944913238800784,7.912018808594201,5.356415606404236,2.3366182107271634,8.956636555063875,3.9196339016233783,7.927387694358098,2.1771928867969503,5.937139531031575,0.9732765993483242,5.189100709551363,5.3117019341871154,0.5445681813453519,7.01201567816358,9.591848438219634,0.10519074627505053,6.42442749941552,6.253412074489239,0.7761425113050774,1.309126639588019,8.615437586310408,2.187575262141215,2.823396437765919,2.1196768492828895,1.572269349208109,7.5408730023981265,3.271038338419009,9.346420807368867,1.0910291312750597,8.209205604988076,6.913249851239702])
#testing_param=np.array([6.79366258928424,5.144823592422261,8.694135066455203,6.576125844353742,5.756618540091697,7.7550787896821065,6.492306143258466,6.352070380033129,7.139973128429036,7.218450572141714,0.9372040871406994,9.355149219045234,0.5985708218569029,3.3409627779390347,4.1663276109435845,4.096156800311999,5.744317812419666,4.458910500413916,7.060214251827091,0.7082522195384542,1.9930149186384043,2.0237749711389674,2.918324329876795,8.75728998697439,4.8604466778271895,2.5481864981590134,1.9750880705864595,1.588122605189059,6.413614091159767,1.615111922560049,8.734972424236354,7.252026338182472,7.216684231252305,9.132223195596751,3.549875103086656,7.10031602127225,0.2831305349740587,4.6141245527227985,7.091622457769002,9.300535703558557,9.615705531213308,8.868083558235366,6.020304184916272,4.774818744938585,1.666019359686287,9.958646044756957,1.565136632127708,5.9905441910341946,4.960913885295052,0.1983536748429748])
training_param=[(7.467415654021864,), (0.04778149657650545,), (2.174174499423764,), (4.103298951726431,), (2.349450432186314,), (7.492338244077934,), (3.8743042593266934,), (1.6674382889903223,), (5.739308761422573,), (6.595866899181118,), (1.0303248153791154,), (3.97062140080252,), (8.078846195415274,), (6.477471930419845,), (2.4630677369796947,), (8.005140562704984,), (7.304897982493075,), (0.3142776142136902,), (1.7685995484654005,), (0.944913238800784,), (7.912018808594201,), (5.356415606404236,), (2.3366182107271634,), (8.956636555063875,), (3.9196339016233783,), (7.927387694358098,), (2.1771928867969503,), (5.937139531031575,), (0.9732765993483242,), (5.189100709551363,), (5.3117019341871154,), (0.5445681813453519,), (7.01201567816358,), (9.591848438219634,), (0.10519074627505053,), (6.42442749941552,), (6.253412074489239,), (0.7761425113050774,), (1.309126639588019,), (8.615437586310408,), (2.187575262141215,), (2.823396437765919,), (2.1196768492828895,), (1.572269349208109,), (7.5408730023981265,), (3.271038338419009,), (9.346420807368867,), (1.0910291312750597,), (8.209205604988076,), (6.913249851239702,)]
testing_param=[(6.79366258928424,), (5.144823592422261,), (8.694135066455203,), (6.576125844353742,), (5.756618540091697,), (7.7550787896821065,), (6.492306143258466,), (6.352070380033129,), (7.139973128429036,), (7.218450572141714,), (0.9372040871406994,), (9.355149219045234,), (0.5985708218569029,), (3.3409627779390347,), (4.1663276109435845,), (4.096156800311999,), (5.744317812419666,), (4.458910500413916,), (7.060214251827091,), (0.7082522195384542,), (1.9930149186384043,), (2.0237749711389674,), (2.918324329876795,), (8.75728998697439,), (4.8604466778271895,), (2.5481864981590134,), (1.9750880705864595,), (1.588122605189059,), (6.413614091159767,), (1.615111922560049,), (8.734972424236354,), (7.252026338182472,), (7.216684231252305,), (9.132223195596751,), (3.549875103086656,), (7.10031602127225,), (0.2831305349740587,), (4.6141245527227985,), (7.091622457769002,), (9.300535703558557,), (9.615705531213308,), (8.868083558235366,), (6.020304184916272,), (4.774818744938585,), (1.666019359686287,), (9.958646044756957,), (1.565136632127708,), (5.9905441910341946,), (4.960913885295052,), (0.1983536748429748,)]
start=time()
for i in range(50):
    reduced_problem.set_mu(training_param[i])
    reduced_problem.solve()
    reduced_problem.export_solution(filename="online_solution_train_{}".format(i))

for i in range(50):
    reduced_problem.set_mu(testing_param[i])
    reduced_problem.solve()
    reduced_problem.export_solution(filename="online_solution_test_{}".format(i))

end=time()
np.save('time.npy',end-start)