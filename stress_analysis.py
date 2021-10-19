from fenics import *
import numpy as np
from mshr import *
import dolfin as df


def stress_analysis(N, x_c, y_c, a, b, angle, Er=4.):

    '''
    Function 'stress_analysis' computes the stress tensor for the given input parameters

    :param N: mesh refinement parameter
    :param: x_c, y_c center of the inclusion
    :param angle: angle of tilt of the inclusion from x-axis (ccw) in degrees
    :param a, b: semi-minor and major axis of ellipse
    :param Er: ratio of Young's modulus of inclusion to that of the substrate

    :return S_22: 2D output array of vertical traction field where the index [0,0] refers to the xy-coordinate (0,10)
    '''

    # Obtaining clockwise rotated elliptic hole
    
    # Angle of rotation degree to radian
    theta = -(np.pi / 180) * angle  # in radians

    # Rotation matrix for c.c.w rotation
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Vertices rotation in c.c.w direction
    c = np.array([x_c, y_c])
    p1 = np.array([0, 0])
    p2 = np.array([10, 0])
    p3 = np.array([10, 10])
    p4 = np.array([0, 10])
    
    # Rotating the coordinates
    p1 = np.matmul(R, p1 - c) + c
    p2 = np.matmul(R, p2 - c) + c
    p3 = np.matmul(R, p3 - c) + c
    p4 = np.matmul(R, p4 - c) + c

    domain1 = Polygon([Point(p1), Point(p2), Point(p3), Point(p4)])
    domain = domain1

    domain.set_subdomain(2, Ellipse(Point(c), a, b))

    mesh = generate_mesh(domain, N)

    # Rotate back and align the mesh
    mesh.rotate(angle, 2, Point(c))

    celltag = MeshFunction("size_t", mesh, mesh.topology().dim())
    celltag.set_all(0)

    class Inclusion(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-4
            xo = x[0] - x_c
            yo = x[1] - y_c
            c = np.cos(theta)
            s = np.sin(theta)
            X = xo * c - yo * s
            Y = xo * s + yo * c
            return pow(X, 2) / (a * a) + pow(Y, 2) / (b * b) <= 1 + tol

    inclusion = Inclusion()
    inclusion.mark(celltag, 1)

    facettag = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facettag.set_all(0)

    # fig = plt.figure(figsize=(3,3),dpi=200)
    # pic = plot(mesh,linewidth=0.1)
    # plt.savefig('mesh_N={}'.format(N))
    
    # defining fixed boundary
    class Fixed(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-3
            return x[1] <= 0 + tol

    fixed = Fixed()
    fixed.mark(facettag, 1)

    # defining traction boundary
    class Traction(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-3
            return x[1] >= 10 - tol

    traction = Traction()
    traction.mark(facettag, 2)

    # Vector function space
    p_order = 2
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), p_order)
    V2 = FunctionSpace(mesh, P2)

    # Constants
    nu = Constant(0.3)
    E = Constant(1e6)
    nu_i = nu
    E_i = Er * E

    lamda = E * nu / ((1 - 2 * nu) * (1 + nu))
    mu = E / (2 * (1 + nu))
    lamda_i = E * nu_i / ((1 - 2 * nu_i) * (1 + nu_i))
    mu_i = E_i / (2 * (1 + nu_i))

    # Define forcing function
    fu = Expression(("0.0", "0.0"), element=V2.ufl_element())
    du = Expression(("0.0", "1e-5"), element=V2.ufl_element())

    # Define boundary condition
    constraint = Expression((("0.0", "0.0")), element=V2.ufl_element())
    bc1 = DirichletBC(V2, constraint, facettag, 1)
    bc2 = DirichletBC(V2, du, facettag, 2)
    bc = [bc1, bc2]

    # Define variational problem
    u = TrialFunction(V2)
    w = TestFunction(V2)

    # Constitutive equation
    I = Identity(2)
    sigma = 2 * mu * 0.5 * (grad(u) + grad(u).T) + lamda * div(u) * I
    sigma_i = 2 * mu_i * 0.5 * (grad(u) + grad(u).T) + lamda_i * div(u) * I

    dx = Measure("dx", domain=mesh, subdomain_data=celltag)
    ds = Measure("ds", domain=mesh, subdomain_data=facettag)

    # Write equation in weak form
    # integrate(grad(w):sigma)dx = integrate(w.fu)*dx
    a = inner(grad(w), sigma) * dx(0) + inner(grad(w), sigma_i) * dx(1)
    L = dot(w, fu) * dx  # *ds(2)

    # Compute solution
    u = Function(V2)
    dof = len(u.vector()) / 2
    solve(a == L, u, bc)

    # Computing stresses from displacement field
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), p_order)
    V1 = FunctionSpace(mesh, P1)

    P3 = TensorElement("Lagrange", mesh.ufl_cell(), p_order)
    V3 = FunctionSpace(mesh, P3)

    def stress_compute():
        Sigma = 2 * mu * 0.5 * (grad(u) + grad(u).T) + lamda * div(u) * I
        Sigma_i = 2 * mu_i * 0.5 * (grad(u) + grad(u).T) + lamda_i * div(u) * I

        stress_tensor = TrialFunction(V3)
        w = TestFunction(V3)

        a = inner(w, stress_tensor) * dx
        L = inner(w, Sigma) * dx(0) + inner(w, Sigma_i) * dx(1)

        stress_tensor = Function(V3)
        solve(a == L, stress_tensor)
        return stress_tensor

    stress_tensor = stress_compute()

    sigma_11 = project(stress_tensor[0, 0], V1)
    sigma_12 = project(stress_tensor[0, 1], V1)
    sigma_21 = project(stress_tensor[1, 0], V1)
    sigma_22 = project(stress_tensor[1, 1], V1)

    rmesh = refine(mesh)
    Vh1 = FunctionSpace(rmesh, "Lagrange", 1)

    sigma_11.set_allow_extrapolation(True)
    s_11 = interpolate(sigma_11, Vh1)
    sigma_12.set_allow_extrapolation(True)
    s_12 = interpolate(sigma_12, Vh1)
    sigma_21.set_allow_extrapolation(True)
    s_21 = interpolate(sigma_21, Vh1)
    sigma_22.set_allow_extrapolation(True)
    s_22 = interpolate(sigma_22, Vh1)

    s_Von_Mises = dolfin.sqrt(s_11 * s_11 - s_11 * s_22 + s_22 * s_22 + 3 * s_12 * s_12)
    s_Von_Mises = project(s_Von_Mises, Vh1)
    s_Von_Mises.set_allow_extrapolation(True)

    numdiv = 49
    S_22 = np.zeros((numdiv + 1, numdiv + 1))
    for i in range(S_22.shape[0]):
        for j in range(S_22.shape[1]):
            x_j = 10 / numdiv * j
            y_i = 10 - 10 / numdiv * i
            S_22[i, j] = sigma_22(Point(x_j, y_i))

    return S_22
