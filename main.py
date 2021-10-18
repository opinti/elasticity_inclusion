import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from stress_analysis import stress_analysis

# N = 60
# Er = 4
# angle = 70
# a = 3
# b = 2
#
# Fxy, dof, s_Von_Mises = stress_analysis(N, Er, angle, a, b)
#
# # Displaying Output
# plt.figure()
# Fplot = Fxy
# plt.imshow(Fplot, interpolation='bilinear', extent=[0, 10, 0, 10])
# plt.colorbar()
# plt.show()


def compute(N, E, a, b, angle):

    params = np.reshape(np.array([E, a, b, angle]), [4, 1])

    Fxy, dof, s_Von_Mises = stress_analysis(N, E, angle, a, b)
    x_lf = np.reshape(Fxy, [-1, 1])

    return x_lf, params

N = 15
youngs = np.linspace(1, 6, 15)
angles = np.linspace(0, 180, 18)
A = np.linspace(1, 2, 10)
B = np.linspace(1, 2, 10)

X_LF = np.array([])
Params = np.array([])

for E in youngs:
    for a in A:
        for b in B:

            if np.abs(a - b) < 1e-3:
                x_lf, params = compute(N, E, a, b, angle=0.)
                if len(X_LF) == 0:
                    X_LF = x_lf
                    Params = params
                else:
                    X_LF = np.append(X_LF, x_lf, axis=1)
                    Params = np.append(Params, params, axis=1)
                # X_LF.append(x_lf)
                # Params.append(params)

            else:
                for angle in angles:
                    x_lf, params = compute(N, E, a, b, angle)
                    if len(X_LF) == 0:
                        X_LF = x_lf
                        Params = params
                    else:
                        X_LF = np.append(X_LF, x_lf, axis=1)
                        Params = np.append(Params, params, axis=1)

np.save('X_LF', X_LF)
np.save('Params', Params)
