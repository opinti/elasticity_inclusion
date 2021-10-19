import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from stress_analysis import stress_analysis

def compute(N, x_c, y_c, a, b, angle):

    params = np.reshape(np.array([x_c, y_c, a, b, angle]), [5, 1])

    S_22 = stress_analysis(N, x_c, y_c, a, b, angle)
    x_lf = np.reshape(S_22, [-1, 1])

    return x_lf, params

N = 15
X_c = np.array([7])
Y_c = np.array([7])
A = np.linspace(1, 2, 10)
B = np.linspace(1, 2, 10)
angles = np.linspace(0, 180, 18)

X_LF = np.array([])
Params = np.array([])

for x_c in X_c:
    for y_c in Y_c:
        for a in A:
            for b in B:
    
                if np.abs(a - b) < 1e-3:
                    x_lf, params = compute(N, x_c, y_c, a, b, angle=0.)
                    if len(X_LF) == 0:
                        X_LF = x_lf
                        Params = params
                    else:
                        X_LF = np.append(X_LF, x_lf, axis=1)
                        Params = np.append(Params, params, axis=1)
                        
                else:
                    for angle in angles:
                        x_lf, params = compute(N, x_c, y_c, a, b, angle)
                        if len(X_LF) == 0:
                            X_LF = x_lf
                            Params = params
                        else:
                            X_LF = np.append(X_LF, x_lf, axis=1)
                            Params = np.append(Params, params, axis=1)

np.save('X_LF', X_LF)
np.save('Params', Params)
