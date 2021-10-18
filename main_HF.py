import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from stress_analysis import stress_analysis


def compute(N, E, a, b, angle):

    Fxy, dof, s_Von_Mises = stress_analysis(N, E, angle, a, b)
    x = np.reshape(Fxy, [-1, 1])

    return x


Params = np.load('Params.npy')
pivot_cols = np.array([24449, 17, 24439, 99, 368, 22967, 23609, 654, 152, 23622, 200, 24202, 23211, 24297, 22665, 24123, 23948, 23578, 24025, 24398])
Params_HF = Params[:, pivot_cols]

N = 60
X_HF = np.array([])

for i in range(len(pivot_cols)):

    E = Params_HF[0, i]
    a = Params_HF[1, i]
    b = Params_HF[2, i]
    angle = Params_HF[3, i]

    x_hf = compute(N, E, a, b, angle)

    if len(X_HF) == 0:
        X_HF = x_hf
    else:
        X_HF = np.append(X_HF, x_hf, axis=1)

np.save('X_HF', X_HF)
np.save('Params_HF', Params_HF)
