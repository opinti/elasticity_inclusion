import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from stress_analysis import stress_analysis


def compute(N, x_c, y_c, a, b, angle):

    S_22 = stress_analysis(N, x_c, y_c, a, b, angle)
    x = np.reshape(S_22, [-1, 1])

    return x


Params = np.load('Params.npy')
pivot_cols = np.load('pivot_cols.npy')
Params_HF = Params[:, pivot_cols]

N = 60
X_HF = np.array([])

for i in range(len(pivot_cols)):

    x_c = Params_HF[0, i]
    y_c = Params_HF[1, i]
    a = Params_HF[2, i]
    b = Params_HF[3, i]
    angle = Params_HF[4, i]

    x_hf = compute(N, x_c, y_c, a, b, angle)

    if len(X_HF) == 0:
        X_HF = x_hf
    else:
        X_HF = np.append(X_HF, x_hf, axis=1)

np.save('X_HF', X_HF)
np.save('Params_HF', Params_HF)
