import numpy as np
import pandas as pd
from matplotlib import pyplot

# ----- read W,b -----
W = {}
b = {}
skip = 1

df = pd.read_csv('Wb.csv', skiprows=skip, nrows=1, header=None)
N_HL = int(df[0])
skip += 1

df = pd.read_csv('Wb.csv', skiprows=skip, nrows=1, header=None)
N_W = df.values
skip += 1

for iW in range(1, N_HL + 1 + 1):
    df = pd.read_csv('Wb.csv', skiprows=iW + skip, nrows=N_W[0, iW - 1], header=None)
    W[str(iW)] = df.values
    skip += N_W[0, iW - 1]

skip = iW + skip + 1

df = pd.read_csv('Wb.csv', skiprows=skip, nrows=1, header=None)
N_HL = int(df[0])
skip += 1

df = pd.read_csv('Wb.csv', skiprows=skip, nrows=1, header=None)
N_b = df.values
skip += 1

for ib in range(1, N_HL + 1 + 1):
    df = pd.read_csv('Wb.csv', skiprows=ib + skip, nrows=1, header=None)
    b[str(ib)] = df.values
    skip += 1


# ----- formula -----
def f(x):
    y = x
    for i in range(1, N_HL + 1):
        y = np.tanh(np.matmul(y, W[str(i)]) + b[str(i)])
    y = (np.matmul(y, W[str(N_HL + 1)]) + b[str(N_HL + 1)])
    return y


N = 50
pi = np.pi
x = np.zeros([N, 1])
for i in range(1, N + 1):
    x[i - 1, 0] = np.linspace(-pi, pi, N)[i - 1]
y1 = np.sin(x)**3+np.cos(x)**3
y2 = f(x)
pyplot.plot(x, y1, color='blue')
pyplot.scatter(x, y2, color='red', marker='.', s=10)
pyplot.show()
