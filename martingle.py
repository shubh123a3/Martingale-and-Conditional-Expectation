import numpy as np
import matplotlib.pyplot as plt

t=10
s=1
NoOfPaths=100
NoOfSteps=100

def martingaleA():
    W_t=np.random.normal(0.0,pow(t,0.5),[NoOfPaths,1])
    E_W_t=np.mean(W_t)
    print("mean value equals to: %.2f while the expected value is W(0) =%0.2f " %(E_W_t,0.0))
    # nexted martingle


def martingaleB():
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])

    # time stemp from [t0,s]

    dt1 = s / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        # normil;isation
        Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + pow(dt1, 0.5) * Z[:, i]
        # last column of W
    W_s = W[:, -1]
    # print("W S is this -",W_s)

    # for evwry path we create asub simulation until time t and caluculation
    # time step form [s,t]

    dt2 = (t - s) / float(NoOfSteps)
    W_t = np.zeros([NoOfPaths, NoOfSteps + 1]);
    # to store the result
    E_W_t = np.zeros([NoOfPaths])
    Error = []
    for i in range(0, NoOfSteps):

        # Sub-simulation from time "s" until "t"
        W_t[:, 0] = W_s[i];
        for j in range(0, NoOfSteps):
            Z[:, j] = (Z[:, j] - np.mean(Z[:, j])) / np.std(Z[:, j]);
            # path simulation, from "s" until "t"
            W_t[:, j + 1] = W_t[:, j] + pow(dt2, 0.5) * Z[:, j];
        E_W_t[i] = np.mean(W_t[:, -1])
        Error.append(E_W_t[i] - W_s[i])

        if i == 0:

            plt.figure(figsize=(10, 5))
            plt.plot(np.linspace(0, s, NoOfSteps + 1), W[0, :], label="W(t) from 0 to s")
            for j in range(0, NoOfPaths):
                plt.plot(np.linspace(s, t, NoOfSteps + 1), W_t[j, :], alpha=0.3)
            plt.title("Nested Brownian Motion Simulation")
            plt.xlabel("Time")
            plt.ylabel("W(t)")
            plt.legend()
            plt.grid()
            plt.show()
    print(Error)
    error = np.max(np.abs(E_W_t - W_s))
    print("The error is equal to: %.18f" % (error))
martingaleA()
martingaleB()
