# Function File for Computational and Mathematical Modeling of Infectious Diseases 2023
# Created by Casey Middleton

import numpy as np

def SIR_solver(S0,I0,R0,beta,gamma,tvals):
    # uses forward Euler's method to solve the SIR model
    # assumes S0, I0, and R0 are integer values
    # returns S,I,R vectors for t in tvals
    dt = tvals[1] - tvals[0]
    N = S0 + I0 + R0
    S = np.zeros(len(tvals)); I = np.zeros(len(tvals)); R = np.zeros(len(tvals));
    for ii,t in enumerate(tvals):
        # base case - set initial conditions
        if ii == 0:
            S[0] = S0; I[0] = I0; R[0] = R0;
            continue
        # otherwise, fill in vector using forward euler
        S[ii] = S[ii-1] + (-beta*S[ii-1]*I[ii-1]/N)*dt
        I[ii] = I[ii-1] + (beta*S[ii-1]*I[ii-1]/N - gamma*I[ii-1])*dt
        R[ii] = R[ii-1] + (gamma*I[ii-1])*dt
    return S,I,R
