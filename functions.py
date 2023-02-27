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

def SIS_solver(S0,I0,beta,gamma,tvals):
    # uses forward Euler's method to solve the SIS normalized model
    # assumes S0, I0 are values in [0,1]
    # returns S,I vectors for t in tvals
    dt = tvals[1] - tvals[0]
    S = np.zeros(len(tvals)); I = np.zeros(len(tvals));
    for ii,t in enumerate(tvals):
        # base case - set initial conditions
        if ii == 0:
            S[0] = S0; I[0] = I0;
            continue
        # otherwise, fill in vector using forward euler
        S[ii] = S[ii-1] + (-beta*S[ii-1]*I[ii-1] + gamma*I[ii-1])*dt
        I[ii] = I[ii-1] + (beta*S[ii-1]*I[ii-1] - gamma*I[ii-1])*dt
    return S,I

def calc_error(euler,analytical,tvals):
    '''
    Computes the error over delta t using
        | Euler(t) - Analytical(t) |
    '''
    dt = tvals[1] - tvals[0]
    errors = np.abs(euler - analytical)
    return np.max(errors)

def SIR_AoN_vax(S0,I0,R0,Vnull0,Vall0,beta,gamma,tvals):
    # All or Nothing Vaccine Model
    # Assumes all compartments are proportions of population
    dt = tvals[1] - tvals[0]
    S = np.zeros(len(tvals)); I = np.zeros(len(tvals)); R = np.zeros(len(tvals));
    Vnull = np.zeros(len(tvals)); Vall = np.zeros(len(tvals))
    for ii,t in enumerate(tvals):
        # base case - set initial conditions
        if ii == 0:
            S[0] = S0; I[0] = I0; R[0] = R0; Vnull[0] = Vnull0; Vall[0] = Vall0;
            continue
        # otherwise, fill in vector using forward euler
        S[ii] = S[ii-1] + (-beta*S[ii-1]*I[ii-1])*dt
        I[ii] = I[ii-1] + (beta*S[ii-1]*I[ii-1] + beta*Vnull[ii-1]*I[ii-1] - gamma*I[ii-1])*dt
        Vnull[ii] = Vnull[ii-1] + (-beta*Vnull[ii-1]*I[ii-1])*dt
        R[ii] = R[ii-1] + (gamma*I[ii-1])*dt
        Vall[ii] = Vall[ii-1]
    return S,I,R,Vnull,Vall

def SIR_Leaky_vax(S0,I0,R0,V0,VE,beta,gamma,tvals):
    # Leaky Vaccine Model
    # Assumes all compartments are proportions of population
    dt = tvals[1] - tvals[0]
    S = np.zeros(len(tvals)); I = np.zeros(len(tvals)); R = np.zeros(len(tvals));
    V = np.zeros(len(tvals));
    for ii,t in enumerate(tvals):
        # base case - set initial conditions
        if ii == 0:
            S[0] = S0; I[0] = I0; R[0] = R0; V[0] = V0;
            continue
        # otherwise, fill in vector using forward euler
        S[ii] = S[ii-1] + (-beta*S[ii-1]*I[ii-1])*dt
        I[ii] = I[ii-1] + (beta*I[ii-1]*(S[ii-1]+V[ii-1]*(1-VE)) - gamma*I[ii-1])*dt
        V[ii] = V[ii-1] + (-beta*S[ii-1]*I[ii-1]*(1-VE))*dt
        R[ii] = R[ii-1] + (gamma*I[ii-1])*dt
    return S,I,R,V
