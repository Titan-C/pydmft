# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:52:28 2015

@author: oscar
"""
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
N = 9


def bc(s, N):
    if s == -1:
        return N-1, -1
    elif s == N:
        return 0, -1
    else:
        return s, 1


def state_label(i, j):
    return 2**i + 2**j


def get_index(state, states):
    return states.index(state)

v = [(i, j, state_label(i, j)) for i in range(N) for j in range(i+1, N)]
states = [state_label(i, j) for i in range(N) for j in range(i+1, N)]
states = sorted(states)
hops = []
for i in range(N):
    for j in range(i+1, N):
        state = state_label(i, j)
        # cases
        ns, si = bc(i+1, N)
        if ns != j:
            hops.append((state, state_label(ns, j), si))

        ns, si = bc(j+1, N)
        if ns != i:
            hops.append((state, state_label(i, ns), si))

#        h.c.
#        s=bc(i-1, N)
#        if s!=j:
#            hops.append((state, state_label(s,j)))


#        s=bc(j-1, N)
#        if s!=i:
#            hops.append((state, state_label(i,s)))
H = np.zeros((len(states), len(states)))
t = 1  # 0.5
for sta, end, sig in hops:
    j = get_index(sta, states)
    i = get_index(end, states)
    H[i, j] = -t*sig
H += H.T
e, v = LA.eigh(H)
plt.figure()
plt.imshow(H, interpolation='none')
plt.colorbar()
#hist(h, 32, normed=True)
hist(e, sqrt(len(states)), normed=True)
ho=np.eye(1024,k=1)
H1p=-0.5*(ho+ho.T)
H1p[-1,0]=-0.5
H1p[0,-1]=-0.5

e1,v1=LA.eigh(H1p)