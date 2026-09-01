# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:16:59 2026

@author: TilenK
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#PARAMETERS

a = 2.040
b = 0.835
c = 16.647
d = 0.0420
h = 15

I = 1.24      #interaction strength
sign = 1

#INITIAL CONDITIONS

theta_i0 = 0.5255
phi_i0 = 0.263

theta_j0 = 0.5793
phi_j0 = 0.263

#PHYSICAL CHECK

s_i0 = 1 - theta_i0 - phi_i0
s_j0 = 1 - theta_j0 - phi_j0

if s_i0 < 0:
    raise ValueError(
        "Particle i initial condition is not physical."
    )

if s_j0 < 0:
    raise ValueError(
        "Particle j initial condition is not physical."
    )

#DIFFERENTIAL EQUATIONS

def system(t, y):

    theta_i, phi_i, theta_j, phi_j = y

    s_i = 1.0 - theta_i - phi_i
    s_j = 1.0 - theta_j - phi_j

    dtheta_i = (
        a*h*s_i
        - b*theta_i
        - c*h*theta_i*s_i**2
        - theta_i*s_i
    )

    dtheta_j = (
        a*h*s_j
        - b*theta_j
        - c*h*theta_j*s_j**2
        - theta_j*s_j
    )

    dphi_i = (
        theta_i*s_i
        - d*h*phi_i
        + sign*I*dtheta_j
    )

    dphi_j = (
        theta_j*s_j
        - d*h*phi_j
        + sign*I*dtheta_i
    )

    return [
        dtheta_i,
        dphi_i,
        dtheta_j,
        dphi_j
    ]

#TIME SETTINGS

t_start = 0
t_end = 100

dt = 0.0001      #time step

t_eval = np.arange(
    t_start,
    t_end + dt,
    dt
)

#SOLVE SYSTEM

y0 = [
    theta_i0,
    phi_i0,
    theta_j0,
    phi_j0
]

# PHYSICAL PROJECTION

def project(theta, phi):
    theta = max(theta, 0.0)
    phi = max(phi, 0.0)
    if theta + phi > 1.0:
        s = theta + phi
        theta /= s
        phi /= s
    return theta, phi

Y = np.zeros((4,len(t_eval)))
Y[:,0]=y0

for k in range(1,len(t_eval)):
    sol = solve_ivp(
        system,
        (t_eval[k-1], t_eval[k]),
        Y[:,k-1],
        method='BDF',
        rtol=1e-8,
        atol=1e-10
    )
    y = sol.y[:,-1]
    y[0],y[1]=project(y[0],y[1])
    y[2],y[3]=project(y[2],y[3])
    Y[:,k]=y

theta_i=Y[0]
phi_i=Y[1]
theta_j=Y[2]
phi_j=Y[3]

#CHECK PHYSICAL REGION

s_i = 1 - theta_i - phi_i
s_j = 1 - theta_j - phi_j

if (
    np.any(theta_i < -1e-8)
    or np.any(phi_i < -1e-8)
    or np.any(s_i < -1e-8)
    or np.any(theta_j < -1e-8)
    or np.any(phi_j < -1e-8)
    or np.any(s_j < -1e-8)
):
    print("WARNING: trajectory left physical region")

#PHASE DIAGRAM

plt.figure(figsize=(10,10))

#particle i

plt.plot(
    theta_i,
    phi_i,
    linewidth=2,
    label='particle i',
    color="red"
)

plt.plot(
    theta_i[0],
    phi_i[0],
    'o',
    markersize=8
)

#particle j

plt.plot(
    theta_j,
    phi_j,
    linewidth=2,
    label='particle j',
    color="blue"
)

plt.plot(
    theta_j[0],
    phi_j[0],
    'o',
    markersize=8
)

#EQUILIBRIA FROM CUBIC ROOTS

A = c*h
B = a/d + 1 - c*h
C = a*h + b - 1

roots = np.roots([A, B, C, -b])

roots = roots[np.abs(np.imag(roots)) < 1e-8]
roots = np.real(roots)
roots.sort()

eq_colors = ["green", "red", "green"]

for k, s in enumerate(roots):

    theta_eq = d*h*(1-s)/(d*h+s)
    phi_eq = s*(1-s)/(d*h+s)

    plt.plot(
        theta_eq,
        phi_eq,
        '^',
        markersize=12,
        color=eq_colors[k],
        markeredgecolor='black',
        zorder=20
    )

x = np.linspace(0,1,500)

plt.plot(
    x,
    1-x,
    'k--',
    linewidth=2,
    label=r'$\theta+\phi=1$'
)

plt.xlabel(r'$\theta$')
plt.ylabel(r'$\phi$')

plt.title(
    f'Coupled particles phase diagram (I={I})'
)

plt.xlim(0,1)
plt.ylim(0,1)

plt.grid(True)
plt.legend()

plt.show()