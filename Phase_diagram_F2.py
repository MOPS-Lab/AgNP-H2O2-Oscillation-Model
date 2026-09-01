# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:13:31 2026

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
h = 15.0

#DIFFERENTIAL EQUATIONS

def system(t, y):

    theta = y[0]
    phi = y[1]

    s = 1.0 - theta - phi

    dtheta_dt = (
        a*h*s
        - b*theta
        - c*h*theta*s**2
        - theta*s
    )

    dphi_dt = (
        theta*s
        - d*h*phi
    )

    return [dtheta_dt, dphi_dt]

#TIME SETTINGS

t_start = 0
t_end = 1000

t_eval = np.linspace(
    t_start,
    t_end,
    4000
)

#INITIAL CONDITIONS

theta_vals = np.linspace(0, 1, 40)
phi_vals = np.linspace(0, 1, 40)

#ATTRACTORS

attractors = []

colors = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "black",
    "brown",
    "cyan"
]

bad_trajectories = 0

#PHASE DIAGRAM

plt.figure(figsize=(10,10))

for theta0 in theta_vals:

    for phi0 in phi_vals:

        s0 = 1 - theta0 - phi0
        
        #PHYSICAL INITIAL CONDITIONS
       
        if theta0 < 0 or theta0 > 1:
            continue

        if phi0 < 0 or phi0 > 1:
            continue

        if s0 < 0 or s0 > 1:
            continue

        y0 = [theta0, phi0]

        #SOLVE SYSTEM

        solution = solve_ivp(
            system,
            [t_start, t_end],
            y0,
            t_eval=t_eval,
            method='BDF',
            rtol=1e-8,
            atol=1e-10
        )

        theta = solution.y[0]
        phi = solution.y[1]

        s = 1 - theta - phi

        #CHECK PHYSICAL REGION

        if (
            np.any(theta < -1e-8)
            or np.any(theta > 1 + 1e-8)
            or np.any(phi < -1e-8)
            or np.any(phi > 1 + 1e-8)
            or np.any(s < -1e-8)
            or np.any(s > 1 + 1e-8)
        ):

            bad_trajectories += 1
            continue

        #DETERMINE ATTRACTOR

        final_state = np.array([
            theta[-1],
            phi[-1]
        ])

        tolerance = 1e-3

        attractor_index = None

        for i, attractor in enumerate(attractors):

            distance = np.linalg.norm(
                final_state - attractor
            )

            if distance < tolerance:

                attractor_index = i
                break

        if attractor_index is None:

            attractors.append(final_state)

            attractor_index = len(attractors) - 1

        color = colors[
            attractor_index % len(colors)
        ]

    
        #DRAW TRAJECTORY
        

        plt.plot(
            theta,
            phi,
            linewidth=0.7,
            color=color
        )

        plt.plot(
            theta[0],
            phi[0],
            'o',
            markersize=2,
            color=color
        )

        plt.plot(
            theta[-1],
            phi[-1],
            's',
            markersize=3,
            color=color
        )


#PHYSICAL BOUNDARY

x = np.linspace(0, 1, 500)

plt.plot(
    x,
    1 - x,
    'k--',
    linewidth=2,
    label=r'$\theta+\phi=1$'
)

#EQUILIBRIA FROM CUBIC ROOTS

A = c*h
B = a/d + 1 - c*h
C = a*h + b - 1

roots = np.roots([A, B, C, -b])

roots = roots[np.abs(np.imag(roots)) < 1e-8]
roots = np.real(roots)
roots.sort()

colors = ["green", "red", "green"]

for i, s in enumerate(roots):

    theta_eq = d*h*(1-s)/(d*h+s)
    phi_eq = s*(1-s)/(d*h+s)

    plt.plot(
        theta_eq,
        phi_eq,
        '^',
        markersize=10,
        color=colors[i],
        markeredgecolor='black',
        zorder=10
)

#PLOT SETTINGS

plt.xlabel(
    r'$\theta$',
    fontsize=14
)

plt.ylabel(
    r'$\phi$',
    fontsize=14
)

plt.title(
    'Phase Space Trajectories',
    fontsize=16
)

plt.xlim(0,1)
plt.ylim(0,1)

plt.grid(True)
plt.legend()

plt.show()

#ATTRACTORS

print("\nDetected attractors:\n")

for i, attractor in enumerate(attractors):

    print(
        f"{i}: {attractor}"
    )

print(
    "\nNumber of trajectories leaving physical region:",
    bad_trajectories
)