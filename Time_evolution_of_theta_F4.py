# -*- coding: utf-8 -*-
"""
Time evolution of two coupled silver nanoparticles.

The model is integrated in a single solve_ivp call.
Only the interval tau = 5–55 is displayed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ==========================================================
# PARAMETERS
# ==========================================================

a = 2.040
b = 0.835
c = 16.647
d = 0.0420

h = 15.0

I = 1.24
sign = 1.0


# ==========================================================
# INITIAL CONDITIONS
# ==========================================================

theta_i0 = 0.5255
phi_i0 = 0.2630

theta_j0 = 0.5793
phi_j0 = 0.2630

y0 = [
    theta_i0,
    phi_i0,
    theta_j0,
    phi_j0
]


# ==========================================================
# PHYSICAL CHECK OF INITIAL CONDITIONS
# ==========================================================

if theta_i0 < 0 or phi_i0 < 0:
    raise ValueError(
        "Particle i initial values must be non-negative."
    )

if theta_j0 < 0 or phi_j0 < 0:
    raise ValueError(
        "Particle j initial values must be non-negative."
    )

if theta_i0 + phi_i0 > 1:
    raise ValueError(
        "Particle i initial condition is not physical."
    )

if theta_j0 + phi_j0 > 1:
    raise ValueError(
        "Particle j initial condition is not physical."
    )


# ==========================================================
# DIFFERENTIAL EQUATIONS
# ==========================================================

def system(t, y):

    theta_i, phi_i, theta_j, phi_j = y

    s_i = 1.0 - theta_i - phi_i
    s_j = 1.0 - theta_j - phi_j

    dtheta_i = (
        a * h * s_i
        - b * theta_i
        - c * h * theta_i * s_i**2
        - theta_i * s_i
    )

    dtheta_j = (
        a * h * s_j
        - b * theta_j
        - c * h * theta_j * s_j**2
        - theta_j * s_j
    )

    dphi_i = (
        theta_i * s_i
        - d * h * phi_i
        + sign * I * dtheta_j
    )

    dphi_j = (
        theta_j * s_j
        - d * h * phi_j
        + sign * I * dtheta_i
    )

    return [
        dtheta_i,
        dphi_i,
        dtheta_j,
        dphi_j
    ]


# ==========================================================
# TIME SETTINGS
# ==========================================================

t_start = 0.0
t_end = 55.0

# This is only the output spacing.
# The solver internally selects its own integration steps.
dt_output = 0.005

t_eval = np.arange(
    t_start,
    t_end + dt_output,
    dt_output
)


# ==========================================================
# SOLVE SYSTEM
# ==========================================================

solution = solve_ivp(
    fun=system,
    t_span=(t_start, t_end),
    y0=y0,
    method="BDF",
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
    max_step=0.05
)

if not solution.success:
    raise RuntimeError(
        "Integration failed: "
        + solution.message
    )


# ==========================================================
# EXTRACT RESULTS
# ==========================================================

t = solution.t

theta_i = solution.y[0]
phi_i = solution.y[1]

theta_j = solution.y[2]
phi_j = solution.y[3]

s_i = 1.0 - theta_i - phi_i
s_j = 1.0 - theta_j - phi_j


# ==========================================================
# CHECK PHYSICAL REGION
# ==========================================================

physical_tolerance = 1e-7

if (
    np.any(theta_i < -physical_tolerance)
    or np.any(phi_i < -physical_tolerance)
    or np.any(s_i < -physical_tolerance)
    or np.any(theta_j < -physical_tolerance)
    or np.any(phi_j < -physical_tolerance)
    or np.any(s_j < -physical_tolerance)
):
    print(
        "WARNING: The numerical solution leaves "
        "the physically meaningful region."
    )


# ==========================================================
# PLOT THETA(t)
# ==========================================================

plot_mask = (
    (t >= 5.0)
    & (t <= 55.0)
)

plt.figure(figsize=(10, 6))

plt.plot(
    t[plot_mask],
    theta_i[plot_mask],
    linewidth=1.5,
    label=r"$\theta_i(\tau)$"
)

plt.plot(
    t[plot_mask],
    theta_j[plot_mask],
    linewidth=1.5,
    label=r"$\theta_j(\tau)$"
)

plt.xlabel(
    r"$\tau$",
    fontsize=13
)

plt.ylabel(
    r"$\theta$",
    fontsize=13
)

plt.title(
    rf"Time evolution of $\theta$ ($I={I}$)",
    fontsize=14
)

plt.xlim(20, 55)
plt.ylim(0.8, 1)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()