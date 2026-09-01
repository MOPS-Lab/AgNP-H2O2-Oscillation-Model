# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:15:06 2026

@author: TilenK
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
h = np.array([7.5, 9.375, 11.25, 13.125, 15.0, 16.875, 18.75, 22.5])

# Experimental data
experiment = np.array([
    0.12987,
    0.15000,
    0.18000,
    0.19000,
    0.19500,
    0.204082,
    0.229885,
    0.288462
])

# Simulation data
simulation = np.array([
    0.135238,
    0.152219,
    0.170400,
    0.193733,
    0.213000,
    0.236667,
    0.258182,
    0.304503
])

# Figure
plt.figure(figsize=(8, 5.5), dpi=150)

# Experimental data – blue points only
plt.scatter(
    h,
    experiment,
    color='tab:blue',
    s=100,
    label='Experiment',
    zorder=3
)

# Simulation – orange dashed line only
plt.plot(
    h,
    simulation,
    color='tab:orange',
    linestyle='--',
    linewidth=2.5,
    label='Simulation',
    zorder=2
)

# Axis labels
plt.xlabel(r'$h$', fontsize=14)
plt.ylabel(r'Frequency (min$^{-1}$)', fontsize=14)

# Axis limits
plt.xlim(0, 25)
plt.ylim(0, 0.35)

# Ticks
plt.xticks(np.arange(0, 26, 5), fontsize=12)
plt.yticks(np.arange(0, 0.351, 0.05), fontsize=12)

# Grid
plt.grid(
    True,
    linestyle='--',
    linewidth=0.8,
    alpha=0.5
)

# Legend
plt.legend(
    loc='upper left',
    fontsize=11,
    frameon=True
)

# Layout
plt.tight_layout()

# Optional: save publication-quality figure
plt.savefig(
    'frequency_vs_h.png',
    dpi=600,
    bbox_inches='tight'
)

plt.show()