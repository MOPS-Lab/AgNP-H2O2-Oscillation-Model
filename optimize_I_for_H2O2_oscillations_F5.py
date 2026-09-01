# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:43:26 2026

@author: TilenK
"""

# -*- coding: utf-8 -*-
"""
Determination of parameter \(I\) for selected H₂O₂ concentrations (parameter \(h\)), optimized to obtain the most stable and well-defined oscillations.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


# ==========================================================
# MODEL PARAMETERS
# ==========================================================

a  = 3.769
b  = 1.790
c  = 26.116
d  = 0.01345

sign = 1.0


# ==========================================================
# h VALUES
# ==========================================================

h_values = np.array([
    7.5, 
    9.375, 
    11.25, 
    13.125, 
    15.0, 
    16.875, 
    18.75, 
    22.5
])


# ==========================================================
# INITIAL CONDITIONS
# ==========================================================

theta_i0 = 0.5255
phi_i0 = 0.2630

theta_j0 = 0.5793
phi_j0 = 0.2630

y0 = np.array([
    theta_i0,
    phi_i0,
    theta_j0,
    phi_j0
], dtype=float)


# ==========================================================
# PHYSICAL CHECK
# ==========================================================

def check_initial_condition(theta, phi, particle_name):

    if theta < 0.0 or phi < 0.0:
        raise ValueError(
            f"Initial values for particle {particle_name} "
            "must be non-negative."
        )

    if theta + phi > 1.0:
        raise ValueError(
            f"Initial condition for particle {particle_name} "
            f"is not physical: theta + phi = "
            f"{theta + phi:.6f} > 1."
        )


check_initial_condition(theta_i0, phi_i0, "i")
check_initial_condition(theta_j0, phi_j0, "j")


# ==========================================================
# DIFFERENTIAL EQUATIONS
# ==========================================================

def system(t, y, h, I):

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
t_end = 500.0

dt_output = 0.05

t_eval = np.arange(
    t_start,
    t_end + dt_output,
    dt_output
)

# Late intervals used for analysis
window_1 = (300.0, 400.0)
window_2 = (400.0, 500.0)

# Final interval used for peak and shape analysis
analysis_start = 400.0


# ==========================================================
# I SEARCH SETTINGS
# ==========================================================

I_min = 0.5
I_max = 3.5

# Coarse search
I_coarse_step = 0.05

# Fine search around the best coarse value
I_fine_half_width = 0.08
I_fine_step = 0.002

I_coarse_values = np.arange(
    I_min,
    I_max + I_coarse_step,
    I_coarse_step
)


# ==========================================================
# OSCILLATION-DETECTION SETTINGS
# ==========================================================

relative_prominence = 0.10
minimum_absolute_prominence = 1e-10

minimum_peak_distance_time = 1.0

minimum_peak_distance_points = max(
    1,
    int(minimum_peak_distance_time / dt_output)
)

minimum_number_of_peaks = 4

# Oscillations below this amplitude are not treated as
# physically meaningful sustained oscillations.
minimum_amplitude = 1e-4

# Amplitude in the final window must remain at least this
# fraction of the amplitude in the previous window.
minimum_amplitude_ratio = 0.80

# Reject cases where the amplitude grows without stabilizing.
maximum_amplitude_ratio = 1.20


# ==========================================================
# QUALITY-SCORE WEIGHTS
# ==========================================================
#
# Larger values give more importance to a criterion.
#
# The score is heuristic. It identifies regular and smooth
# oscillations but is not a fitted physical parameter.
# ==========================================================

weight_amplitude = 1.0
weight_amplitude_stability = 2.0
weight_period_regularity = 2.0
weight_peak_regularity = 1.5
weight_smoothness = 1.5
weight_sinusoidal_shape = 1.0


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def coefficient_of_variation(values):
    """
    Standard deviation divided by the absolute mean.
    Returns infinity if the mean is zero or data are insufficient.
    """

    values = np.asarray(values)

    if len(values) < 2:
        return np.inf

    mean_value = np.mean(values)

    if abs(mean_value) < 1e-14:
        return np.inf

    return np.std(values) / abs(mean_value)


def interpolate_cycle(t_cycle, y_cycle, number_of_points=200):
    """
    Interpolate one oscillation cycle onto a uniform normalized
    time coordinate from 0 to 1.
    """

    if len(t_cycle) < 3:
        return None, None

    duration = t_cycle[-1] - t_cycle[0]

    if duration <= 0.0:
        return None, None

    normalized_time = (
        t_cycle - t_cycle[0]
    ) / duration

    uniform_time = np.linspace(
        0.0,
        1.0,
        number_of_points
    )

    uniform_signal = np.interp(
        uniform_time,
        normalized_time,
        y_cycle
    )

    return uniform_time, uniform_signal


def calculate_sinusoidal_shape_error(
    t_analysis,
    theta_analysis,
    peaks
):
    """
    Compare complete cycles with a fitted first-harmonic sinusoid.

    A lower value means a more sinusoidal and smoother waveform.
    """

    if len(peaks) < 3:
        return np.inf

    cycle_errors = []

    # Each cycle is taken between two consecutive peaks.
    for index in range(len(peaks) - 1):

        start_index = peaks[index]
        stop_index = peaks[index + 1]

        t_cycle = t_analysis[
            start_index:stop_index + 1
        ]

        y_cycle = theta_analysis[
            start_index:stop_index + 1
        ]

        uniform_time, uniform_signal = interpolate_cycle(
            t_cycle,
            y_cycle
        )

        if uniform_signal is None:
            continue

        signal_mean = np.mean(uniform_signal)
        centered_signal = uniform_signal - signal_mean

        amplitude_scale = np.ptp(uniform_signal)

        if amplitude_scale <= 1e-14:
            continue

        # Linear least-squares fit:
        # y = A*sin(2*pi*x) + B*cos(2*pi*x) + C
        design_matrix = np.column_stack(
            (
                np.sin(2.0 * np.pi * uniform_time),
                np.cos(2.0 * np.pi * uniform_time),
                np.ones_like(uniform_time)
            )
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix,
            uniform_signal,
            rcond=None
        )

        fitted_signal = (
            design_matrix @ coefficients
        )

        normalized_rmse = (
            np.sqrt(
                np.mean(
                    (
                        uniform_signal
                        - fitted_signal
                    )**2
                )
            )
            / amplitude_scale
        )

        cycle_errors.append(normalized_rmse)

    if len(cycle_errors) == 0:
        return np.inf

    return np.mean(cycle_errors)


# ==========================================================
# SIMULATION AND OSCILLATION ANALYSIS
# ==========================================================

def simulate_and_analyse(h, I):

    solution = solve_ivp(
        fun=lambda t, y: system(t, y, h, I),
        t_span=(t_start, t_end),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.2
    )

    invalid_result = {
        "valid": False,
        "score": -np.inf,
        "amplitude": np.nan,
        "amplitude_ratio": np.nan,
        "tau_period": np.nan,
        "period_cv": np.nan,
        "peak_height_cv": np.nan,
        "normalized_max_slope": np.nan,
        "sinusoidal_error": np.nan,
        "number_of_peaks": 0,
        "solution": None
    }

    if not solution.success:
        return invalid_result

    t = solution.t
    theta = solution.y[0]

    if not np.all(np.isfinite(theta)):
        return invalid_result

    # ------------------------------------------------------
    # AMPLITUDE STABILITY
    # ------------------------------------------------------

    mask_window_1 = (
        (t >= window_1[0])
        & (t <= window_1[1])
    )

    mask_window_2 = (
        (t >= window_2[0])
        & (t <= window_2[1])
    )

    theta_window_1 = theta[mask_window_1]
    theta_window_2 = theta[mask_window_2]

    if (
        len(theta_window_1) < 2
        or len(theta_window_2) < 2
    ):
        return invalid_result

    amplitude_1 = np.ptp(theta_window_1)
    amplitude_2 = np.ptp(theta_window_2)

    if amplitude_1 > 1e-14:
        amplitude_ratio = amplitude_2 / amplitude_1
    else:
        amplitude_ratio = 0.0

    amplitude = amplitude_2

    # ------------------------------------------------------
    # FINAL ANALYSIS WINDOW
    # ------------------------------------------------------

    analysis_mask = t >= analysis_start

    t_analysis = t[analysis_mask]
    theta_analysis = theta[analysis_mask]

    if len(t_analysis) < 3:
        return invalid_result

    analysis_amplitude = np.ptp(theta_analysis)

    prominence_value = max(
        relative_prominence * analysis_amplitude,
        minimum_absolute_prominence
    )

    peaks, _ = find_peaks(
        theta_analysis,
        prominence=prominence_value,
        distance=minimum_peak_distance_points
    )

    number_of_peaks = len(peaks)

    if number_of_peaks >= 2:

        peak_times = t_analysis[peaks]
        periods = np.diff(peak_times)

        tau_period = np.mean(periods)
        period_cv = coefficient_of_variation(periods)

        peak_heights = theta_analysis[peaks]
        peak_height_cv = coefficient_of_variation(
            peak_heights
        )

    else:

        tau_period = np.nan
        period_cv = np.inf
        peak_height_cv = np.inf

    # ------------------------------------------------------
    # SMOOTHNESS
    # ------------------------------------------------------

    dtheta = np.gradient(
        theta_analysis,
        t_analysis
    )

    max_absolute_slope = np.max(
        np.abs(dtheta)
    )

    if (
        np.isfinite(tau_period)
        and amplitude > 1e-14
    ):
        # Dimensionless normalized slope.
        # For similarly shaped oscillations this should be
        # comparable across amplitudes and periods.
        normalized_max_slope = (
            max_absolute_slope
            * tau_period
            / amplitude
        )
    else:
        normalized_max_slope = np.inf

    # ------------------------------------------------------
    # SINUSOIDAL SHAPE ERROR
    # ------------------------------------------------------

    sinusoidal_error = calculate_sinusoidal_shape_error(
        t_analysis,
        theta_analysis,
        peaks
    )

    # ------------------------------------------------------
    # VALIDITY CHECK
    # ------------------------------------------------------

    valid = (
        number_of_peaks >= minimum_number_of_peaks
        and amplitude >= minimum_amplitude
        and minimum_amplitude_ratio
        <= amplitude_ratio
        <= maximum_amplitude_ratio
        and np.isfinite(tau_period)
        and tau_period > 0.0
        and np.isfinite(period_cv)
        and np.isfinite(peak_height_cv)
        and np.isfinite(normalized_max_slope)
        and np.isfinite(sinusoidal_error)
    )

    if not valid:
        return {
            "valid": False,
            "score": -np.inf,
            "amplitude": amplitude,
            "amplitude_ratio": amplitude_ratio,
            "tau_period": tau_period,
            "period_cv": period_cv,
            "peak_height_cv": peak_height_cv,
            "normalized_max_slope": normalized_max_slope,
            "sinusoidal_error": sinusoidal_error,
            "number_of_peaks": number_of_peaks,
            "solution": solution
        }

    # ------------------------------------------------------
    # QUALITY COMPONENTS
    # ------------------------------------------------------

    # Amplitude score is bounded between 0 and 1.
    amplitude_score = amplitude / (amplitude + 0.1)

    # Best value is amplitude_ratio = 1.
    amplitude_stability_score = np.exp(
        -abs(np.log(amplitude_ratio))
    )

    period_regularity_score = 1.0 / (
        1.0 + 10.0 * period_cv
    )

    peak_regularity_score = 1.0 / (
        1.0 + 10.0 * peak_height_cv
    )

    smoothness_score = 1.0 / (
        1.0 + normalized_max_slope
    )

    sinusoidal_shape_score = 1.0 / (
        1.0 + 10.0 * sinusoidal_error
    )

    # ------------------------------------------------------
    # TOTAL QUALITY SCORE
    # ------------------------------------------------------

    total_weight = (
        weight_amplitude
        + weight_amplitude_stability
        + weight_period_regularity
        + weight_peak_regularity
        + weight_smoothness
        + weight_sinusoidal_shape
    )

    score = (
        weight_amplitude
        * amplitude_score

        + weight_amplitude_stability
        * amplitude_stability_score

        + weight_period_regularity
        * period_regularity_score

        + weight_peak_regularity
        * peak_regularity_score

        + weight_smoothness
        * smoothness_score

        + weight_sinusoidal_shape
        * sinusoidal_shape_score
    ) / total_weight

    return {
        "valid": True,
        "score": score,
        "amplitude": amplitude,
        "amplitude_ratio": amplitude_ratio,
        "tau_period": tau_period,
        "period_cv": period_cv,
        "peak_height_cv": peak_height_cv,
        "normalized_max_slope": normalized_max_slope,
        "sinusoidal_error": sinusoidal_error,
        "number_of_peaks": number_of_peaks,
        "solution": solution
    }


# ==========================================================
# STORAGE
# ==========================================================

optimal_I = np.full(
    len(h_values),
    np.nan
)

optimal_score = np.full(
    len(h_values),
    np.nan
)

optimal_amplitude = np.full(
    len(h_values),
    np.nan
)

optimal_amplitude_ratio = np.full(
    len(h_values),
    np.nan
)

optimal_tau_period = np.full(
    len(h_values),
    np.nan
)

optimal_period_cv = np.full(
    len(h_values),
    np.nan
)

optimal_peak_height_cv = np.full(
    len(h_values),
    np.nan
)

optimal_normalized_slope = np.full(
    len(h_values),
    np.nan
)

optimal_sinusoidal_error = np.full(
    len(h_values),
    np.nan
)

optimal_number_of_peaks = np.zeros(
    len(h_values),
    dtype=int
)

best_solutions = {}


# ==========================================================
# SEARCH FOR BEST I
# ==========================================================

print()
print("Searching for the most regular oscillations...")
print()

for h_index, h in enumerate(h_values):

    print(f"Processing h = {h:.3f}")

    # ------------------------------------------------------
    # COARSE SEARCH
    # ------------------------------------------------------

    best_coarse_I = np.nan
    best_coarse_score = -np.inf
    best_coarse_result = None

    for I in I_coarse_values:

        result = simulate_and_analyse(h, I)

        if not result["valid"]:
            continue

        if result["score"] > best_coarse_score:

            best_coarse_score = result["score"]
            best_coarse_I = I
            best_coarse_result = result

    if not np.isfinite(best_coarse_I):

        print(
            "  No sustained, sufficiently large and regular "
            "oscillations found in the selected I range."
        )

        print()
        continue

    # ------------------------------------------------------
    # FINE SEARCH
    # ------------------------------------------------------

    fine_start = max(
        I_min,
        best_coarse_I - I_fine_half_width
    )

    fine_end = min(
        I_max,
        best_coarse_I + I_fine_half_width
    )

    I_fine_values = np.arange(
        fine_start,
        fine_end + I_fine_step,
        I_fine_step
    )

    best_I = best_coarse_I
    best_score = best_coarse_score
    best_result = best_coarse_result

    for I in I_fine_values:

        result = simulate_and_analyse(h, I)

        if not result["valid"]:
            continue

        if result["score"] > best_score:

            best_score = result["score"]
            best_I = I
            best_result = result

    # ------------------------------------------------------
    # STORE RESULT
    # ------------------------------------------------------

    optimal_I[h_index] = best_I
    optimal_score[h_index] = best_score
    optimal_amplitude[h_index] = (
        best_result["amplitude"]
    )

    optimal_amplitude_ratio[h_index] = (
        best_result["amplitude_ratio"]
    )

    optimal_tau_period[h_index] = (
        best_result["tau_period"]
    )

    optimal_period_cv[h_index] = (
        best_result["period_cv"]
    )

    optimal_peak_height_cv[h_index] = (
        best_result["peak_height_cv"]
    )

    optimal_normalized_slope[h_index] = (
        best_result["normalized_max_slope"]
    )

    optimal_sinusoidal_error[h_index] = (
        best_result["sinusoidal_error"]
    )

    optimal_number_of_peaks[h_index] = (
        best_result["number_of_peaks"]
    )

    best_solutions[h] = best_result["solution"]

    print(f"  Best I                 = {best_I:.4f}")
    print(f"  Oscillation score      = {best_score:.6f}")
    print(
        f"  Amplitude              = "
        f"{best_result['amplitude']:.6f}"
    )

    print(
        f"  Amplitude ratio        = "
        f"{best_result['amplitude_ratio']:.6f}"
    )

    print(
        f"  Dimensionless period   = "
        f"{best_result['tau_period']:.6f}"
    )

    print(
        f"  Period CV              = "
        f"{best_result['period_cv']:.6f}"
    )

    print(
        f"  Peak-height CV         = "
        f"{best_result['peak_height_cv']:.6f}"
    )

    print(
        f"  Normalized max slope   = "
        f"{best_result['normalized_max_slope']:.6f}"
    )

    print(
        f"  Sinusoidal error       = "
        f"{best_result['sinusoidal_error']:.6f}"
    )

    print()


# ==========================================================
# FINAL TABLE
# ==========================================================

print()
print("=" * 165)

print(
    f"{'h':>9}"
    f"{'Best I':>12}"
    f"{'Score':>12}"
    f"{'Amplitude':>14}"
    f"{'Amp ratio':>14}"
    f"{'Period':>14}"
    f"{'Period CV':>14}"
    f"{'Peak CV':>14}"
    f"{'Norm slope':>15}"
    f"{'Sine error':>14}"
    f"{'Peaks':>10}"
)

print("=" * 165)

for index in range(len(h_values)):

    print(
        f"{h_values[index]:9.3f}"
        f"{optimal_I[index]:12.4f}"
        f"{optimal_score[index]:12.6f}"
        f"{optimal_amplitude[index]:14.6f}"
        f"{optimal_amplitude_ratio[index]:14.6f}"
        f"{optimal_tau_period[index]:14.6f}"
        f"{optimal_period_cv[index]:14.6f}"
        f"{optimal_peak_height_cv[index]:14.6f}"
        f"{optimal_normalized_slope[index]:15.6f}"
        f"{optimal_sinusoidal_error[index]:14.6f}"
        f"{optimal_number_of_peaks[index]:10d}"
    )

print("=" * 165)


# ==========================================================
# PLOT BEST I VERSUS h
# ==========================================================

valid = np.isfinite(optimal_I)

plt.figure(figsize=(9, 6))

plt.plot(
    h_values[valid],
    optimal_I[valid],
    marker="o",
    linewidth=2
)

plt.xlabel(
    r"Dimensionless H$_2$O$_2$ concentration, $h$",
    fontsize=13
)

plt.ylabel(
    r"Best coupling parameter, $I$",
    fontsize=13
)

plt.title(
    r"Coupling parameter producing the most regular oscillations"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================================
# PLOT QUALITY SCORE
# ==========================================================

plt.figure(figsize=(9, 6))

plt.plot(
    h_values[valid],
    optimal_score[valid],
    marker="o",
    linewidth=2
)

plt.xlabel(
    r"Dimensionless H$_2$O$_2$ concentration, $h$",
    fontsize=13
)

plt.ylabel(
    "Oscillation quality score",
    fontsize=13
)

plt.title(
    "Quality of the best oscillatory solution"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================================
# PLOT BEST TIME SERIES
# ==========================================================

plot_start = 450.0
plot_end = 500.0

plt.figure(figsize=(11, 7))

for h, solution in best_solutions.items():

    result_index = np.argmin(
        np.abs(h_values - h)
    )

    plot_mask = (
        (solution.t >= plot_start)
        & (solution.t <= plot_end)
    )

    plt.plot(
        solution.t[plot_mask],
        solution.y[0, plot_mask],
        linewidth=1.5,
        label=(
            rf"$h={h:.3f}$, "
            rf"$I={optimal_I[result_index]:.3f}$"
        )
    )

plt.xlabel(
    r"$\tau$",
    fontsize=13
)

plt.ylabel(
    r"$\theta_i$",
    fontsize=13
)

plt.title(
    r"Best late-time oscillations for each value of $h$"
)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# INDIVIDUAL PLOTS
# ==========================================================

for h, solution in best_solutions.items():

    result_index = np.argmin(
        np.abs(h_values - h)
    )

    plot_mask = (
        (solution.t >= plot_start)
        & (solution.t <= plot_end)
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        solution.t[plot_mask],
        solution.y[0, plot_mask],
        linewidth=2
    )

    plt.xlabel(
        r"$\tau$",
        fontsize=13
    )

    plt.ylabel(
        r"$\theta_i$",
        fontsize=13
    )

    plt.title(
        (
            rf"$h={h:.3f}$, "
            rf"$I={optimal_I[result_index]:.3f}$, "
            rf"score={optimal_score[result_index]:.3f}"
        )
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================================
# SAVE RESULTS
# ==========================================================

output_data = np.column_stack(
    (
        h_values,
        optimal_I,
        optimal_score,
        optimal_amplitude,
        optimal_amplitude_ratio,
        optimal_tau_period,
        optimal_period_cv,
        optimal_peak_height_cv,
        optimal_normalized_slope,
        optimal_sinusoidal_error,
        optimal_number_of_peaks
    )
)

header = (
    "h,"
    "best_I,"
    "oscillation_quality_score,"
    "late_amplitude,"
    "late_to_previous_amplitude_ratio,"
    "dimensionless_period_tau,"
    "period_coefficient_of_variation,"
    "peak_height_coefficient_of_variation,"
    "normalized_maximum_slope,"
    "sinusoidal_shape_error,"
    "number_of_detected_peaks"
)

np.savetxt(
    "best_I_for_regular_oscillations.csv",
    output_data,
    delimiter=",",
    header=header,
    comments=""
)

print()
print(
    "Results saved to "
    "best_I_for_regular_oscillations.csv"
)