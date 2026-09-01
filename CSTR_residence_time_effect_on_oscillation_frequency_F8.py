# -*- coding: utf-8 -*-
"""
Influence of CSTR residence time on the oscillation frequency
of the H2O2-AgNP system.

The script is based on the original coupled two-nanoparticle model.
Only the following extension is added:

    dh/dtau = (h_in - h)/tau_R_star - beta_surface*r_H2O2

where:

    tau_R_star = k3*t_R = k3*V/Q

To isolate the effect of residence time:

    h_in = constant
    I = constant
    beta_surface = constant

and only t_R is varied.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


# ==========================================================
# 1. MODEL PARAMETERS
# ==========================================================

a = 3.769
b = 1.790
c = 26.116
d = 0.01345

sign = 1.0

# Dimensional kinetic scaling constant
k3 = 0.426  # min^-1


# ==========================================================
# 2. FIXED EXPERIMENTAL / MODEL CONDITIONS
# ==========================================================

# Constant inlet concentration of H2O2
h_in = 15.0

# Initial concentration in the reactor
h0 = h_in

# Fixed interparticle coupling parameter.
# Do not optimize this separately for each residence time.
I_fixed = 1.11

# Surface-to-bulk conversion parameter.
#
# beta_surface = 0.0:
#     control calculation; residence time has no effect.
#
# beta_surface > 0.0:
#     active CSTR calculation with H2O2 depletion.
#
# Suggested tests:
#     0.001
#     0.005
#     0.010
#     0.020
#     0.050
beta_surface = 0.01


# ==========================================================
# 3. RESIDENCE-TIME SETTINGS
# ==========================================================

# Choose:
#     residence_time_mode = "direct"
# or:
#     residence_time_mode = "volume_and_flow"

residence_time_mode = "direct"


# ----------------------------------------------------------
# OPTION A: residence time entered directly
# ----------------------------------------------------------

# Residence times from 0 to 10 min in 1 min increments.
# t_R = 0 is treated below as the limiting case t_R -> 0,
# for which the reactor concentration is pinned at h = h_in.
residence_times_min_direct = np.array([
    3.0,
    6.0,
    9.0
], dtype=float)


# ----------------------------------------------------------
# OPTION B: residence time calculated from V/Q
# ----------------------------------------------------------

reactor_volume_ml = 10.0

flow_rates_ml_min = np.array([
    20.0,
    10.0,
    5.0,
    2.0,
    1.0,
    0.5,
    0.333333333,
    0.222222222,
    0.166666667
], dtype=float)


def calculate_residence_times():
    """Return residence times in minutes."""

    if residence_time_mode == "direct":

        residence_times = np.asarray(
            residence_times_min_direct,
            dtype=float
        )

    elif residence_time_mode == "volume_and_flow":

        if reactor_volume_ml <= 0.0:
            raise ValueError(
                "reactor_volume_ml must be greater than zero."
            )

        flow_rates = np.asarray(
            flow_rates_ml_min,
            dtype=float
        )

        if np.any(flow_rates <= 0.0):
            raise ValueError(
                "All flow rates must be greater than zero."
            )

        residence_times = (
            reactor_volume_ml
            / flow_rates
        )

    else:

        raise ValueError(
            "residence_time_mode must be either "
            "'direct' or 'volume_and_flow'."
        )

    if np.any(residence_times < 0.0):

        raise ValueError(
            "Residence times must be non-negative."
        )

    return residence_times


residence_times_min = calculate_residence_times()

# Dimensionless residence time because tau = k3*t
tau_R_star_values = (
    k3
    * residence_times_min
)


# ==========================================================
# 4. INITIAL CONDITIONS
# ==========================================================

theta_i0 = 0.5255
phi_i0 = 0.2630

theta_j0 = 0.5793
phi_j0 = 0.2630

y0 = np.array([
    theta_i0,
    phi_i0,
    theta_j0,
    phi_j0,
    h0
], dtype=float)


# ==========================================================
# 5. DIFFERENTIAL EQUATIONS
# ==========================================================

def system(
    tau,
    y,
    tau_R_star
):
    """
    Coupled nanoparticle model with a dynamic CSTR H2O2 balance.

    State vector:
        y[0] = theta_i
        y[1] = phi_i
        y[2] = theta_j
        y[3] = phi_j
        y[4] = h
    """

    theta_i, phi_i, theta_j, phi_j, h = y

    s_i = (
        1.0
        - theta_i
        - phi_i
    )

    s_j = (
        1.0
        - theta_j
        - phi_j
    )

    # ------------------------------------------------------
    # ORIGINAL SURFACE MODEL
    # ------------------------------------------------------

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
        + sign
        * I_fixed
        * dtheta_j
    )

    dphi_j = (
        theta_j * s_j
        - d * h * phi_j
        + sign
        * I_fixed
        * dtheta_i
    )

    # ------------------------------------------------------
    # REPRESENTATIVE H2O2 CONSUMPTION
    # ------------------------------------------------------
    #
    # The surface-only model does not contain catalyst loading,
    # total surface area or active-site concentration. Therefore,
    # beta_surface is required to convert the representative
    # surface rate into a bulk concentration-consumption rate.
    # ------------------------------------------------------

    r_h2o2_i = (
        a * h * s_i
        - b * theta_i
        + c * h * theta_i * s_i**2
        + d * h * phi_i
    )

    r_h2o2_j = (
        a * h * s_j
        - b * theta_j
        + c * h * theta_j * s_j**2
        + d * h * phi_j
    )

    r_h2o2_mean = (
        0.5
        * (
            r_h2o2_i
            + r_h2o2_j
        )
    )

    # ------------------------------------------------------
    # IDEAL CSTR BALANCE
    # ------------------------------------------------------

    # At exactly t_R = 0, tau_R_star = 0 and the standard CSTR
    # balance would contain a division by zero. We therefore treat
    # this point as the limiting case t_R -> 0: infinitely rapid
    # refresh keeps the reactor concentration equal to h_in. Since
    # h0 = h_in, setting dh = 0 keeps h exactly at h_in.
    if tau_R_star == 0.0:
        dh = 0.0
    else:
        dh = (
            (h_in - h)
            / tau_R_star
            - beta_surface
            * r_h2o2_mean
        )

    return [
        dtheta_i,
        dphi_i,
        dtheta_j,
        dphi_j,
        dh
    ]


# ==========================================================
# 6. TIME SETTINGS
# ==========================================================

tau_start = 0.0
tau_end = 500.0

dt_output = 0.05

tau_eval = np.arange(
    tau_start,
    tau_end + dt_output,
    dt_output
)

# Late intervals used for amplitude-stability analysis
window_1 = (
    300.0,
    400.0
)

window_2 = (
    400.0,
    500.0
)

# Final interval for period and shape analysis
analysis_start = 400.0


# ==========================================================
# 7. SOLVER SETTINGS
# ==========================================================

solver_method = "BDF"
solver_rtol = 1e-8
solver_atol = 1e-10
solver_max_step = 0.2


# ==========================================================
# 8. OSCILLATION-DETECTION SETTINGS
# ==========================================================

relative_prominence = 0.10
minimum_absolute_prominence = 1e-10

minimum_peak_distance_tau = 1.0

minimum_peak_distance_points = max(
    1,
    int(
        minimum_peak_distance_tau
        / dt_output
    )
)

minimum_number_of_peaks = 4

minimum_amplitude = 1e-4

minimum_amplitude_ratio = 0.80
maximum_amplitude_ratio = 1.20


# ==========================================================
# 9. QUALITY-SCORE WEIGHTS
# ==========================================================

weight_amplitude = 1.0
weight_amplitude_stability = 2.0
weight_period_regularity = 2.0
weight_peak_regularity = 1.5
weight_smoothness = 1.5
weight_sinusoidal_shape = 1.0


# ==========================================================
# 10. HELPER FUNCTIONS
# ==========================================================

def coefficient_of_variation(values):
    """
    Standard deviation divided by the absolute mean.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 2:
        return np.inf

    mean_value = np.mean(
        values
    )

    if abs(mean_value) < 1e-14:
        return np.inf

    return (
        np.std(values)
        / abs(mean_value)
    )


def interpolate_cycle(
    tau_cycle,
    signal_cycle,
    number_of_points=200
):
    """
    Interpolate one cycle onto a normalized coordinate from 0 to 1.
    """

    if len(tau_cycle) < 3:
        return None, None

    duration = (
        tau_cycle[-1]
        - tau_cycle[0]
    )

    if duration <= 0.0:
        return None, None

    normalized_time = (
        tau_cycle
        - tau_cycle[0]
    ) / duration

    uniform_time = np.linspace(
        0.0,
        1.0,
        number_of_points
    )

    uniform_signal = np.interp(
        uniform_time,
        normalized_time,
        signal_cycle
    )

    return (
        uniform_time,
        uniform_signal
    )


def calculate_sinusoidal_shape_error(
    tau_analysis,
    theta_analysis,
    peaks
):
    """
    Compare complete cycles with a fitted first-harmonic sinusoid.
    """

    if len(peaks) < 3:
        return np.inf

    cycle_errors = []

    for index in range(
        len(peaks) - 1
    ):

        start_index = peaks[index]
        stop_index = peaks[index + 1]

        tau_cycle = tau_analysis[
            start_index:
            stop_index + 1
        ]

        theta_cycle = theta_analysis[
            start_index:
            stop_index + 1
        ]

        uniform_time, uniform_signal = interpolate_cycle(
            tau_cycle,
            theta_cycle
        )

        if uniform_signal is None:
            continue

        amplitude_scale = np.ptp(
            uniform_signal
        )

        if amplitude_scale <= 1e-14:
            continue

        design_matrix = np.column_stack(
            (
                np.sin(
                    2.0
                    * np.pi
                    * uniform_time
                ),
                np.cos(
                    2.0
                    * np.pi
                    * uniform_time
                ),
                np.ones_like(
                    uniform_time
                )
            )
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix,
            uniform_signal,
            rcond=None
        )

        fitted_signal = (
            design_matrix
            @ coefficients
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

        cycle_errors.append(
            normalized_rmse
        )

    if len(cycle_errors) == 0:
        return np.inf

    return np.mean(
        cycle_errors
    )


def invalid_result(
    solution=None,
    reason="invalid"
):
    """Return a complete invalid-result dictionary."""

    return {
        "valid": False,
        "reason": reason,
        "score": -np.inf,
        "amplitude": np.nan,
        "amplitude_ratio": np.nan,
        "tau_period": np.nan,
        "period_min": np.nan,
        "frequency_dimensionless": np.nan,
        "frequency_min": np.nan,
        "period_cv": np.nan,
        "peak_height_cv": np.nan,
        "normalized_max_slope": np.nan,
        "sinusoidal_error": np.nan,
        "number_of_peaks": 0,
        "mean_h": np.nan,
        "minimum_h": np.nan,
        "maximum_h": np.nan,
        "h_amplitude": np.nan,
        "solution": solution
    }


# ==========================================================
# 11. SIMULATION AND ANALYSIS
# ==========================================================

def simulate_and_analyse(
    tau_R_star
):
    """
    Simulate one residence-time condition and analyse oscillations.
    """

    try:

        solution = solve_ivp(
            fun=lambda tau, y: system(
                tau,
                y,
                tau_R_star
            ),
            t_span=(
                tau_start,
                tau_end
            ),
            y0=y0,
            method=solver_method,
            t_eval=tau_eval,
            rtol=solver_rtol,
            atol=solver_atol,
            max_step=solver_max_step
        )

    except Exception as error:

        return invalid_result(
            solution=None,
            reason=(
                f"Solver exception: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

    if not solution.success:

        return invalid_result(
            solution=solution,
            reason=solution.message
        )

    if not np.all(
        np.isfinite(solution.y)
    ):

        return invalid_result(
            solution=solution,
            reason="The numerical solution contains non-finite values."
        )

    tau = solution.t

    theta = solution.y[0]
    h_reactor = solution.y[4]

    # ------------------------------------------------------
    # AMPLITUDE STABILITY
    # ------------------------------------------------------

    mask_window_1 = (
        (tau >= window_1[0])
        & (tau <= window_1[1])
    )

    mask_window_2 = (
        (tau >= window_2[0])
        & (tau <= window_2[1])
    )

    theta_window_1 = theta[
        mask_window_1
    ]

    theta_window_2 = theta[
        mask_window_2
    ]

    if (
        len(theta_window_1) < 2
        or len(theta_window_2) < 2
    ):

        return invalid_result(
            solution=solution,
            reason="Insufficient data in the late-time windows."
        )

    amplitude_1 = np.ptp(
        theta_window_1
    )

    amplitude_2 = np.ptp(
        theta_window_2
    )

    if amplitude_1 > 1e-14:

        amplitude_ratio = (
            amplitude_2
            / amplitude_1
        )

    else:

        amplitude_ratio = 0.0

    amplitude = amplitude_2

    # ------------------------------------------------------
    # FINAL ANALYSIS WINDOW
    # ------------------------------------------------------

    analysis_mask = (
        tau >= analysis_start
    )

    tau_analysis = tau[
        analysis_mask
    ]

    theta_analysis = theta[
        analysis_mask
    ]

    h_analysis = h_reactor[
        analysis_mask
    ]

    if len(tau_analysis) < 3:

        return invalid_result(
            solution=solution,
            reason="Insufficient data in the final analysis window."
        )

    analysis_amplitude = np.ptp(
        theta_analysis
    )

    prominence_value = max(
        relative_prominence
        * analysis_amplitude,
        minimum_absolute_prominence
    )

    peaks, _ = find_peaks(
        theta_analysis,
        prominence=prominence_value,
        distance=minimum_peak_distance_points
    )

    number_of_peaks = len(
        peaks
    )

    if number_of_peaks >= 2:

        peak_times = tau_analysis[
            peaks
        ]

        periods = np.diff(
            peak_times
        )

        tau_period = np.mean(
            periods
        )

        period_cv = coefficient_of_variation(
            periods
        )

        peak_heights = theta_analysis[
            peaks
        ]

        peak_height_cv = coefficient_of_variation(
            peak_heights
        )

        # tau = k3*t
        # therefore:
        # dimensional period = tau_period/k3
        # dimensional frequency = k3/tau_period
        period_min = (
            tau_period
            / k3
        )

        frequency_dimensionless = (
            1.0
            / tau_period
        )

        frequency_min = (
            k3
            / tau_period
        )

    else:

        tau_period = np.nan
        period_min = np.nan
        frequency_dimensionless = np.nan
        frequency_min = np.nan
        period_cv = np.inf
        peak_height_cv = np.inf

    # ------------------------------------------------------
    # SMOOTHNESS
    # ------------------------------------------------------

    dtheta = np.gradient(
        theta_analysis,
        tau_analysis
    )

    max_absolute_slope = np.max(
        np.abs(dtheta)
    )

    if (
        np.isfinite(tau_period)
        and amplitude > 1e-14
    ):

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
        tau_analysis,
        theta_analysis,
        peaks
    )

    # ------------------------------------------------------
    # REACTOR H2O2 STATISTICS
    # ------------------------------------------------------

    mean_h = np.mean(
        h_analysis
    )

    minimum_h = np.min(
        h_analysis
    )

    maximum_h = np.max(
        h_analysis
    )

    h_amplitude = np.ptp(
        h_analysis
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

        reasons = []

        if number_of_peaks < minimum_number_of_peaks:

            reasons.append(
                f"only {number_of_peaks} peaks"
            )

        if amplitude < minimum_amplitude:

            reasons.append(
                f"amplitude {amplitude:.3e} "
                f"< {minimum_amplitude:.3e}"
            )

        if not (
            minimum_amplitude_ratio
            <= amplitude_ratio
            <= maximum_amplitude_ratio
        ):

            reasons.append(
                f"amplitude ratio "
                f"{amplitude_ratio:.4f} "
                f"outside "
                f"[{minimum_amplitude_ratio:.2f}, "
                f"{maximum_amplitude_ratio:.2f}]"
            )

        return {
            "valid": False,
            "reason": "; ".join(reasons),
            "score": -np.inf,
            "amplitude": amplitude,
            "amplitude_ratio": amplitude_ratio,
            "tau_period": tau_period,
            "period_min": period_min,
            "frequency_dimensionless": frequency_dimensionless,
            "frequency_min": frequency_min,
            "period_cv": period_cv,
            "peak_height_cv": peak_height_cv,
            "normalized_max_slope": normalized_max_slope,
            "sinusoidal_error": sinusoidal_error,
            "number_of_peaks": number_of_peaks,
            "mean_h": mean_h,
            "minimum_h": minimum_h,
            "maximum_h": maximum_h,
            "h_amplitude": h_amplitude,
            "solution": solution
        }

    # ------------------------------------------------------
    # QUALITY COMPONENTS
    # ------------------------------------------------------

    amplitude_score = (
        amplitude
        / (
            amplitude
            + 0.1
        )
    )

    amplitude_stability_score = np.exp(
        -abs(
            np.log(
                amplitude_ratio
            )
        )
    )

    period_regularity_score = (
        1.0
        / (
            1.0
            + 10.0
            * period_cv
        )
    )

    peak_regularity_score = (
        1.0
        / (
            1.0
            + 10.0
            * peak_height_cv
        )
    )

    smoothness_score = (
        1.0
        / (
            1.0
            + normalized_max_slope
        )
    )

    sinusoidal_shape_score = (
        1.0
        / (
            1.0
            + 10.0
            * sinusoidal_error
        )
    )

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
        "reason": "valid sustained oscillation",
        "score": score,
        "amplitude": amplitude,
        "amplitude_ratio": amplitude_ratio,
        "tau_period": tau_period,
        "period_min": period_min,
        "frequency_dimensionless": frequency_dimensionless,
        "frequency_min": frequency_min,
        "period_cv": period_cv,
        "peak_height_cv": peak_height_cv,
        "normalized_max_slope": normalized_max_slope,
        "sinusoidal_error": sinusoidal_error,
        "number_of_peaks": number_of_peaks,
        "mean_h": mean_h,
        "minimum_h": minimum_h,
        "maximum_h": maximum_h,
        "h_amplitude": h_amplitude,
        "solution": solution
    }


# ==========================================================
# 12. STORAGE
# ==========================================================

number_of_conditions = len(
    residence_times_min
)

valid_result = np.zeros(
    number_of_conditions,
    dtype=bool
)

oscillation_score = np.full(
    number_of_conditions,
    np.nan
)

oscillation_amplitude = np.full(
    number_of_conditions,
    np.nan
)

amplitude_ratio = np.full(
    number_of_conditions,
    np.nan
)

tau_period = np.full(
    number_of_conditions,
    np.nan
)

period_min = np.full(
    number_of_conditions,
    np.nan
)

frequency_dimensionless = np.full(
    number_of_conditions,
    np.nan
)

frequency_min = np.full(
    number_of_conditions,
    np.nan
)

period_cv = np.full(
    number_of_conditions,
    np.nan
)

peak_height_cv = np.full(
    number_of_conditions,
    np.nan
)

normalized_slope = np.full(
    number_of_conditions,
    np.nan
)

sinusoidal_error = np.full(
    number_of_conditions,
    np.nan
)

number_of_peaks = np.zeros(
    number_of_conditions,
    dtype=int
)

mean_h = np.full(
    number_of_conditions,
    np.nan
)

minimum_h = np.full(
    number_of_conditions,
    np.nan
)

maximum_h = np.full(
    number_of_conditions,
    np.nan
)

h_amplitude = np.full(
    number_of_conditions,
    np.nan
)

solutions = {}


# ==========================================================
# 13. RUN RESIDENCE-TIME SERIES
# ==========================================================

print()
print("=" * 78)
print("CSTR RESIDENCE-TIME ANALYSIS")
print("=" * 78)

print(
    f"Fixed inlet concentration h_in = "
    f"{h_in:.6f}"
)

print(
    f"Fixed coupling parameter I     = "
    f"{I_fixed:.6f}"
)

print(
    f"beta_surface                   = "
    f"{beta_surface:.8f}"
)

print(
    f"Residence-time mode            = "
    f"{residence_time_mode}"
)

print(
    f"Solver                          = "
    f"{solver_method}"
)

print("=" * 78)
print()


for index, (
    residence_time_min,
    tau_R_star
) in enumerate(
    zip(
        residence_times_min,
        tau_R_star_values
    )
):

    print(
        f"Processing t_R = "
        f"{residence_time_min:.4f} min; "
        f"tau_R* = {tau_R_star:.4f}"
    )

    result = simulate_and_analyse(
        tau_R_star
    )

    valid_result[index] = (
        result["valid"]
    )

    oscillation_score[index] = (
        result["score"]
    )

    oscillation_amplitude[index] = (
        result["amplitude"]
    )

    amplitude_ratio[index] = (
        result["amplitude_ratio"]
    )

    tau_period[index] = (
        result["tau_period"]
    )

    period_min[index] = (
        result["period_min"]
    )

    frequency_dimensionless[index] = (
        result["frequency_dimensionless"]
    )

    frequency_min[index] = (
        result["frequency_min"]
    )

    period_cv[index] = (
        result["period_cv"]
    )

    peak_height_cv[index] = (
        result["peak_height_cv"]
    )

    normalized_slope[index] = (
        result["normalized_max_slope"]
    )

    sinusoidal_error[index] = (
        result["sinusoidal_error"]
    )

    number_of_peaks[index] = (
        result["number_of_peaks"]
    )

    mean_h[index] = (
        result["mean_h"]
    )

    minimum_h[index] = (
        result["minimum_h"]
    )

    maximum_h[index] = (
        result["maximum_h"]
    )

    h_amplitude[index] = (
        result["h_amplitude"]
    )

    if result["solution"] is not None:

        solutions[
            residence_time_min
        ] = result["solution"]

    if result["valid"]:

        print(
            f"  Sustained oscillation: YES"
        )

        print(
            f"  Frequency             = "
            f"{result['frequency_min']:.6f} min^-1"
        )

        print(
            f"  Period                = "
            f"{result['period_min']:.6f} min"
        )

        print(
            f"  Dimensionless period  = "
            f"{result['tau_period']:.6f}"
        )

        print(
            f"  Mean reactor h        = "
            f"{result['mean_h']:.6f}"
        )

        print(
            f"  h range               = "
            f"{result['minimum_h']:.6f} to "
            f"{result['maximum_h']:.6f}"
        )

        print(
            f"  Amplitude             = "
            f"{result['amplitude']:.6f}"
        )

        print(
            f"  Oscillation score     = "
            f"{result['score']:.6f}"
        )

    else:

        print(
            f"  Sustained oscillation: NO"
        )

        print(
            f"  Diagnostic            = "
            f"{result['reason']}"
        )

        if np.isfinite(
            result["frequency_min"]
        ):

            print(
                f"  Detected frequency    = "
                f"{result['frequency_min']:.6f} min^-1"
            )

        if np.isfinite(
            result["mean_h"]
        ):

            print(
                f"  Mean reactor h        = "
                f"{result['mean_h']:.6f}"
            )

    print()


# ==========================================================
# 14. FINAL TABLE
# ==========================================================

print()
print("=" * 186)

print(
    f"{'t_R,min':>10}"
    f"{'tau_R*':>11}"
    f"{'valid':>8}"
    f"{'mean h':>12}"
    f"{'min h':>12}"
    f"{'max h':>12}"
    f"{'h amp':>13}"
    f"{'tau period':>14}"
    f"{'T,min':>12}"
    f"{'f,min^-1':>13}"
    f"{'amplitude':>14}"
    f"{'amp ratio':>12}"
    f"{'score':>12}"
    f"{'peaks':>8}"
)

print("=" * 186)

for index in range(
    number_of_conditions
):

    print(
        f"{residence_times_min[index]:10.4f}"
        f"{tau_R_star_values[index]:11.4f}"
        f"{str(valid_result[index]):>8}"
        f"{mean_h[index]:12.6f}"
        f"{minimum_h[index]:12.6f}"
        f"{maximum_h[index]:12.6f}"
        f"{h_amplitude[index]:13.6e}"
        f"{tau_period[index]:14.6f}"
        f"{period_min[index]:12.6f}"
        f"{frequency_min[index]:13.6f}"
        f"{oscillation_amplitude[index]:14.6f}"
        f"{amplitude_ratio[index]:12.6f}"
        f"{oscillation_score[index]:12.6f}"
        f"{number_of_peaks[index]:8d}"
    )

print("=" * 186)


# ==========================================================
# 15. CONTROL INTERPRETATION
# ==========================================================

print()

if beta_surface == 0.0:

    print(
        "CONTROL RESULT:"
    )

    print(
        "beta_surface = 0, therefore h remains equal to h_in."
    )

    print(
        "Residence time should not change the frequency."
    )

else:

    print(
        "ACTIVE CSTR RESULT:"
    )

    print(
        "beta_surface > 0, therefore residence time can change "
        "the reactor H2O2 concentration and consequently the "
        "oscillation frequency."
    )


# ==========================================================
# 16. PLOT FREQUENCY VERSUS RESIDENCE TIME
# ==========================================================

plot_mask = np.isfinite(
    frequency_min
)

if np.any(plot_mask):

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        residence_times_min[plot_mask],
        frequency_min[plot_mask],
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        r"Residence time, $t_R=V/Q$ (min)",
        fontsize=13
    )

    plt.ylabel(
        r"Oscillation frequency (min$^{-1}$)",
        fontsize=13
    )

    plt.title(
        r"Influence of residence time on oscillation frequency"
    )

    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================================
# 17. PLOT PERIOD VERSUS RESIDENCE TIME
# ==========================================================

plot_mask = np.isfinite(
    period_min
)

if np.any(plot_mask):

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        residence_times_min[plot_mask],
        period_min[plot_mask],
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        r"Residence time, $t_R=V/Q$ (min)",
        fontsize=13
    )

    plt.ylabel(
        r"Oscillation period (min)",
        fontsize=13
    )

    plt.title(
        r"Influence of residence time on oscillation period"
    )

    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================================
# 18. PLOT MEAN REACTOR h VERSUS RESIDENCE TIME
# ==========================================================

plot_mask = np.isfinite(
    mean_h
)

if np.any(plot_mask):

    plt.figure(
        figsize=(9, 6)
    )

    plt.axhline(
        h_in,
        linestyle="--",
        linewidth=2,
        label=rf"Inlet concentration $h_{{in}}={h_in:.2f}$"
    )

    plt.plot(
        residence_times_min[plot_mask],
        mean_h[plot_mask],
        marker="o",
        linewidth=2,
        label=r"Mean reactor concentration $\overline{h}$"
    )

    plt.xlabel(
        r"Residence time, $t_R=V/Q$ (min)",
        fontsize=13
    )

    plt.ylabel(
        r"Dimensionless H$_2$O$_2$ concentration",
        fontsize=13
    )

    plt.title(
        r"Influence of residence time on reactor H$_2$O$_2$ concentration"
    )

    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 19. FREQUENCY VERSUS MEAN REACTOR h
# ==========================================================

plot_mask = (
    np.isfinite(mean_h)
    & np.isfinite(frequency_min)
)

if np.any(plot_mask):

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        mean_h[plot_mask],
        frequency_min[plot_mask],
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        r"Mean reactor concentration, $\overline{h}$",
        fontsize=13
    )

    plt.ylabel(
        r"Oscillation frequency (min$^{-1}$)",
        fontsize=13
    )

    plt.title(
        r"Frequency response to the residence-time-induced change in $h$"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================================
# 20. PLOT LATE-TIME THETA OSCILLATIONS
# ==========================================================

plot_start = 450.0
plot_end = 500.0

if len(solutions) > 0:

    plt.figure(
        figsize=(11, 7)
    )

    for (
        residence_time_min,
        solution
    ) in solutions.items():

        mask = (
            (solution.t >= plot_start)
            & (solution.t <= plot_end)
        )

        plt.plot(
            solution.t[mask],
            solution.y[0, mask],
            linewidth=1.5,
            label=(
                rf"$t_R={residence_time_min:.2f}$ min"
            )
        )

    plt.xlabel(
        r"Dimensionless time, $\tau$",
        fontsize=13
    )

    plt.ylabel(
        r"$\theta_i$",
        fontsize=13
    )

    plt.title(
        r"Late-time oscillations at different residence times"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 21. PLOT LATE-TIME REACTOR h
# ==========================================================

if len(solutions) > 0:

    plt.figure(
        figsize=(11, 7)
    )

    for (
        residence_time_min,
        solution
    ) in solutions.items():

        mask = (
            (solution.t >= plot_start)
            & (solution.t <= plot_end)
        )

        plt.plot(
            solution.t[mask],
            solution.y[4, mask],
            linewidth=1.5,
            label=(
                rf"$t_R={residence_time_min:.2f}$ min"
            )
        )

    plt.axhline(
        h_in,
        linestyle="--",
        linewidth=2,
        label=rf"$h_{{in}}={h_in:.2f}$"
    )

    plt.xlabel(
        r"Dimensionless time, $\tau$",
        fontsize=13
    )

    plt.ylabel(
        r"Reactor concentration, $h(\tau)$",
        fontsize=13
    )

    plt.title(
        r"Late-time H$_2$O$_2$ concentration at different residence times"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 22. SAVE RESULTS
# ==========================================================

output_data = np.column_stack(
    (
        residence_times_min,
        tau_R_star_values,
        valid_result.astype(int),
        np.full(
            number_of_conditions,
            h_in
        ),
        np.full(
            number_of_conditions,
            I_fixed
        ),
        np.full(
            number_of_conditions,
            beta_surface
        ),
        mean_h,
        minimum_h,
        maximum_h,
        h_amplitude,
        tau_period,
        period_min,
        frequency_dimensionless,
        frequency_min,
        oscillation_amplitude,
        amplitude_ratio,
        oscillation_score,
        period_cv,
        peak_height_cv,
        normalized_slope,
        sinusoidal_error,
        number_of_peaks
    )
)

header = (
    "residence_time_min,"
    "dimensionless_residence_time_tau_R_star,"
    "valid_sustained_oscillation,"
    "inlet_h,"
    "fixed_I,"
    "beta_surface,"
    "mean_reactor_h,"
    "minimum_reactor_h,"
    "maximum_reactor_h,"
    "reactor_h_amplitude,"
    "dimensionless_period_tau,"
    "period_min,"
    "dimensionless_frequency,"
    "frequency_min-1,"
    "theta_amplitude,"
    "late_to_previous_amplitude_ratio,"
    "oscillation_quality_score,"
    "period_coefficient_of_variation,"
    "peak_height_coefficient_of_variation,"
    "normalized_maximum_slope,"
    "sinusoidal_shape_error,"
    "number_of_detected_peaks"
)

output_filename = (
    "CSTR_residence_time_frequency_results.csv"
)

np.savetxt(
    output_filename,
    output_data,
    delimiter=",",
    header=header,
    comments=""
)

print()
print(
    f"Results saved to: "
    f"{output_filename}"
)
