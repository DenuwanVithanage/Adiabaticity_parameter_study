"""
adiabatic_manual_window_average.py

Calculate rho, adiabaticity parameter xi, and relative transition factor
for all theta angles and every available r_Li2 value for a selected
Li2-X collision partner and collision energy.

User inputs:
    1. collision partner: He, Ne, Xe
    2. collision energy in cm^-1
    3. lower potential-energy bound for rho fit in cm^-1
    4. upper potential-energy bound for rho fit in cm^-1

Hard-coded:
    OMEGA_CM1 = 255.47 cm^-1

Outputs:
    1. One tab-delimited .tsv file for each r_Li2
    2. One combined .tsv file for all r_Li2 values
    3. rho vs angle plot, one curve per r_Li2
    4. xi vs angle plot, one curve per r_Li2
    5. exp(-xi) vs angle plot, one curve per r_Li2
    6. average-over-r_Li2 table as a function of angle
    7. average rho vs angle plot
    8. average xi vs angle plot
    9. average exp(-xi) vs angle plot

Potential file format expected:
    line 1: header
    line 2: grid dimensions
    lines 3+: r_Li2   R   theta   V(cm-1)

Important:
    This script assumes potential-file distances r_Li2 and R are in bohr.
    Therefore rho is first fitted in bohr, then converted to Angstrom.

Model:
    V(R) = A exp(-R/rho)

Taking log:
    ln(V) = ln(A) - R/rho

Therefore:
    slope = -1/rho
    rho = -1/slope

Adiabaticity:
    xi = 2*pi*c*omega*rho/v

where:
    omega = vibrational frequency in cm^-1
    rho   = range parameter in cm
    v     = relative velocity in cm/s
    c     = speed of light in cm/s

Transition factor:
    P_if = C * exp(-xi)

Since C is unknown, this script calculates:
    P_rel = exp(-xi)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Constants
# ============================================================

FILES = {
    "He": "kpdata_he.pot",
    "Ne": "kpdata_ne.pot",
    "Xe": "kpdata_xe.pot",
}

MASSES = {
    "He": 4.0026,
    "Ne": 20.180,
    "Xe": 131.293,
}

# Li2 mass in amu
M_LI2 = 14.0

# Hard-coded Li2 vibrational frequency in cm^-1
OMEGA_CM1 = 255.47

# Minimum number of points required for rho fit
MIN_POINTS = 3

# Distance conversion
BOHR_TO_ANGSTROM = 0.529177210903

# Physical constants
H = 6.62607015e-34          # J s
C_M_S = 2.99792458e8        # m/s
C_CM_S = 2.99792458e10      # cm/s
AMU_TO_KG = 1.66053906660e-27


# ============================================================
# 2. Load potential file
# ============================================================

def load_pot(fname):
    """
    Load potential file.

    Expected columns after first two header lines:
        r_Li2   R   theta   V

    Returns:
        numpy array with columns [r_Li2, R, theta, V]
    """

    data = []

    with open(fname, "r") as f:
        lines = f.readlines()

    for line in lines[2:]:
        parts = line.strip().split()

        if len(parts) == 4:
            try:
                data.append(list(map(float, parts)))
            except ValueError:
                continue

    data = np.array(data)

    if data.size == 0:
        raise ValueError(f"No numeric potential data found in {fname}")

    return data


# ============================================================
# 3. Reduced mass and velocity
# ============================================================

def reduced_mass(m1_amu, m2_amu):
    """
    Reduced mass in amu.
    """
    return (m1_amu * m2_amu) / (m1_amu + m2_amu)


def velocity_from_collision_energy(E_collision_cm1, mu_amu):
    """
    Calculate relative velocity from collision energy.

    Inputs:
        E_collision_cm1 : collision energy in cm^-1
        mu_amu          : reduced mass in amu

    Returns:
        v_m_s           : velocity in m/s
        v_cm_s          : velocity in cm/s
    """

    # Convert cm^-1 to joules:
    # E = h c wavenumber
    # wavenumber in m^-1 = 100 * cm^-1
    E_joule = H * C_M_S * 100.0 * E_collision_cm1

    mu_kg = mu_amu * AMU_TO_KG

    v_m_s = np.sqrt(2.0 * E_joule / mu_kg)
    v_cm_s = v_m_s * 100.0

    return v_m_s, v_cm_s


# ============================================================
# 4. Fit rho for one angle using manual V bounds
# ============================================================

def fit_rho(R_bohr, V_cm1, V_lower_cm1, V_upper_cm1, min_points=MIN_POINTS):
    """
    Fit ln(V) = mR + c using a manually selected potential-energy window.

    The fitting window is:
        V_lower_cm1 < V(R) < V_upper_cm1

    Since:
        V(R) = A exp(-R/rho)

    then:
        ln(V) = ln(A) - R/rho

    Therefore:
        m = -1/rho
        rho = -1/m
    """

    R_bohr = np.asarray(R_bohr)
    V_cm1 = np.asarray(V_cm1)

    # Keep only positive repulsive-wall values
    mask = V_cm1 > 0.0
    R_bohr = R_bohr[mask]
    V_cm1 = V_cm1[mask]

    # Manual fitting window
    V_low = V_lower_cm1
    V_high = V_upper_cm1

    mask = (V_cm1 > V_low) & (V_cm1 < V_high)
    R_fit = R_bohr[mask]
    V_fit = V_cm1[mask]

    if len(R_fit) < min_points:
        return {
            "rho_bohr": np.nan,
            "rho_angstrom": np.nan,
            "A_cm-1": np.nan,
            "slope_m_bohr-1": np.nan,
            "intercept_c": np.nan,
            "R2": np.nan,
            "N_fit_points": len(R_fit),
            "R_min_fit_bohr": np.nan,
            "R_max_fit_bohr": np.nan,
            "R_min_fit_angstrom": np.nan,
            "R_max_fit_angstrom": np.nan,
            "V_min_fit_cm-1": np.nan,
            "V_max_fit_cm-1": np.nan,
            "V_lower_bound_cm-1": V_low,
            "V_upper_bound_cm-1": V_high,
            "status": "not enough points",
        }

    lnV = np.log(V_fit)

    # Linear fit: ln(V) = mR + c
    m, c = np.polyfit(R_fit, lnV, 1)

    rho_bohr = -1.0 / m
    rho_angstrom = rho_bohr * BOHR_TO_ANGSTROM
    A_cm1 = np.exp(c)

    # R-squared
    lnV_calc = m * R_fit + c
    ss_res = np.sum((lnV - lnV_calc) ** 2)
    ss_tot = np.sum((lnV - np.mean(lnV)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "rho_bohr": rho_bohr,
        "rho_angstrom": rho_angstrom,
        "A_cm-1": A_cm1,
        "slope_m_bohr-1": m,
        "intercept_c": c,
        "R2": R2,
        "N_fit_points": len(R_fit),
        "R_min_fit_bohr": np.min(R_fit),
        "R_max_fit_bohr": np.max(R_fit),
        "R_min_fit_angstrom": np.min(R_fit) * BOHR_TO_ANGSTROM,
        "R_max_fit_angstrom": np.max(R_fit) * BOHR_TO_ANGSTROM,
        "V_min_fit_cm-1": np.min(V_fit),
        "V_max_fit_cm-1": np.max(V_fit),
        "V_lower_bound_cm-1": V_low,
        "V_upper_bound_cm-1": V_high,
        "status": "ok",
    }


# ============================================================
# 5. Adiabaticity and transition factor
# ============================================================

def adiabaticity_lab_units(rho_angstrom, v_cm_s, omega_cm1=OMEGA_CM1):
    """
    Calculate adiabaticity parameter using lab-friendly units.

    xi = 2*pi*c*omega*rho/v

    Inputs:
        rho_angstrom : rho in Angstrom
        v_cm_s       : relative velocity in cm/s
        omega_cm1    : vibrational frequency in cm^-1

    Conversion:
        rho_cm = rho_angstrom * 1e-8

    Returns:
        xi dimensionless
    """

    if np.isnan(rho_angstrom):
        return np.nan

    rho_cm = rho_angstrom * 1.0e-8

    xi = 2.0 * np.pi * C_CM_S * omega_cm1 * rho_cm / v_cm_s

    return xi


def transition_factor(xi):
    """
    Relative transition factor.

    Landau-Teller-type form:
        P_if = C * exp(-xi)

    Since C is unknown, this function returns:
        P_rel = exp(-xi)
    """

    if np.isnan(xi):
        return np.nan

    return np.exp(-xi)


# ============================================================
# 6. Analyze one r_Li2 value
# ============================================================

def analyze_one_r(
    data,
    partner,
    E_collision_cm1,
    r_selected_bohr,
    mu_amu,
    v_m_s,
    v_cm_s,
    V_lower_cm1,
    V_upper_cm1,
):
    """
    For one r_Li2 value, calculate rho, xi, and exp(-xi)
    for all available theta values.
    """

    r_selected_angstrom = r_selected_bohr * BOHR_TO_ANGSTROM

    mask_r = np.isclose(data[:, 0], r_selected_bohr)
    theta_values = np.unique(data[mask_r, 2])

    rows = []

    for theta in theta_values:

        mask = (
            np.isclose(data[:, 0], r_selected_bohr)
            &
            np.isclose(data[:, 2], theta)
        )

        subset = data[mask]

        if len(subset) == 0:
            continue

        subset = subset[np.argsort(subset[:, 1])]

        R_bohr = subset[:, 1]
        V_cm1 = subset[:, 3]

        fit = fit_rho(
            R_bohr=R_bohr,
            V_cm1=V_cm1,
            V_lower_cm1=V_lower_cm1,
            V_upper_cm1=V_upper_cm1,
            min_points=MIN_POINTS,
        )

        xi = adiabaticity_lab_units(
            rho_angstrom=fit["rho_angstrom"],
            v_cm_s=v_cm_s,
            omega_cm1=OMEGA_CM1,
        )

        P_rel = transition_factor(xi)

        row = {
            "partner": partner,
            "E_collision_cm-1": E_collision_cm1,
            "omega_cm-1": OMEGA_CM1,
            "mu_amu": mu_amu,
            "v_m/s": v_m_s,
            "v_cm/s": v_cm_s,
            "r_Li2_used_bohr": r_selected_bohr,
            "r_Li2_used_angstrom": r_selected_angstrom,
            "theta_deg": theta,
            "V_lower_cm-1": V_lower_cm1,
            "V_upper_cm-1": V_upper_cm1,
            "xi": xi,
            "P_rel_exp_minus_xi": P_rel,
            "P_model_symbolic": "C*exp(-xi)",
        }

        row.update(fit)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 7. Detailed plotting: one curve per r_Li2
# ============================================================

def make_combined_plots(combined_df, partner, E_collision_cm1, output_dir):
    """
    Make detailed plots for one collision partner and one collision energy.

    Plot 1:
        rho_angstrom vs theta, one curve per r_Li2

    Plot 2:
        xi vs theta, one curve per r_Li2

    Plot 3:
        exp(-xi) vs theta, one curve per r_Li2
    """

    plot_df = combined_df[combined_df["status"] == "ok"].copy()

    if plot_df.empty:
        print("\nNo valid fitted points available for detailed plotting.")
        return

    plot_df = plot_df.sort_values(
        by=["r_Li2_used_bohr", "theta_deg"]
    )

    unique_r = plot_df["r_Li2_used_bohr"].unique()

    # -----------------------------
    # Plot 1: rho vs theta
    # -----------------------------
    plt.figure(figsize=(10, 6))

    for r in unique_r:
        sub = plot_df[plot_df["r_Li2_used_bohr"] == r]
        plt.plot(
            sub["theta_deg"],
            sub["rho_angstrom"],
            marker="o",
            linewidth=1.5,
            label=f"r = {r:.3f} bohr"
        )

    plt.xlabel("Angle θ (degrees)")
    plt.ylabel("ρ (Angstrom)")
    plt.title(f"{partner}: ρ vs angle at E = {E_collision_cm1:.0f} cm$^{{-1}}$")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    rho_plot_file = os.path.join(
        output_dir,
        f"rho_vs_angle_{partner}_E{E_collision_cm1:.0f}.png"
    )

    plt.savefig(rho_plot_file, dpi=300)
    plt.close()

    # -----------------------------
    # Plot 2: xi vs theta
    # -----------------------------
    plt.figure(figsize=(10, 6))

    for r in unique_r:
        sub = plot_df[plot_df["r_Li2_used_bohr"] == r]
        plt.plot(
            sub["theta_deg"],
            sub["xi"],
            marker="o",
            linewidth=1.5,
            label=f"r = {r:.3f} bohr"
        )

    plt.xlabel("Angle θ (degrees)")
    plt.ylabel("Adiabaticity parameter ξ")
    plt.title(f"{partner}: ξ vs angle at E = {E_collision_cm1:.0f} cm$^{{-1}}$")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    xi_plot_file = os.path.join(
        output_dir,
        f"xi_vs_angle_{partner}_E{E_collision_cm1:.0f}.png"
    )

    plt.savefig(xi_plot_file, dpi=300)
    plt.close()

    # -----------------------------
    # Plot 3: transition factor vs theta
    # -----------------------------
    plt.figure(figsize=(10, 6))

    for r in unique_r:
        sub = plot_df[plot_df["r_Li2_used_bohr"] == r]
        plt.plot(
            sub["theta_deg"],
            sub["P_rel_exp_minus_xi"],
            marker="o",
            linewidth=1.5,
            label=f"r = {r:.3f} bohr"
        )

    plt.xlabel("Angle θ (degrees)")
    plt.ylabel("Relative transition factor exp(-ξ)")
    plt.title(
        f"{partner}: exp(-ξ) vs angle at E = "
        f"{E_collision_cm1:.0f} cm$^{{-1}}$"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    transition_plot_file = os.path.join(
        output_dir,
        f"transition_factor_vs_angle_{partner}_E{E_collision_cm1:.0f}.png"
    )

    plt.savefig(transition_plot_file, dpi=300)
    plt.close()

    print(f"\nSaved detailed plot: {rho_plot_file}")
    print(f"Saved detailed plot: {xi_plot_file}")
    print(f"Saved detailed plot: {transition_plot_file}")


# ============================================================
# 8. Average over r_Li2 at each angle
# ============================================================

def save_average_over_r_by_angle(combined_df, partner, E_collision_cm1, output_dir):
    """
    Average rho, xi, and transition factor over all r_Li2 values
    for each theta angle.

    Output preserves angle dependence:
        theta -> average over r_Li2
    """

    df_ok = combined_df[combined_df["status"] == "ok"].copy()

    if df_ok.empty:
        print("\nNo valid rows available for averaging.")
        return None

    angle_avg = (
        df_ok
        .groupby("theta_deg", as_index=False)
        .agg(
            partner=("partner", "first"),
            E_collision_cm_1=("E_collision_cm-1", "first"),
            omega_cm_1=("omega_cm-1", "first"),

            rho_bohr_avg=("rho_bohr", "mean"),
            rho_bohr_std=("rho_bohr", "std"),

            rho_angstrom_avg=("rho_angstrom", "mean"),
            rho_angstrom_std=("rho_angstrom", "std"),

            xi_avg=("xi", "mean"),
            xi_std=("xi", "std"),

            P_rel_exp_minus_xi_avg=("P_rel_exp_minus_xi", "mean"),
            P_rel_exp_minus_xi_std=("P_rel_exp_minus_xi", "std"),

            N_r_values=("r_Li2_used_bohr", "count"),
        )
    )

    avg_file = os.path.join(
        output_dir,
        f"average_over_r_by_angle_{partner}_E{E_collision_cm1:.0f}.tsv"
    )

    angle_avg.to_csv(avg_file, sep="\t", index=False)

    print(f"\nSaved average-over-r-by-angle file: {avg_file}")

    return angle_avg


# ============================================================
# 9. Average plots
# ============================================================

def plot_average_over_r_by_angle(angle_avg, partner, E_collision_cm1, output_dir):
    """
    Make three averaged plots:
        1. average rho vs theta
        2. average xi vs theta
        3. average exp(-xi) vs theta

    Error bars show the standard deviation over r_Li2 values.
    """

    if angle_avg is None or angle_avg.empty:
        print("\nNo averaged data available for plotting.")
        return

    angle_avg = angle_avg.sort_values("theta_deg")

    # -----------------------------
    # Plot 1: average rho vs theta
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        angle_avg["theta_deg"],
        angle_avg["rho_angstrom_avg"],
        yerr=angle_avg["rho_angstrom_std"],
        marker="o",
        capsize=3,
        linewidth=1.5,
    )
    plt.xlabel("Angle θ (degrees)")
    plt.ylabel("Average ρ over r_Li2 (Angstrom)")
    plt.title(f"{partner}: average ρ vs angle at E = {E_collision_cm1:.0f} cm$^{{-1}}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    rho_avg_plot = os.path.join(
        output_dir,
        f"avg_rho_vs_angle_{partner}_E{E_collision_cm1:.0f}.png"
    )
    plt.savefig(rho_avg_plot, dpi=300)
    plt.close()

    # -----------------------------
    # Plot 2: average xi vs theta
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        angle_avg["theta_deg"],
        angle_avg["xi_avg"],
        yerr=angle_avg["xi_std"],
        marker="o",
        capsize=3,
        linewidth=1.5,
    )
    plt.xlabel("Angle θ (degrees)")
    plt.ylabel("Average ξ over r_Li2")
    plt.title(f"{partner}: average ξ vs angle at E = {E_collision_cm1:.0f} cm$^{{-1}}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    xi_avg_plot = os.path.join(
        output_dir,
        f"avg_xi_vs_angle_{partner}_E{E_collision_cm1:.0f}.png"
    )
    plt.savefig(xi_avg_plot, dpi=300)
    plt.close()

    # -----------------------------
    # Plot 3: average transition factor vs theta
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        angle_avg["theta_deg"],
        angle_avg["P_rel_exp_minus_xi_avg"],
        yerr=angle_avg["P_rel_exp_minus_xi_std"],
        marker="o",
        capsize=3,
        linewidth=1.5,
    )
    plt.xlabel("Angle θ (degrees)")
    plt.ylabel("Average exp(-ξ) over r_Li2")
    plt.title(
        f"{partner}: average exp(-ξ) vs angle at E = "
        f"{E_collision_cm1:.0f} cm$^{{-1}}$"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    transition_avg_plot = os.path.join(
        output_dir,
        f"avg_transition_factor_vs_angle_{partner}_E{E_collision_cm1:.0f}.png"
    )
    plt.savefig(transition_avg_plot, dpi=300)
    plt.close()

    print(f"Saved averaged plot: {rho_avg_plot}")
    print(f"Saved averaged plot: {xi_avg_plot}")
    print(f"Saved averaged plot: {transition_avg_plot}")


# ============================================================
# 10. Main program
# ============================================================

def main():

    print("\nAvailable collision partners:")
    print("He, Ne, Xe")

    partner = input("\nEnter collision partner: ").strip()

    if partner not in FILES:
        raise ValueError("Invalid partner. Use He, Ne, or Xe.")

    E_collision_cm1 = float(
        input("Enter collision energy E_collision in cm^-1: ")
    )

    V_lower_cm1 = float(
        input("Enter lower potential-energy bound for rho fit in cm^-1: ")
    )

    V_upper_cm1 = float(
        input("Enter upper potential-energy bound for rho fit in cm^-1: ")
    )

    if V_upper_cm1 <= V_lower_cm1:
        raise ValueError("Upper bound must be greater than lower bound.")

    fname = FILES[partner]

    print(f"\nReading file: {fname}")
    data = load_pot(fname)

    available_r = np.unique(data[:, 0])
    n_r = len(available_r)

    print(f"\nFound {n_r} available r_Li2 values in file.")

    mu_amu = reduced_mass(M_LI2, MASSES[partner])
    v_m_s, v_cm_s = velocity_from_collision_energy(E_collision_cm1, mu_amu)

    print(f"\nUsing hard-coded vibrational frequency omega = {OMEGA_CM1:.2f} cm^-1")
    print("Using manual fitting window:")
    print(f"    V_lower = {V_lower_cm1:.6f} cm^-1")
    print(f"    V_upper = {V_upper_cm1:.6f} cm^-1")
    print(f"Reduced mass mu = {mu_amu:.8f} amu")
    print(f"Relative velocity v = {v_m_s:.8f} m/s")
    print(f"Relative velocity v = {v_cm_s:.8f} cm/s")

    # Output directory
    output_dir = (
        f"rho_xi_transition_{partner}_"
        f"E{E_collision_cm1:.0f}_"
        f"V{V_lower_cm1:.0f}-{V_upper_cm1:.0f}_all_r"
    )

    os.makedirs(output_dir, exist_ok=True)

    all_clean_tables = []

    for i, r_selected_bohr in enumerate(available_r, start=1):

        r_selected_angstrom = r_selected_bohr * BOHR_TO_ANGSTROM

        df = analyze_one_r(
            data=data,
            partner=partner,
            E_collision_cm1=E_collision_cm1,
            r_selected_bohr=r_selected_bohr,
            mu_amu=mu_amu,
            v_m_s=v_m_s,
            v_cm_s=v_cm_s,
            V_lower_cm1=V_lower_cm1,
            V_upper_cm1=V_upper_cm1,
        )

        # Clean output columns only
        clean_cols = [
            "partner",
            "E_collision_cm-1",
            "omega_cm-1",
            "r_Li2_used_bohr",
            "r_Li2_used_angstrom",
            "theta_deg",
            "rho_bohr",
            "rho_angstrom",
            "xi",
            "P_rel_exp_minus_xi",
            "P_model_symbolic",
            "V_lower_cm-1",
            "V_upper_cm-1",
            "R2",
            "N_fit_points",
            "status",
        ]

        df_clean = df[clean_cols]

        output_file = os.path.join(
            output_dir,
            f"rho_xi_transition_{partner}_"
            f"E{E_collision_cm1:.0f}_"
            f"V{V_lower_cm1:.0f}-{V_upper_cm1:.0f}_"
            f"r{i:02d}_{r_selected_bohr:.6f}bohr.tsv"
        )

        # Tab-delimited output
        df_clean.to_csv(output_file, sep="\t", index=False)

        all_clean_tables.append(df_clean)

        print(
            f"Saved {i:02d}/{n_r}: "
            f"r = {r_selected_bohr:.6f} bohr "
            f"({r_selected_angstrom:.6f} Angstrom) -> {output_file}"
        )

    # Combined file
    combined_df = pd.concat(all_clean_tables, ignore_index=True)

    combined_file = os.path.join(
        output_dir,
        f"rho_xi_transition_{partner}_"
        f"E{E_collision_cm1:.0f}_"
        f"V{V_lower_cm1:.0f}-{V_upper_cm1:.0f}_all_r_combined.tsv"
    )

    combined_df.to_csv(combined_file, sep="\t", index=False)

    # Detailed plots: one curve per r_Li2
    make_combined_plots(
        combined_df=combined_df,
        partner=partner,
        E_collision_cm1=E_collision_cm1,
        output_dir=output_dir,
    )

    # Average over r_Li2 at each angle
    angle_avg = save_average_over_r_by_angle(
        combined_df=combined_df,
        partner=partner,
        E_collision_cm1=E_collision_cm1,
        output_dir=output_dir,
    )

    # Averaged plots
    plot_average_over_r_by_angle(
        angle_avg=angle_avg,
        partner=partner,
        E_collision_cm1=E_collision_cm1,
        output_dir=output_dir,
    )

    print("\nCalculation complete.")
    print(f"Created {n_r} separate r_Li2 files.")
    print(f"Combined file saved to: {combined_file}")

    if angle_avg is not None:
        print("\nPreview of average-over-r-by-angle results:")
        print(angle_avg.to_string(index=False))


if __name__ == "__main__":
    main()
