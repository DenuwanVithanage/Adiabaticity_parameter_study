"""
adiabatic_lab_units.py

Calculate rho and adiabaticity parameter xi for all theta angles
for a selected Li2-X collision partner, collision energy, and r_Li2.

Potential file format expected:
    line 1: header
    line 2: grid dimensions
    lines 3+: r_Li2   R   theta   V(cm-1)

Important:
    This script assumes the potential-file distances r_Li2 and R are in bohr.
    Therefore rho is first fitted in bohr, then converted to Angstrom.

Model:
    V(R) = A exp(-R/rho)

Taking log:
    ln(V) = ln(A) - R/rho

Therefore:
    slope = -1/rho
    rho = -1/slope

Adiabaticity:
    xi = 2*pi*c*omega*rho / v

where:
    omega = vibrational frequency in cm^-1
    rho   = range parameter in cm
    v     = relative velocity in cm/s
    c     = speed of light in cm/s
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. User-adjustable constants
# ============================================================

FILES = {
    "He": "kpdata_he.pot",
    "Ne": "kpdata_ne.pot",
    "Xe": "kpdata_xe.pot",
}

# Atomic masses in amu
MASSES = {
    "He": 4.0026,
    "Ne": 20.180,
    "Xe": 131.293,
}

# Li2 mass in amu
M_LI2 = 14

# Hard-coded Li2 vibrational frequency in cm^-1
OMEGA_CM1 = 255.47

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
    Loads potential file.

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

    Uses:
        E = h c nu
        with nu in m^-1.

        Since E_collision is in cm^-1:
            nu_m^-1 = 100 * E_collision_cm^-1
    """

    E_joule = H * C_M_S * 100.0 * E_collision_cm1
    mu_kg = mu_amu * AMU_TO_KG

    v_m_s = np.sqrt(2.0 * E_joule / mu_kg)
    v_cm_s = v_m_s * 100.0

    return v_m_s, v_cm_s


# ============================================================
# 4. Fit rho for one angle
# ============================================================

def fit_rho(R_bohr, V_cm1, E_collision_cm1, window_factor=5.0, min_points=3):
    """
    Fit ln(V) = mR + c near the collision energy.

    The fitting window is:
        E_collision/window_factor < V < E_collision*window_factor

    Since:
        V(R) = A exp(-R/rho)

    then:
        ln(V) = ln(A) - R/rho

    Therefore:
        m = -1/rho
        rho = -1/m

    Inputs:
        R_bohr          : R values in bohr
        V_cm1           : potential values in cm^-1
        E_collision_cm1 : collision energy in cm^-1
        window_factor   : fitting window factor
        min_points      : minimum points needed for linear fit

    Returns:
        dictionary containing rho, fit data, and status
    """

    R_bohr = np.asarray(R_bohr)
    V_cm1 = np.asarray(V_cm1)

    # Keep only positive repulsive-wall values
    mask = V_cm1 > 0.0
    R_bohr = R_bohr[mask]
    V_cm1 = V_cm1[mask]

    # Select local energy window
    V_low = E_collision_cm1 / window_factor
    V_high = E_collision_cm1 * window_factor

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
            "V_low_window_cm-1": V_low,
            "V_high_window_cm-1": V_high,
            "status": "not enough points",
        }

    lnV = np.log(V_fit)

    # Linear fit: ln(V) = mR + c
    m, c = np.polyfit(R_fit, lnV, 1)

    # rho in bohr
    rho_bohr = -1.0 / m

    # rho in Angstrom
    rho_angstrom = rho_bohr * BOHR_TO_ANGSTROM

    # A in cm^-1
    A_cm1 = np.exp(c)

    # Fit quality
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
        "V_low_window_cm-1": V_low,
        "V_high_window_cm-1": V_high,
        "status": "ok",
    }


# ============================================================
# 5. Adiabaticity in lab units
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


# ============================================================
# 6. Main program
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

    r_input_bohr = float(
        input("Enter r_Li2 in bohr, for example 4.82: ")
    )

    window_input = input(
        "Enter window factor for fitting [default = 5.0]: "
    ).strip()

    if window_input == "":
        window_factor = 5.0
    else:
        window_factor = float(window_input)

    fname = FILES[partner]

    print(f"\nReading file: {fname}")
    data = load_pot(fname)

    # Find closest available r_Li2 in potential file
    available_r = np.unique(data[:, 0])
    r_selected_bohr = available_r[np.argmin(np.abs(available_r - r_input_bohr))]
    r_selected_angstrom = r_selected_bohr * BOHR_TO_ANGSTROM

    print(f"\nRequested r_Li2 = {r_input_bohr:.6f} bohr")
    print(f"Using closest available r_Li2 = {r_selected_bohr:.6f} bohr")
    print(f"Using closest available r_Li2 = {r_selected_angstrom:.6f} Angstrom")
    print(f"Using hard-coded vibrational frequency omega = {OMEGA_CM1:.2f} cm^-1")
    print(f"Using fitting window factor = {window_factor:.3f}")

    # Reduced mass and velocity
    mu_amu = reduced_mass(M_LI2, MASSES[partner])
    v_m_s, v_cm_s = velocity_from_collision_energy(E_collision_cm1, mu_amu)

    print(f"\nReduced mass mu = {mu_amu:.8f} amu")
    print(f"Relative velocity v = {v_m_s:.8f} m/s")
    print(f"Relative velocity v = {v_cm_s:.8f} cm/s")

    # Get all theta values for selected r
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

        # Sort by R
        subset = subset[np.argsort(subset[:, 1])]

        R_bohr = subset[:, 1]
        V_cm1 = subset[:, 3]

        fit = fit_rho(
            R_bohr=R_bohr,
            V_cm1=V_cm1,
            E_collision_cm1=E_collision_cm1,
            window_factor=window_factor,
            min_points=3,
        )

        xi = adiabaticity_lab_units(
            rho_angstrom=fit["rho_angstrom"],
            v_cm_s=v_cm_s,
            omega_cm1=OMEGA_CM1,
        )

        row = {
            "partner": partner,
            "E_collision_cm-1": E_collision_cm1,
            "omega_cm-1": OMEGA_CM1,
            "mass_Li2_amu": M_LI2,
            "mass_partner_amu": MASSES[partner],
            "mu_amu": mu_amu,
            "v_m/s": v_m_s,
            "v_cm/s": v_cm_s,
            "r_Li2_input_bohr": r_input_bohr,
            "r_Li2_used_bohr": r_selected_bohr,
            "r_Li2_used_angstrom": r_selected_angstrom,
            "theta_deg": theta,
            "window_factor": window_factor,
            "xi": xi,
        }

        row.update(fit)
        rows.append(row)

    df = pd.DataFrame(rows)

    output_file = (
        f"rho_xi_lab_{partner}_"
        f"E{E_collision_cm1:.0f}_"
        f"r{r_selected_bohr:.3f}bohr.csv"
    )

    df.to_csv(output_file, index=False)

    print("\nCalculation complete.")
    print(f"Output saved to: {output_file}")

    print("\nPreview:")
    preview_cols = [
        "partner",
        "E_collision_cm-1",
        "r_Li2_used_bohr",
        "r_Li2_used_angstrom",
        "theta_deg",
        "rho_bohr",
        "rho_angstrom",
        "v_cm/s",
        "xi",
        "R2",
        "N_fit_points",
        "status",
    ]

    print(df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
