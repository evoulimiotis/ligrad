import numpy as np
import pandas as pd
from ligrad import vsini2omega
from astropy import constants as const


M_sun = const.M_sun.value
M_jup = const.M_jup.value
R_sun = const.R_sun.value
G_SI = const.G.value
AU_m = const.au.value
DAY_s = 86400.0
YR_s = 31556926
RAD2DEG = 180.0/np.pi



df = pd.read_csv('stellar_tables_claret_2017.txt', sep='\s+', comment='#', skiprows=48, 
    names=['Mi', 'Z', 'fov', 'Age (yr)', 'Mass (Msun)', 'log(L)', 'log(Teff)', 'log(g)', 'logk2', 'logk3', 'logk4', 'apot', 'beta'])



def best_logk2(df, queries):
    chars = ['Age (yr)', 'Mass (Msun)', 'log(L)', 'log(Teff)', 'log(g)']
    queries = np.asarray(queries, dtype=float)

    if queries.ndim == 1:
        queries = queries.reshape(1, -1)

    df_chars = df[chars]
    means = df_chars.mean().to_numpy()
    stds = df_chars.std().replace(0, 1).to_numpy()
    table = (df_chars.to_numpy() - means)/stds
    queries_norm = (queries - means)/stds
    d = np.linalg.norm(queries_norm[:, None, :] - table[None, :, :], axis=2)
    i = np.argmin(d, axis=1)
    return df.iloc[i]['logk2'].to_numpy()



def precession_rates(Ms, Rs, P, Mp, psi_deg, vsini, i_s, k2, e=0.0):
    omega = vsini2omega(vsini, Ms, Rs, i_s)
    M_star = Ms*M_sun
    R_star = Rs*R_sun
    a_m = (G_SI*M_star*((P*86400)**2)/(4*np.pi**2))**(1/3)
    M_p = Mp*M_jup
    psi_rad = np.deg2rad(psi_deg)
    cospsi = np.cos(psi_rad)
    n = np.sqrt(G_SI*M_star/(a_m**3))
    f = 1 - (2/(2 + omega**2))

    Omega_real = omega*np.sqrt(G_SI*M_star/(R_star**3))
    J2 = (k2/3)*((R_star**3)*(Omega_real**2)/(G_SI*M_star))
    
    den = (1.0 - e**2)**2
    omega_L_rad_s = abs(-1.5*n*J2*((R_star/a_m)**2)*cospsi/den) ###### nodal precession
    C = J2/f
    omega_small_L_rad_s = abs(-1.5*n*J2*((R_star/a_m)**2)*(2 - 2.5*np.sin(psi_rad)**2)/den)  ####### periastron precession
    den_spin = (1.0 - e**2)**1.5
    omega_ps_rad_s = abs(-(3.0*G_SI*M_p*J2*cospsi)/(2.0*C*n*(a_m**3)*den_spin))  ##### spin precession
    omega_L_deg_day = omega_L_rad_s*RAD2DEG*DAY_s
    omega_small_L_deg_day = omega_small_L_rad_s*RAD2DEG*DAY_s
    omega_ps_deg_day = omega_ps_rad_s*RAD2DEG*DAY_s
    
    nodal_period = 2*np.pi*(1/(omega_L_rad_s))
    apsidal_period = 2*np.pi*(1/omega_small_L_rad_s)
    spin_period = 2*np.pi*(1/(omega_ps_rad_s))
    print("Nodal precession:", omega_L_deg_day, "deg/day")
    print("Periastron precession:", omega_small_L_deg_day, "deg/day")
    print("Spin precession", omega_ps_deg_day, "deg/day")
    print("\nPeriods:\n", nodal_period/31556926, "yrs\n", apsidal_period/31556926, "yrs\n", spin_period/31556926, "yrs")
    return None



def vsini2omega_v(vsini_kms, st_mass_solar, R_mean_solar, i_s_deg):
    st_mass_kg = np.asarray(st_mass_solar, dtype=float)*M_sun
    R_mean_m = np.asarray(R_mean_solar, dtype=float)*R_sun
    vsini_ms = np.asarray(vsini_kms, dtype=float)*1e3
    i_s = np.deg2rad(np.asarray(i_s_deg, dtype=float))
    sin_is = np.sin(i_s)
    omega = np.full(np.broadcast(vsini_ms, st_mass_kg, R_mean_m, sin_is).shape, 0.3, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(500):
            R_eq = R_mean_m*(((2 + omega**2)/2)**(1/3))
            v_eq = vsini_ms/sin_is
            Omega = v_eq/R_eq
            Omega_crit = np.sqrt(G_SI*st_mass_kg/(R_eq**3))
            omega = Omega/Omega_crit

    omega = np.where(np.abs(sin_is) < 1e-9, np.inf, omega)
    return omega


def split_normal_sample(mu, sigma_minus, sigma_plus, size):
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    sigma_minus = np.atleast_1d(np.asarray(sigma_minus, dtype=float))
    sigma_plus = np.atleast_1d(np.asarray(sigma_plus, dtype=float))
    n_targets = mu.shape[0]
    samples = np.empty((size, n_targets))
    for i in range(n_targets):
        p_left = sigma_minus[i]/(sigma_minus[i] + sigma_plus[i])
        u = np.random.random(size)
        is_left = u < p_left
        s = np.empty(size)
        s[is_left] = mu[i] - np.abs(np.random.normal(0.0, sigma_minus[i], size=is_left.sum()))
        s[~is_left] = mu[i] + np.abs(np.random.normal(0.0, sigma_plus[i], size=(~is_left).sum()))
        samples[:, i] = s
    return samples


def _summ(samples_2d):
    lo, med, hi = np.percentile(samples_2d, [16, 50, 84], axis=0)
    return med, med - lo, hi - med


def precession_rates_err(Ms, Ms_err_minus, Ms_err_plus, Rs, Rs_err_minus, Rs_err_plus, P, P_err_minus, P_err_plus, Mp,
                                 psi_deg, psi_err_minus, psi_err_plus, vsini, vsini_err_minus, vsini_err_plus, i_s, i_s_err_minus,
                                 i_s_err_plus, k2, e=0.0, n_samples=25000, labels=None):
    Ms = np.atleast_1d(np.asarray(Ms, dtype=float))
    n_targets = Ms.shape[0]
    if labels is None:
        labels = [f"target_{i}" for i in range(n_targets)]

    def sample(val, err_minus, err_plus):
        return split_normal_sample(val, err_minus, err_plus, n_samples)

    Ms_s = sample(Ms, Ms_err_minus, Ms_err_plus)
    Rs_s = sample(Rs, Rs_err_minus, Rs_err_plus)
    P_s = sample(P, P_err_minus, P_err_plus)
    psi_s = sample(psi_deg, psi_err_minus, psi_err_plus)
    vsini_s = sample(vsini, vsini_err_minus, vsini_err_plus)
    is_s = sample(i_s, i_s_err_minus, i_s_err_plus)
    Mp_arr = np.broadcast_to(np.asarray(Mp, dtype=float), (n_samples, n_targets))
    k2_arr = np.broadcast_to(np.asarray(k2, dtype=float), (n_samples, n_targets))
    omega_s = vsini2omega_v(vsini_s, Ms_s, Rs_s, is_s)
    M_star = Ms_s*M_sun
    R_star = Rs_s*R_sun
    M_p = Mp_arr*M_jup
    a_m = (G_SI*(M_star + M_p)*((P_s*DAY_s)**2)/(4*np.pi**2))**(1/3)
    psi_rad = np.deg2rad(psi_s)
    cospsi = np.cos(psi_rad)
    n_mm = np.sqrt(G_SI*M_star/(a_m**3))
    f = 1 - (2/(2 + omega_s**2))
    
    Omega_real = omega_s*np.sqrt(G_SI*M_star/(R_star**3))
    J2 = (2*k2_arr/3)*((R_star**3)*(Omega_real**2)/(G_SI*M_star))
    den = (1.0 - e**2)**2
    omega_L_rad_s = np.abs(-1.5*n_mm*J2*((R_star/a_m)**2)*cospsi/den) # nodal precession
    C = J2/f
    omega_small_L_rad_s = np.abs(-1.5*n_mm*J2*((R_star/a_m)**2)*(2 - 2.5*np.sin(psi_rad)**2)/den)# periastron precession
    den_spin = (1.0 - e**2)**1.5
    omega_ps_rad_s = np.abs(-(3.0*G_SI*M_p*J2*cospsi)/(2.0*C*n_mm*(a_m**3)*den_spin))# spin precession

    omega_L_deg_day = omega_L_rad_s*RAD2DEG*DAY_s
    omega_small_L_deg_day = omega_small_L_rad_s*RAD2DEG*DAY_s
    omega_ps_deg_day = omega_ps_rad_s*RAD2DEG*DAY_s
    nodal_period_yr = (2*np.pi/omega_L_rad_s)/YR_s
    apsidal_period_yr = (2*np.pi/omega_small_L_rad_s)/YR_s
    spin_period_yr = (2*np.pi/omega_ps_rad_s)/YR_s

    quantities = {"omega_L_deg_day": omega_L_deg_day, "omega_small_L_deg_day": omega_small_L_deg_day, "omega_ps_deg_day": omega_ps_deg_day,
                  "nodal_period_yr": nodal_period_yr, "apsidal_period_yr": apsidal_period_yr, "spin_period_yr": spin_period_yr}

    results = {name: _summ(arr) for name, arr in quantities.items()}
    print("Nodal precession rate, periastron precession rate, spin precession rate (deg/yr)")
    for i, label in enumerate(labels):
        print(f"--- {label} ---")
        for name in ["omega_L_deg_day", "omega_small_L_deg_day", "omega_ps_deg_day"]:
            med, sm, sp = results[name]
            print(f"  {name:24s} = {365.25*med[i]:.5f} (+{365.25*sp[i]:.5f}/-{365.25*sm[i]:.5f}) deg/yr")
        for name in ["nodal_period_yr", "apsidal_period_yr", "spin_period_yr"]:
            med, sm, sp = results[name]
            print(f"  {name:24s} = {med[i]:.5f} (+{sp[i]:.5f}/-{sm[i]:.5f}) yr")
        print()
    return results



queries = np.array([
    [0.45e9, 2.32, 1.59, 4.007, 4.037],# KELT-9
    [0.20e9, 1.89, 1.10, 3.953, 4.31],# KELT-20
    [0.80e9, 1.90, 1.08, 3.878, 4.09],# MASCARA-1
    [0.80e9, 1.75, 1.087, 3.892, 4.10],# MASCARA-4
    [0.60e9, 1.89, 1.22, 3.927, 4.181],# HAT-P-70
    [0.46e9, 2.18, 1.338, 3.918, 4.064],# KELT-25
    [0.43e9, 1.93, 1.21, 3.971, 4.211],# WASP-178
    [2.17e9, 1.64, 0.63, 3.811, 4.197],# NGTS-2
])

logk2 = best_logk2(df, queries)
print("best logk2:", logk2)
k2 = 10**logk2
print("best k2:", k2)


labels = ["KELT-9b", "KELT-20b", "MASCARA-1b", "MASCARA-4b", "HAT-P-70b", "KELT-25b", "WASP-178b", "NGTS-2b"]
Ms_val = np.array([2.463, 1.735, 1.895, 1.761, 1.888, 2.148, 2.075, 1.590])
Ms_err_plus = np.array([0.145, 0.141, 0.065, 0.048, 0.011, 0.109, 0.097, 0.146])
Ms_err_minus = np.array([0.157, 0.159, 0.066, 0.046, 0.012, 0.103, 0.107, 0.143])
Rs_val = np.array([2.423, 1.546, 2.049, 1.786, 1.874, 2.283, 1.733, 1.710])
Rs_err_plus = np.array([0.049, 0.042, 0.045, 0.031, 0.153, 0.043, 0.034, 0.040])
Rs_err_minus = np.array([0.062, 0.050, 0.042, 0.024, 0.138, 0.045, 0.034, 0.040])
P_val = np.array([1.4811235, 3.4741085, 2.1487738, 2.824062, 2.7443246, 4.401126, 3.3448413, 4.511162])
P_err_plus = np.array([0.0000011, 0.0000019, 0.0000009, 0.000025, 0.0000007, 0.000045, 0.0000033, 0.000055])
P_err_minus = np.array([0.0000011, 0.0000019, 0.0000009, 0.000024, 0.0000007, 0.000053, 0.0000033, 0.000055])
vsini_val = np.array([111.30, 116.92, 101.46, 46.47, 99.92, 114.20, 12.25, 15.20])
vsini_err_plus = np.array([1.01, 2.20, 3.04, 0.95, 0.62, 1.22, 0.79, 0.79])
vsini_err_minus = np.array([1.10, 2.67, 3.10, 0.98, 0.60, 1.22, 0.79, 0.80])
is_val = np.array([35.7, 99.5, 44.5, 164.8, 34.0, 75.4, 91.6, 91.9])
is_err_plus = np.array([6.9, 26.6, 9.5, 2.8, 5.7, 34.5, 58.2, 54.6])
is_err_minus = np.array([6.2, 25.6, 9.7, 3.9, 3.6, 25.1, 56.3, 55.9])
psi_val = np.array([89.89, 15.05, 79.77, 100.79, 97.83, 20.81, 91.88, 15.89])
psi_err_plus = np.array([13.41, 23.80, 18.25, 4.99, 8.25, 19.21, 20.58, 17.86])
psi_err_minus = np.array([10.21, 22.90, 15.29, 4.59, 5.35, 14.03, 20.92, 17.92])
k2 = np.array(k2)
Mp = np.array([2.88, 2.0, 3.7, 1.68, 4.0, 3.0, 1.66, 0.74])



results = precession_rates_err(Ms=Ms_val, Ms_err_minus=Ms_err_minus, Ms_err_plus=Ms_err_plus,
                               Rs=Rs_val, Rs_err_minus=Rs_err_minus, Rs_err_plus=Rs_err_plus,
                               P=P_val, P_err_minus=P_err_minus, P_err_plus=P_err_plus, Mp=Mp,
                               psi_deg=psi_val, psi_err_minus=psi_err_minus, psi_err_plus=psi_err_plus,
                               vsini=vsini_val, vsini_err_minus=vsini_err_minus, vsini_err_plus=vsini_err_plus,
                               i_s=is_val, i_s_err_minus=is_err_minus, i_s_err_plus=is_err_plus,
                               k2=k2, labels=labels)