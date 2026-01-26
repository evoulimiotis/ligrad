import numpy as np

G_SI    = 6.67430e-11
M_sun   = 1.98847e30
R_sun   = 6.957e8
AU_m    = 1.495978707e11
M_jup   = 1.89813e27
DAY_s   = 86400.0
RAD2DEG = 180.0/np.pi


def precession_rates(Ms, Rs, a_Rs, Mp, psi_deg, omega, e=0.0):
    M_star = Ms*M_sun
    R_star = Rs*R_sun
    a_m    = a_Rs*R_star
    M_p    = Mp*M_jup
    psi_rad = np.deg2rad(psi_deg)
    cospsi = np.cos(psi_rad)
    n = np.sqrt(G_SI*M_star/(a_m**3))
    f = 1 - (2/(2 + omega**2))

    Omega_real = omega*np.sqrt(G_SI*M_star/(R_star**3))
    J2 = -(2*f/3) + ((R_star**3)*(Omega_real**2)/(3*G_SI*M_star))
    
    denom_node = (1.0 - e**2)**2
    omega_L_rad_s = abs(-1.5*n*J2*((R_star / a_m)**2)*cospsi/denom_node) ###### Ω
    C = J2/f
    omega_small_L_rad_s = abs(-1.5*n*J2*((R_star / a_m)**2)*(2 - 2.5*np.sin(psi_rad)**2)/denom_node)  ####### ω
    denom_spin = (1.0 - e**2)**1.5
    omega_ps_rad_s = abs(-(3.0*G_SI*M_p*J2*cospsi)/(2.0*C*Omega_real*(a_m**3)*denom_spin))  ##### spin
    omega_L_deg_day  = omega_L_rad_s*RAD2DEG*DAY_s
    omega_small_L_deg_day = omega_small_L_rad_s*RAD2DEG*DAY_s
    omega_ps_deg_day = omega_ps_rad_s*RAD2DEG*DAY_s
    
    nodal_period = 2*np.pi*(1/(omega_L_rad_s))
    apsidal_period = 2*np.pi*(1/omega_small_L_rad_s)
    spin_period = 2*np.pi*(1/(omega_ps_rad_s))
    print(omega_L_deg_day, omega_small_L_deg_day, omega_ps_deg_day, "degrees/day")
    print("\n", nodal_period/31556926, apsidal_period/31556926, spin_period/31556926, "yrs")
    return None


Ms_list = np.array([2.11412, 1.822, 2.05, 1.795, 1.958, 2.122, 1.65])
Rs_list = np.array([2.2484, 1.792, 1.94, 1.6, 2.121, 1.753, 1.765])
a_Rs_list = np.array([3.2, 5.97, 5.31, 7.5, 4.258, 7.13, 8.0])
Mp_list = np.array([2.6, 2.0, 6.7, 3.3, 3.7, 1.41, 0.7])
psi_list = np.array([88.08, 66.61, 98.16, 5.3, 71.81, 100.03, 37.45])
omega_list = np.array([0.36245, 0.29484, 0.45977, 0.34525, 0.34102, 0.28378, 0.27714])

precession_rates(Ms_list, Rs_list, a_Rs_list, Mp_list, psi_list, omega_list)