import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from astropy import constants as const



Ms = const.M_sun.value
Ls = const.L_sun.value
Rs = const.R_sun.value
G = const.G.value
h = const.h.value
c = const.c.value
k_B = const.k_B.value



def spheroid(Re_eq, Rp_polar, n):
    i = np.arange(n)
    g_ang = np.pi*(3 - np.sqrt(5))  ##### uniform surface points from the Fibonacci sequence (golden angle)
    lat = np.arccos(1 - 2*(i+0.5)/n)
    lon = i*g_ang
    r = Re_eq*np.sqrt(1/(np.sin(lat)**2 + ((Re_eq/Rp_polar)*np.cos(lat))**2))
    x = r*np.sin(lat)*np.cos(lon)
    y = r*np.sin(lat)*np.sin(lon)
    z = r*np.cos(lat)
    return x, y, z, lat



def rotation_matrix_x(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle),  np.cos(angle)]])



def rotation_matrix_y(angle):
    return np.array([[ np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])



def rotation_matrix_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle),  np.cos(angle), 0],
                    [0, 0, 1]])



def rotated_spheroid(x, y, z, lamda, i_s):
    p = np.array([x.flatten(), y.flatten(), z.flatten()])
    R = rotation_matrix_x(-lamda) @ rotation_matrix_y(-i_s + np.pi/2)
    rot_p = R @ p  ######## rotating the spheroid (points) to match the projected view on the plane of the celestial sphere
    x_rot, y_rot, z_rot = rot_p[0], rot_p[1], rot_p[2]
    return x_rot, y_rot, z_rot, R



def vis_mask(x, y, z, Re_eq, Rp_polar, R):
    Nx = x/Re_eq**2
    Ny = y/Re_eq**2
    Nz = z/Rp_polar**2  ##### x,y,z normals
    norm = np.array([Nx, Ny, Nz])
    rot_norm = R @ norm #### rotate the normals
    mask = rot_norm[0] > 0  #### only the points with positive norm elements should be visible
    norm_mag = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    mu = rot_norm[0]/norm_mag
    return mask, mu



def ellipse_radius(theta, Re_eq, Rp_polar, i_s, lamda):  ### analytical expression derived from the Schur complement
    sini = np.sin(i_s)
    cosi = np.cos(i_s)
    D = (Re_eq*cosi)**2 + (Rp_polar*sini)**2
    return Re_eq*np.sqrt(D/(D + (Re_eq**2 - Rp_polar**2)*(sini*np.sin(theta + lamda))**2))



def gravity_darkening(st_mass, st_mean_temperature, beta, omega, obs_wavelength, Re_eq, Rp_polar, theta):
    r = Re_eq*np.sqrt(1/(np.sin(theta)**2 + ((Re_eq/Rp_polar)*np.cos(theta))**2))
    Omega_Kepl = np.sqrt(G*st_mass/(Re_eq**3))
    g = np.sqrt((-G*st_mass/(r**2) + r*((omega*Omega_Kepl)**2)*(np.sin(theta)**2))**2 + (r*((omega*Omega_Kepl)**2)*np.sin(theta)*np.cos(theta))**2)
    g_pole = np.abs(-G*st_mass/(Rp_polar**2))
    integral = np.mean((g/g_pole)**(4*beta))  ### deriving the polar temperature from the mean temperature
    T_pole = st_mean_temperature*(integral**(-0.25))
    temperature = T_pole*((g/g_pole)**beta)  ##### using the von Zeipel theorem for temperaturte variation
    I_planck = (2*h*c**2)/(obs_wavelength**5)/(np.exp(h*c/(obs_wavelength*k_B*temperature)) - 1)
    return I_planck



def limb_darkening(mu, u1, u2):
    mu = np.clip(mu, 0, 1)
    return 1.0 - u1*(1 - mu) - u2*(1 - mu)**2



def unocculted_flux(I_total, points):
    hull = ConvexHull(points)
    proj_area = hull.volume  ##### projected area of the visible stellar disk
    dA = proj_area/len(I_total)  ### the area element for integration
    return np.sum(I_total)*dA



def occulted_flux(Y_proj, Z_proj, Rocc_phys, I_total, R_eq, R_polar, i_s, lamda, tree, dA):
    if Rocc_phys <= 0:
        return 0.0
    
    proj_theta = np.arctan2(Z_proj, Y_proj)
    r_proj_theta = ellipse_radius(proj_theta, R_eq, R_polar, np.deg2rad(i_s), np.deg2rad(lamda))
    on_star = (Y_proj**2 + Z_proj**2) <= r_proj_theta**2
    Y_final, Z_final = Y_proj[on_star], Z_proj[on_star]
    
    if len(Y_final) == 0:
        return 0.0
    
    _, iss = tree.query(np.column_stack([Y_final, Z_final]), k=1)  ###### index of the nearest stellar point for each occulted grid point
    return np.sum(I_total[iss])*dA



def kep2car(t, a, e, i_0, raan, omega_p, t_mid, orb_period):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    inc_r = np.radians(i_0)
    Om_r = np.radians(raan)
    om_r = np.radians(omega_p)
    n = 2*np.pi/orb_period

    f_c = np.pi/2 - om_r  ### true anomaly at inferior conjunction, u = om_r + f_c = pi/2
    E_c = 2*np.arctan2(np.sqrt(1 - e)*np.sin(f_c/2), np.sqrt(1 + e)*np.cos(f_c/2))
    M_c = E_c - e*np.sin(E_c)

    M = n*(t - t_mid) + M_c
    M = np.mod(M, 2*np.pi)

    E = M.copy()
    for _ in range(50):
        dE = (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        E -= dE
        if np.all(np.abs(dE) < 1e-8):
            break

    f = 2.0*np.arctan2(np.sqrt(1 + e)*np.sin(E/2), np.sqrt(1 - e)*np.cos(E/2))
    r = a*(1 - e*np.cos(E))

    u = om_r + f
    cosu, sinu = np.cos(u), np.sin(u)
    cosi, sini = np.cos(inc_r), np.sin(inc_r)
    cosO, sinO = np.cos(Om_r), np.sin(Om_r)

    x = r*sinu*sini
    y = -r*(cosO*cosu - sinO*sinu*cosi)
    z = -r*(sinO*cosu + cosO*sinu*cosi)

    if x.size == 1:
        return x[0], y[0], z[0]
    return x, y, z



def vsini2omega(vsini_kms, st_mass_solar, R_mean_solar, i_s_deg):
    st_mass_solar, R_mean_solar = st_mass_solar*Ms, R_mean_solar*Rs
    
    i_s = np.deg2rad(i_s_deg)
    if np.abs(np.sin(i_s)) < 1e-9:
        return np.inf
    omega = 0.3  ### guess
    for _ in range(100):
        R_eq = R_mean_solar*(((2 + omega**2)/2)**(1/3))
        v_eq = (vsini_kms*1e3)/np.sin(i_s)
        Omega = v_eq/R_eq
        Omega_crit = np.sqrt(G*st_mass_solar/(R_eq**3))
        omega_new = Omega/Omega_crit
        if np.abs(omega_new - omega) < 1e-9:
            break
        omega = omega_new
    return omega_new



def grav_dark_transit_model(t_vals, orbital_period, st_mass, st_mean_radius, st_mean_temperature, 
                             beta, lamda, i_s, omega, u1, u2, e, i_0, omega_p, raan, t_mid, 
                             rp_rs, obs_wavelength=800e-9, integration_grid_size=1):
    st_mass, st_mean_radius, st_mean_temperature = st_mass*Ms, st_mean_radius*Rs, st_mean_temperature*10000
    R_eq = st_mean_radius*((2 + omega**2)/2)**(1/3)  ##### for the volume-preserving stellar mean radius
    R_polar = R_eq*(2/(2 + omega**2))
    x, y, z, lat = spheroid(R_eq, R_polar, 2500)  ##### constructing the stellar surface
    _, y_rot, z_rot, R = rotated_spheroid(x, y, z, np.deg2rad(lamda), np.deg2rad(i_s))
    mask, mu_all = vis_mask(x, y, z, R_eq, R_polar, R)
    y_vis = y_rot[mask]
    z_vis = z_rot[mask]
    points = np.column_stack([y_vis, z_vis])
    mu_vis = mu_all[mask]
    
    
    I_grav = gravity_darkening(st_mass, st_mean_temperature, beta, omega, obs_wavelength, R_eq, R_polar, lat)
    I_limb = limb_darkening(mu_vis, u1, u2)
    I_total = I_grav[mask]*I_limb    ###### computing the full intensity profile
    
    
    F_out = unocculted_flux(I_total, points)
    tree = cKDTree(points)
    Rp_phys = rp_rs*st_mean_radius  ###### again for the volume-preserving mean radius
    
    
    integration_grid_size = 250*integration_grid_size
    i = np.arange(integration_grid_size)
    g_rat = (1 + np.sqrt(5))/2
    r = Rp_phys*np.sqrt((i + 0.5)/integration_grid_size)
    theta = 2*np.pi*i/(g_rat**2)  ###### making the uniform planet disk points using the Fibonacci sequence
    planet_dy = r*np.cos(theta)
    planet_dz = r*np.sin(theta)
    dA = (np.pi*Rp_phys**2)/integration_grid_size
    
    a = (G*st_mass*((orbital_period*86400)**2)/(4*np.pi**2))**(1/3)
    x_p, y_p, z_p = kep2car(t_vals, a, e, i_0, raan, omega_p, t_mid, orbital_period)
    d_proj = np.sqrt((y_p - 0)**2 + (z_p - 0)**2)
    position_mask = (x_p >= 0) & (d_proj <= (R_eq + 1.001*Rp_phys))
    
    flux = np.ones_like(t_vals, dtype=float)
    js = np.where(position_mask)[0]
    
    for j in js:
        DF = occulted_flux(y_p[j] + planet_dy, z_p[j] + planet_dz, Rp_phys, I_total, R_eq, R_polar, i_s, lamda, tree, dA)
        flux[j] = (F_out - DF)/F_out
    return np.array(flux)

