import numpy as np
from numpy.typing import NDArray
from astropy import constants as const


Ms = const.M_sun.value
Rs = const.R_sun.value
G = const.G.value
h = const.h.value
c = const.c.value
k_B = const.k_B.value
_GOLDEN_ANGLE = np.pi*(3.0 - np.sqrt(5.0))



def spheroid(Re_eq, Rp_polar, n):
    i = np.arange(n)
    lat = np.arccos(1 - 2*(i+0.5)/n)  ##### uniform surface points from the Fibonacci sequence
    lon = i*_GOLDEN_ANGLE
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
    R = np.matmul(rotation_matrix_x(-lamda), rotation_matrix_y(-i_s + np.pi/2))
    rot_p = np.matmul(R, p)  ######## rotating the spheroid (points) to match the projected view on the plane of the celestial sphere
    x_rot, y_rot, z_rot = rot_p[0], rot_p[1], rot_p[2]
    return x_rot, y_rot, z_rot, R



def vis_mask(x, y, z, Re_eq, Rp_polar, R):
    Nx = x/Re_eq**2
    Ny = y/Re_eq**2
    Nz = z/Rp_polar**2  ##### x,y,z normals
    norm = np.array([Nx, Ny, Nz])
    rot_norm = np.matmul(R, norm) #### rotate the normals
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
    temperature = T_pole*((g/g_pole)**beta)  ##### using the von Zeipel theorem for temperature variation
    I_planck = (2*h*c**2)/(obs_wavelength**5)/(np.exp(h*c/(obs_wavelength*k_B*temperature)) - 1)
    return I_planck, T_pole



def limb_darkening(mu, u1, u2):
    mu = np.clip(mu, 0, 1)
    return 1.0 - u1*(1 - mu) - u2*(1 - mu)**2



def sphere_unocculted_flux(Rs, u1, u2):
    return np.pi*(Rs**2)*(1.0 - u1/3.0 - u2/6.0)



def unocculted_flux(I_total, proj_area):
    dA = proj_area/len(I_total)  ### the area element for integration
    return np.sum(I_total)*dA



def sphere_intensity(Y, Z, Rs, u1, u2):
    rho2 = Y*Y + Z*Z
    mu2 = np.maximum(1.0 - rho2/(Rs*Rs), 0.0)
    mu = np.sqrt(mu2)
    return limb_darkening(mu, u1, u2)



def intensity(Y, Z, R, Re, Rp, T_pole, st_mass, beta, omega, obs_wavelength, u1, u2):
    Rt = R.T
    a11, a12, a13 = Rt[0]### replaced our old method for the intensity calculation (K-D tree) to an analytical expression
    a21, a22, a23 = Rt[1]
    a31, a32, a33 = Rt[2]
    lin_x = a12*Y + a13*Z
    lin_y = a22*Y + a23*Z
    lin_z = a32*Y + a33*Z
    inv_Re2 = 1.0/Re**2
    inv_Rp2 = 1.0/Rp**2

    A = (a11*a11 + a21*a21)*inv_Re2 + a31*a31*inv_Rp2
    B = 2*((a11*lin_x + a21*lin_y)*inv_Re2 + a31*lin_z*inv_Rp2)
    C = (lin_x*lin_x + lin_y*lin_y)*inv_Re2 + lin_z*lin_z*inv_Rp2 - 1

    disc = np.maximum(B*B - 4*A*C, 0.0)
    sq = np.sqrt(disc)
    x_los = (-B + sq)/(2*A)
    x = a11*x_los + a12*Y + a13*Z
    yv = a21*x_los + a22*Y + a23*Z
    zv = a31*x_los + a32*Y + a33*Z
    r = np.sqrt(x*x + yv*yv + zv*zv)

    cos_th = np.clip(zv/r, -1.0, 1.0)
    sin_th2 = np.maximum(1.0 - cos_th*cos_th, 0.0)
    sin_th = np.sqrt(sin_th2)

    Nx, Ny, Nz = x*inv_Re2, yv*inv_Re2, zv*inv_Rp2
    norm_mag = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz)
    mu = np.clip((R[0, 0]*Nx + R[0, 1]*Ny + R[0, 2]*Nz)/norm_mag, 0.0, 1.0)
    ratio2 = (Re*Re)*inv_Rp2
    r_th = Re/np.sqrt(sin_th2 + ratio2*cos_th*cos_th)
    GM = G*st_mass
    Om_sq = (omega*omega)*GM/Re**3
    g_pole = GM/Rp**2
    t1 = -GM/r_th**2 + r_th*Om_sq*sin_th2
    t2 = r_th*Om_sq*sin_th*cos_th
    g = np.sqrt(t1*t1 + t2*t2)
    temperature = T_pole*(g/g_pole)**beta
    I_planck = (2*h*c**2)/(obs_wavelength**5)/(np.exp(h*c/(obs_wavelength*k_B*temperature)) - 1)
    one_mu = 1.0 - mu
    I_limb = 1.0 - u1*one_mu - u2*one_mu*one_mu
    return I_planck*I_limb



def _unit_planet_grid(n_points):
    n_points = int(n_points)
    i = np.arange(n_points)
    r = np.sqrt((i + 0.5)/n_points)
    theta = _GOLDEN_ANGLE*i
    dx = r*np.cos(theta)
    dz = r*np.sin(theta)
    return dx, dz



def planet_grid(Rp_phys, n_points):
    dx_unit, dz_unit = _unit_planet_grid(int(n_points))
    Rp_phys = float(Rp_phys)
    dx = Rp_phys*dx_unit
    dz = Rp_phys*dz_unit
    dA = np.pi*(Rp_phys**2)/n_points
    return dx, dz, dA



def _sphere_grid_flux(js_sub, y_p, z_p, dy, dz, dA, Rs, u1, u2):
    Y_all = y_p[js_sub, None] + dy[None, :]
    Z_all = z_p[js_sub, None] + dz[None, :]
    on_star = (Y_all*Y_all + Z_all*Z_all) <= Rs*Rs

    i, j = np.nonzero(on_star)
    out = np.zeros(len(js_sub))
    if i.size == 0:
        return out

    Yq = Y_all[i, j]
    Zq = Z_all[i, j]
    Is = sphere_intensity(Yq, Zq, Rs, u1, u2)
    out[:] = np.bincount(i, weights=Is, minlength=len(js_sub))*dA
    return out



def _grid_flux(js_sub, y_p, z_p, dy, dz, dA, R, R_eq, R_polar, ellipse_geom, T_pole, st_mass, beta, omega, obs_wavelength, u1, u2):
    D, K, sin_lam, cos_lam, Re2D = ellipse_geom
    Y_all = y_p[js_sub, None] + dy[None, :]
    Z_all = z_p[js_sub, None] + dz[None, :]
    rho2 = Y_all*Y_all + Z_all*Z_all
    q = Z_all*cos_lam + Y_all*sin_lam
    on_star = D*rho2 + K*q*q <= Re2D

    i, j = np.nonzero(on_star)
    out = np.zeros(len(js_sub))
    if i.size == 0:
        return out

    Yq = Y_all[i, j]
    Zq = Z_all[i, j]
    Is = intensity(Yq, Zq, R, R_eq, R_polar, T_pole, st_mass, beta, omega, obs_wavelength, u1, u2)
    out[:] = np.bincount(i, weights=Is, minlength=len(js_sub))*dA
    return out



def sphere_occulted_flux(js, y_p, z_p, Rp_phys, Rs, u1, u2, n_coarse, n_fine):
    d_proj = np.sqrt(y_p[js]**2 + z_p[js]**2)
    s = (Rs - d_proj)/Rp_phys
    use_fine = np.abs(s) < 1.005 ### changing the grid size near ingress/egress
    DF = np.zeros(len(js))

    if np.any(~use_fine):
        dy_c, dz_c, dA_c = planet_grid(Rp_phys, n_coarse)
        DF[~use_fine] = _sphere_grid_flux(js[~use_fine], y_p, z_p, dy_c, dz_c, dA_c, Rs, u1, u2)

    if np.any(use_fine):
        dy_f, dz_f, dA_f = planet_grid(Rp_phys, n_fine)
        DF[use_fine] = _sphere_grid_flux(js[use_fine], y_p, z_p, dy_f, dz_f, dA_f, Rs, u1, u2)
    return DF



def occulted_flux(js, y_p, z_p, Rp_phys, R, R_eq, R_polar, i_s, lamda, T_pole, st_mass, beta, omega, obs_wavelength, 
                  u1, u2, n_coarse, n_fine):
    i_rad = np.deg2rad(i_s)
    lam_rad = np.deg2rad(lamda)
    sin_i = np.sin(i_rad)
    cos_i = np.cos(i_rad)
    sin_lam = np.sin(lam_rad)
    cos_lam = np.cos(lam_rad)

    D = (R_eq*cos_i)**2 + (R_polar*sin_i)**2
    K = (R_eq**2 - R_polar**2)*sin_i**2
    Re2D = (R_eq**2)*D

    ellipse_geom = (D, K, sin_lam, cos_lam, Re2D)
    theta_p = np.arctan2(z_p[js], y_p[js])
    r_edge = ellipse_radius(theta_p, R_eq, R_polar, i_rad, lam_rad)
    d_proj = np.sqrt(y_p[js]**2 + z_p[js]**2)
    s = (r_edge - d_proj)/Rp_phys
    use_fine = np.abs(s) < 1.005 ### changing the grid size near ingress/egress

    DF = np.zeros(len(js))

    if np.any(~use_fine):
        dy_c, dz_c, dA_c = planet_grid(Rp_phys, n_coarse)
        DF[~use_fine] = _grid_flux(js[~use_fine], y_p, z_p, dy_c, dz_c, dA_c, R, R_eq, R_polar, ellipse_geom, T_pole,
                                      st_mass, beta, omega, obs_wavelength, u1, u2)

    if np.any(use_fine):
        dy_f, dz_f, dA_f = planet_grid(Rp_phys, n_fine)
        DF[use_fine] = _grid_flux(js[use_fine], y_p, z_p, dy_f, dz_f, dA_f, R, R_eq, R_polar, ellipse_geom, T_pole,
                                     st_mass, beta, omega, obs_wavelength, u1, u2)
    return DF



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
    if np.abs(np.sin(i_s)) < 1e-8:
        return np.inf
    omega = 0.3  ### guess
    for _ in range(100):
        R_eq = R_mean_solar*(((2 + omega**2)/2)**(1/3))
        v_eq = (vsini_kms*1e3)/np.sin(i_s)
        Omega = v_eq/R_eq
        Omega_crit = np.sqrt(G*st_mass_solar/(R_eq**3))
        omega_new = Omega/Omega_crit
        if np.abs(omega_new - omega) < 1e-8:
            break
        omega = omega_new
    return omega_new



def transit_model(t_vals: NDArray[np.float64], orbital_period: float, st_mass: float, st_radius: float,
                  u1: float, u2: float, e: float, i_0: float, omega_p: float, raan: float, t_mid: float,
                  rp_rs: float, pl_grid: int = 1) -> NDArray[np.float64]:
    """
    Calculating the transit light-curve of a spherical, non-rotating star
 
    Parameters
    ----------
    t_vals : NDArray[np.float64]
        Time array. Unit: days.
    orbital_period : float
        Orbital period of the planet. Unit: days.
    st_mass : float
        Stellar mass. Unit: solar masses.
    st_radius : float
        Stellar radius. Unit: solar radii.
    u1, u2 : float
        Quadratic limb-darkening coefficients. Unit: dimensionless.
    e : float
        Orbital eccentricity. Unit: dimensionless.
    i_0 : float
        Orbital inclination. Unit: degrees.
    omega_p : float
        Argument of periastron. Unit: degrees.
    raan : float
        Longitude of the ascending node. Unit: degrees.
    t_mid : float
        Transit midtime. Unit: days.
    rp_rs : float
        Planet-to-star radius ratio. Unit: dimensionless.
    pl_grid : int, optional
        Factor increasing the planet's integration grid resolution. Default: 1.
 
    Returns
    -------
    NDArray[np.float64]
        Normalized flux at each time in ``t_vals``.
    """
    st_mass = st_mass*Ms
    st_radius = st_radius*Rs
    Rp_phys = rp_rs*st_radius
 
    F_out = sphere_unocculted_flux(st_radius, u1, u2)
 
    n1 = 30*int(pl_grid)
    n2 = 750*int(pl_grid)
 
    a = (G*st_mass*((orbital_period*86400)**2)/(4*np.pi**2))**(1/3)
    x_p, y_p, z_p = kep2car(t_vals, a, e, i_0, raan, omega_p, t_mid, orbital_period)
    d_proj = np.sqrt(y_p**2 + z_p**2)
    position_mask = (x_p >= 0) & (d_proj <= (st_radius + 1.001*Rp_phys))
 
    flux = np.ones_like(t_vals, dtype=float)
    js = np.where(position_mask)[0]
 
    DF = sphere_occulted_flux(js, y_p, z_p, Rp_phys, st_radius, u1, u2, n1, n2)
    flux[js] = (F_out - DF)/F_out
    return np.array(flux)



def gd_transit_model(t_vals: NDArray[np.float64], orbital_period: float, st_mass: float, st_mean_radius: float,
                     st_mean_temperature: float, beta: float, lamda: float, i_s: float, omega: float, u1: float,
                     u2: float, e: float, i_0: float, omega_p: float, raan: float, t_mid: float, rp_rs: float,
                     obs_wavelength: float, st_grid: int = 3500, pl_grid: int = 1) -> NDArray[np.float64]:
    """
    Calculating the gravity-darkened transit light-curve of an oblate, rotating star.

    Parameters
    ----------
    t_vals : NDArray[np.float64]
        Time array. Unit: days.

    orbital_period : float
        Orbital period of the planet. Unit: days.

    st_mass : float
        Stellar mass. Unit: solar masses.

    st_mean_radius : float
        Mean stellar radius. Unit: solar radii.

    st_mean_temperature : float
        Mean stellar temperature. Unit: 10000 K.

    beta : float
        Gravity-darkening exponent. Unit: Dimensionless.

    lamda : float
        Stellar projected obliguity. Unit: degrees.

    i_s : float
        Stellar inclination. Unit: degrees.

    omega : float
        Stellar rotational parameter. Unit: Dimensionless.

    u1 : float
        First quadratic limb-darkening coefficient. Unit: Dimensionless.

    u2 : float
        Second quadratic limb-darkening coefficient. Unit: Dimensionless.

    e : float
        Orbital eccentricity. Unit: Dimensionless.

    i_0 : float
        Orbital inclination. Unit: degrees.

    omega_p : float
        Argument of periastron. Unit: degrees.

    raan : float
        Longitude of the ascending node. Unit: degrees.

    t_mid : float
        Transit midtime. Unit: days.

    rp_rs : float
        Planet-to-star radius ratio. Unit: Dimensionless.

    obs_wavelength : float
        Observing wavelength. Unit: meters.

    st_grid : int, optional
        Number of points used to sample the stellar surface. Default: 3500.

    pl_grid : int, optional
        Factor increasing the planet's integration grid resolution. Default: 1.

    Returns
    -------
    NDArray[np.float64]
        Normalized flux at each time in ``t_vals``.
    """
    st_mass, st_mean_radius, st_mean_temperature = st_mass*Ms, st_mean_radius*Rs, st_mean_temperature*10000
    st_grid, pl_grid = int(st_grid), int(pl_grid)
    R_eq = st_mean_radius*((2 + omega**2)/2)**(1/3)
    R_polar = R_eq*(2/(2 + omega**2))
    x, y, z, lat = spheroid(R_eq, R_polar, st_grid)
    _, _, _, R = rotated_spheroid(x, y, z, np.deg2rad(lamda), np.deg2rad(i_s))
    mask, mu_all = vis_mask(x, y, z, R_eq, R_polar, R)
    mu_vis = mu_all[mask]

    I_grav, T_pole = gravity_darkening(st_mass, st_mean_temperature, beta, omega, obs_wavelength, R_eq, R_polar, lat)
    I_limb = limb_darkening(mu_vis, u1, u2)
    I_total = I_grav[mask]*I_limb

    A_ell = np.pi*R_eq*np.sqrt((R_eq*np.cos(np.deg2rad(i_s)))**2 + (R_polar*np.sin(np.deg2rad(i_s)))**2)  ##### projected area of the visible stellar disk
    F_out = unocculted_flux(I_total, A_ell)
    Rp_phys = rp_rs*st_mean_radius

    n1 = 30*pl_grid
    n2 = 750*pl_grid

    a = (G*st_mass*((orbital_period*86400)**2)/(4*np.pi**2))**(1/3)
    x_p, y_p, z_p = kep2car(t_vals, a, e, i_0, raan, omega_p, t_mid, orbital_period)
    d_proj = np.sqrt(y_p**2 + z_p**2)
    position_mask = (x_p >= 0) & (d_proj <= (R_eq + 1.001*Rp_phys))

    flux = np.ones_like(t_vals, dtype=float)
    js = np.where(position_mask)[0]

    DF = occulted_flux(js, y_p, z_p, Rp_phys, R, R_eq, R_polar, i_s, lamda, T_pole, st_mass, beta, omega, obs_wavelength, u1, u2, n1, n2)
    flux[js] = (F_out - DF)/F_out
    return np.array(flux)