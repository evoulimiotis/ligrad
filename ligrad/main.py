import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from astropy import constants as const
import pylightcurve as plc

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
    return rot_norm[0] > 0  #### only the points with positive norm elements should be visible

def ellipse_radius(y_vis, z_vis, theta):
    ell_pts = np.column_stack([y_vis, z_vis])
    hull = ConvexHull(ell_pts)  #### points on the projected ellipse boundary
    ye, ze = y_vis[hull.vertices], z_vis[hull.vertices]
    D = np.column_stack([ye**2, ye*ze, ze**2, ye, ze])
    synt, _, _, _ = np.linalg.lstsq(D, np.ones(len(ye)), rcond=None)  ### fitting the ellipse parameters with least squares
    a, b, c, d, e = synt
    M = np.array([[2*a, b], [b, 2*c]])
    yc, zc = np.linalg.solve(M, [-d, -e])
    A = np.array([[a, b/2], [b/2, c]])
    eigvals, eigvecs = np.linalg.eigh(A)
    f = 1 - a*yc**2 - b*yc*zc - c*zc**2 - d*yc - e*zc
    semi_ax = np.sqrt(f/eigvals)
    a_s, b_s = semi_ax
    psi = np.arctan2(eigvecs[1,0], eigvecs[0,0])
    phi = theta - psi
    r = (a_s*b_s)/np.sqrt((b_s*np.cos(phi))**2 + (a_s*np.sin(phi))**2)  ##### radius of the projected ellipse at theta, centered around (yc, zc)
    return r, yc, zc

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

def limb_darkening(y_proj, z_proj, yc, zc, r, u1, u2, u3, u4):
    y_cent = y_proj - yc
    z_cent = z_proj - zc
    rho = np.sqrt(y_cent**2 + z_cent**2)/r
    mu = np.sqrt(np.clip(1 - rho**2, 0, None))
    I_claret = np.ones_like(rho)
    inside = rho <= 1  #####m using the 4-term Claret law for limb darkening calculation
    I_claret[inside] = 1 - u1*(1 - mu[inside]) - u2*(1 - mu[inside])**2 - u3*(1 - mu[inside])**3 - u4*(1 - mu[inside])**4
    I_claret[~inside] = 0.0
    return I_claret

def baseline_flux(I_total, y_vis, z_vis):
    p = np.column_stack([y_vis, z_vis])
    hull = ConvexHull(p)
    proj_area = hull.volume  ##### projected area of the visible stellar disk
    dA = proj_area/len(I_total)  ### the area element for integration
    return np.sum(I_total)*dA

def planet_integration(y_p, z_p, Rp_phys, I_total, yc, zc, ped, r_ell, tree, integration_grid_size=45):
    if Rp_phys <= 0:
        return 0.0
    n = integration_grid_size
    y_grid = np.linspace(y_p - Rp_phys, y_p + Rp_phys, n)  ###### making a uniform grid over the planet disk
    z_grid = np.linspace(z_p - Rp_phys, z_p + Rp_phys, n)
    Y, Z = np.meshgrid(y_grid, z_grid)
    Y, Z = Y.flatten(), Z.flatten()
    planet = (Y - y_p)**2 + (Z - z_p)**2 <= Rp_phys**2  ##### mask for inside planet disk
    Y, Z = Y[planet], Z[planet]
    if len(Y) == 0:
        return 0.0
    point_ang = np.arctan2(Z - zc, Y - yc)  ## the polar angle of each planet grid point (Yi,Zi) measured from the stellar center
    point_ang = np.where(point_ang < 0, point_ang + 2*np.pi, point_ang)
    r_at_points = np.interp(point_ang, ped, r_ell)
    on_star = np.sqrt((Y - yc)**2 + (Z - zc)**2) <= r_at_points  #### mask for inside the stellar 2-d projected elliptic disk
    Y, Z = Y[on_star], Z[on_star]
    if len(Y) == 0:
        return 0.0
    _, iss = tree.query(np.column_stack([Y, Z]), k=1)  ###### index of the nearest stellar point for each planet grid point
    dA = (2*Rp_phys/n)**2
    return np.sum(I_total[iss])*dA

def planet_position(t, period, e, inc, w, t_p, st_mass):
        a = (G*st_mass*((period*86400)**2)/(4*np.pi**2))**(1/3)
        return plc.planet_orbit(period, a, e, inc, w, t_p, t)

def grav_dark_transit_model(t_vals, orbital_period, st_mass, st_mean_radius, st_mean_temperature, 
                             beta, lamda, i_s, omega, u1, u2, u3, u4, e, i_0, omega_p, t_p, 
                             rp_rs, obs_wavelength=800e-9, integration_grid_size=45):
    st_mass, st_mean_radius, st_mean_temperature = st_mass*Ms, st_mean_radius*Rs, st_mean_temperature*10000
    R_eq = st_mean_radius*((2 + omega**2)/2)**(1/3)  ##### for the volume-preserving stellar mean radius
    R_polar = R_eq*(2/(2 + omega**2))
    x, y, z, lat = spheroid(R_eq, R_polar, 2500)  ##### constructing the stellar surface
    _, y_rot, z_rot, R = rotated_spheroid(x, y, z, np.deg2rad(lamda), np.deg2rad(i_s))
    mask = vis_mask(x, y, z, R_eq, R_polar, R)
    y_vis = y_rot[mask]
    z_vis = z_rot[mask]
    
    I_grav = gravity_darkening(st_mass, st_mean_temperature, beta, omega, obs_wavelength, R_eq, R_polar, lat)
    ped = np.linspace(0, 2*np.pi, 100)
    r_ell, yc, zc = ellipse_radius(y_vis, z_vis, ped)
    point_angles = np.arctan2(z_vis - zc, y_vis - yc)
    point_angles = np.where(point_angles < 0, point_angles + 2*np.pi, point_angles)
    r_points = np.interp(point_angles, ped, r_ell)
    I_limb = limb_darkening(y_vis, z_vis, yc, zc, r_points, u1, u2, u3, u4)
    I_total = I_grav[mask]*I_limb    ###### computing the full intensity profile
    
    F_out = baseline_flux(I_total, y_vis, z_vis)
    tree = cKDTree(np.column_stack([y_vis, z_vis]))
    Rp_phys = rp_rs*st_mean_radius  ###### again for the volume-preserving mean radius
    flux = []
    for t in t_vals:
        x_p, y_p, z_p = planet_position(t, orbital_period, e, i_0, omega_p, t_p, st_mass)
        if x_p >= 0:
            DF = planet_integration(y_p, z_p, Rp_phys, I_total, yc, zc, ped, r_ell, tree, integration_grid_size)
            flux_norm = (F_out - DF)/F_out  #### calculating the normalized flux at each time step
        else:
            flux_norm = 1.0
        flux.append(flux_norm)
    return np.array(flux)

