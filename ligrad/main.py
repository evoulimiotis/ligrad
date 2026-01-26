import numpy as np
from scipy.spatial import Delaunay, ConvexHull, cKDTree
from astropy import constants as const
import pylightcurve as plc

Ms = const.M_sun.value
Ls = const.L_sun.value
Rs = const.R_sun.value
G = const.G.value
h = const.h.value
c = const.c.value
k_B = const.k_B.value

def grav_dark_transit_model(t_vals, orbital_period, st_mass, st_mean_radius, st_mean_temperature, beta, lamda, i_s, omega, 
                            u1, u2, u3, u4, e, i_0, omega_p, t_p, rp_rs, obs_wavelength=800e-9, 
                            planet_focused=True, integration_grid_size=45):
    
    st_mass, st_mean_radius, st_mean_temperature = st_mass*Ms, st_mean_radius*Rs, st_mean_temperature*10000
    if not hasattr(grav_dark_transit_model, '_baseline_cache'):
        grav_dark_transit_model._baseline_cache = {}
    
    def spheroid(Re_eq, Rp_polar, num_points):
        u = np.linspace(0, 2*np.pi, int(np.sqrt(num_points)))
        v = np.linspace(0, np.pi, int(num_points/np.sqrt(num_points)))
        U, V = np.meshgrid(u, v)
        r = Re_eq*np.sqrt(1/(np.sin(V)**2 + ((Re_eq/Rp_polar)*np.cos(V))**2))
        x = r*np.sin(V)*np.cos(U)
        y = r*np.sin(V)*np.sin(U)
        z = r*np.cos(V)
        return x, y, z, V

    def rotation_matrix_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
    
    def rotation_matrix_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle),  np.cos(angle), 0],
                        [0, 0, 1]])

    def rotated_spheroid(x, y, z, lamda, i_s):
        b_angle = np.arctan(1/(np.tan(i_s)*np.cos(lamda)))
        a_angle = np.arctan(np.tan(i_s)*np.sin(lamda))
        points = np.array([x.flatten(), y.flatten(), z.flatten()])
        R = rotation_matrix_z(a_angle) @ rotation_matrix_y(b_angle)
        rotated_points = R @ points
        x_rot, y_rot, z_rot = [rotated_points[i].reshape(x.shape) for i in range(3)]
        return x_rot, y_rot, z_rot

    def edge_radii(y, z, yc, zc):
        y_input = np.asarray(y)
        z_input = np.asarray(z)
        single_point_input = y_input.ndim == 0 and z_input.ndim == 0
        if single_point_input:
            y_input = np.array([y_input])
            z_input = np.array([z_input])
        surface_points = np.column_stack([y_input, z_input])
        hull = ConvexHull(surface_points)
        edge_points = surface_points[hull.vertices]
        if yc is None or zc is None:
            center = np.mean(edge_points, axis=0)
        else:
            center = np.array([yc, zc])
        centered_points = edge_points - center
        thetas = np.arctan2(centered_points[:, 1], centered_points[:, 0])
        radii_raw = np.sqrt(np.sum(centered_points**2, axis=1))
        sorted_idx = np.argsort(thetas)
        thetas = thetas[sorted_idx]
        radii_raw = radii_raw[sorted_idx]
        thetas_ext = np.concatenate([thetas, thetas[0:1] + 2*np.pi])
        radii_ext = np.concatenate([radii_raw, radii_raw[0:1]])
        dy = y_input - center[0]
        dz = z_input - center[1]
        point_angles = np.arctan2(dz, dy)
        point_angles = np.where(point_angles < 0, point_angles + 2*np.pi, point_angles)
        result = np.interp(point_angles, thetas_ext, radii_ext, period=2*np.pi)
        if single_point_input:
            return float(result[0])
        else:
            return result

    def gravity_darkening(omega, Re_eq, Rp_polar, theta):
        r = Re_eq*np.sqrt(1/(np.sin(theta)**2 + ((Re_eq/Rp_polar)*np.cos(theta))**2))
        Omega_Kepl = np.sqrt(G*st_mass/(Re_eq**3))
        g = np.abs(-G*st_mass/(r**2) + r*((omega*Omega_Kepl)**2)*(np.sin(theta)**2))
        g_mean = np.abs(-G*st_mass/(st_mean_radius**2))
        temperature = st_mean_temperature*((g/g_mean)**beta)
        I_planck = (2*h*c**2)/(obs_wavelength**5)/(np.exp(h*c/(obs_wavelength*k_B*temperature)) - 1)
        return I_planck

    def limb_darkening(y_proj, z_proj, u1, u2, u3, u4):
        yc = 0.5*(y_proj.max() + y_proj.min())
        zc = 0.5*(z_proj.max() + z_proj.min())
        y_centered = y_proj - yc
        z_centered = z_proj - zc
        radius_outer = edge_radii(y_proj, z_proj, yc, zc)
        rho = np.sqrt(y_centered**2 + z_centered**2)/radius_outer
        mu = np.sqrt(1 - rho**2)
        I_claret = np.ones_like(rho)
        inside = rho <= 1
        I_claret[inside] = 1 - u1*(1 - mu[inside]) - u2*(1 - mu[inside])**2 - u3*(1 - mu[inside])**3 - u4*(1 - mu[inside])**4
        I_claret[~inside] = 0.0
        return I_claret
    
    def planet_position(t, period, e, inc, w, t_p):
        a = (G*st_mass*((period*86400)**2)/(4*np.pi**2))**(1/3)
        return plc.planet_orbit(period, a, e, inc, w, t_p, t)
    
    def baseline_flux():
        cache_key = (st_mass, st_mean_radius, st_mean_temperature, beta, lamda, i_s, omega, u1, u2, u3, u4)
        if cache_key in grav_dark_transit_model._baseline_cache:
            return grav_dark_transit_model._baseline_cache[cache_key]
        
        Re_eq = st_mean_radius
        Rp_polar = Re_eq*(2/(2 + omega**2))
        x, y, z, V = spheroid(Re_eq, Rp_polar, 2500)
        x_rot, y_rot, z_rot = rotated_spheroid(x, y, z, np.radians(lamda), np.radians(i_s))
        I_gr = gravity_darkening(omega, Re_eq, Rp_polar, V)
        Nx = grad_vector(x_rot, y_rot, z_rot)
        mask = (Nx > 0)
        y_vis = y_rot[mask]
        z_vis = z_rot[mask]
        I_vis = I_gr[mask]
        I_ld = limb_darkening(y_vis, z_vis, u1, u2, u3, u4)
        I_total = I_vis*I_ld
        pts = np.column_stack((y_vis, z_vis))
        tri = Delaunay(pts)
        triangles = tri.simplices
        pts_tri = pts[triangles]
        areas = 0.5*np.abs((pts_tri[:,1,0] - pts_tri[:,0,0])*(pts_tri[:,2,1] - pts_tri[:,0,1]) - 
                          (pts_tri[:,2,0] - pts_tri[:,0,0])*(pts_tri[:,1,1] - pts_tri[:,0,1]))
        I_mean = np.mean(I_total[triangles], axis=1)
        total_flux = np.sum(areas*I_mean)
        yc = 0.5*(y_vis.max() + y_vis.min())
        zc = 0.5*(z_vis.max() + z_vis.min())
        R_eff = np.mean(edge_radii(y_vis, z_vis, yc, zc))
        result = (total_flux, y_vis, z_vis, I_total, R_eff, Re_eq, Rp_polar, x, y, z, V)
        grav_dark_transit_model._baseline_cache[cache_key] = result
        return result
    
    def grad_vector(x, y, z):
        dx_dv = np.gradient(x, axis=0)
        dx_du = np.gradient(x, axis=1)
        dy_dv = np.gradient(y, axis=0)
        dy_du = np.gradient(y, axis=1)
        dz_dv = np.gradient(z, axis=0)
        dz_du = np.gradient(z, axis=1)
        Nx = (dy_dv*dz_du) - (dz_dv*dy_du)
        Ny = (dz_dv*dx_du) - (dx_dv*dz_du)
        Nz = (dx_dv*dy_du) - (dy_dv*dx_du)
        mag = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        return Nx/(10e-3 + mag)

    def planet_integration(y_p, z_p, Rp_phys, y_vis, z_vis, I_total, Re_eq, Rp_polar, x_orig, y_orig, z_orig, V_orig):
        if Rp_phys <= 0:
            return 0.0
        integration_radius = Rp_phys
        
        total_points = integration_grid_size*integration_grid_size
        n_radial = int(np.sqrt(total_points/np.pi))
        n_angular = total_points//n_radial
        y_flat = []
        z_flat = []
        for i in range(n_radial):
            if i == 0:
                y_flat.append(y_p)
                z_flat.append(z_p)
            else:
                r = (Rp_phys*i)/n_radial
                points_in_ring = max(1, int(n_angular*r/Rp_phys))
                for j in range(points_in_ring):
                    theta = 2*np.pi*j/points_in_ring
                    y_flat.append(y_p + r*np.cos(theta))
                    z_flat.append(z_p + r*np.sin(theta))
        y_flat = np.array(y_flat)
        z_flat = np.array(z_flat)
        dist_from_planet = np.sqrt((y_flat - y_p)**2 + (z_flat - z_p)**2)
        planet_mask = dist_from_planet <= Rp_phys
        
        if not np.any(planet_mask):
            return 0.0
        y_shadow = y_flat[planet_mask]
        z_shadow = z_flat[planet_mask]
        yc = 0.5*(y_vis.max() + y_vis.min())
        zc = 0.5*(z_vis.max() + z_vis.min())
        stellar_radius = np.mean(edge_radii(y_vis, z_vis, yc, zc))
        dist_from_stellar_center = np.sqrt((y_shadow - yc)**2 + (z_shadow - zc)**2)
        on_star_mask = dist_from_stellar_center <= stellar_radius
        if not np.any(on_star_mask):
            return 0.0
        y_valid = y_shadow[on_star_mask]
        z_valid = z_shadow[on_star_mask]
        visible_points = np.column_stack([y_vis, z_vis])
        tree = cKDTree(visible_points)
        valid_points = np.column_stack([y_valid, z_valid])
        distances, indices = tree.query(valid_points, k=1)
        I_interpolated = I_total[indices]
        y_centered_valid = y_valid - yc
        z_centered_valid = z_valid - zc
        rho_valid = np.sqrt(y_centered_valid**2 + z_centered_valid**2)/stellar_radius
        valid_star_mask = rho_valid <= 1.0
        I_final = np.zeros_like(I_interpolated)
        if np.any(valid_star_mask):
            mu_valid = np.sqrt(np.maximum(0.0, 1 - rho_valid[valid_star_mask]**2))
            factor = 1 - mu_valid
            limb_correction = 1 - u1*factor - u2*factor**2 - u3*factor**3 - u4*factor**4
            I_final[valid_star_mask] = I_interpolated[valid_star_mask]
        dy = 2 * integration_radius/integration_grid_size
        dz = 2 * integration_radius/integration_grid_size
        cell_area = dy*dz
        total_flux = np.sum(I_final)*cell_area
        return total_flux
    
    def flux_drop(y_p, z_p, Rp_phys, y_vis, z_vis, I_total):
        dist2 = (y_vis - y_p)**2 + (z_vis - z_p)**2
        planet_mask = dist2 <= Rp_phys**2
        if not np.any(planet_mask):
            return 0.0
        y_planet = y_vis[planet_mask]
        z_planet = z_vis[planet_mask]
        I_planet = I_total[planet_mask]
        if len(y_planet) < 3:
            return 0.0
        pts = np.column_stack((y_planet, z_planet))
        try:
            tri = Delaunay(pts)
            triangles = tri.simplices
            pts_tri = pts[triangles]
            areas = 0.5*np.abs((pts_tri[:,1,0] - pts_tri[:,0,0])*(pts_tri[:,2,1] - pts_tri[:,0,1]) - 
                              (pts_tri[:,2,0] - pts_tri[:,0,0])*(pts_tri[:,1,1] - pts_tri[:,0,1]))
            I_mean = np.mean(I_planet[triangles], axis=1)
            flux_drop = np.sum(areas*I_mean)
            return flux_drop
        except:
            return 0.0
    
    def transit_lightcurve(t_vals):
        base_flux, y_vis, z_vis, I_total, R_eff, Re_eq, Rp_polar, x_orig, y_orig, z_orig, V_orig = baseline_flux()
        Rp_phys = rp_rs*R_eff
        flux_vals = []
        for t in t_vals:
            x_p, y_p, z_p = planet_position(t, orbital_period, e, i_0, omega_p, t_p)
            if x_p >= 0:
                if planet_focused:
                    flux_dip = planet_integration(y_p, z_p, Rp_phys, y_vis, z_vis, I_total, 
                                                                   Re_eq, Rp_polar, x_orig, y_orig, z_orig, V_orig)
                    flux_norm = (base_flux - flux_dip)/base_flux
                else:
                    flux_dip = flux_drop(y_p, z_p, Rp_phys, y_vis, z_vis, I_total)
                    flux_norm = (base_flux - flux_dip)/base_flux
            else:
                flux_norm = 1.0
            flux_vals.append(flux_norm)
        return np.array(flux_vals)
    
    tr_lightcurve = transit_lightcurve(t_vals)
    return tr_lightcurve

