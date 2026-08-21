import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation
import time
from astropy import constants as const
from ligrad.main import spheroid, rotated_spheroid, vis_mask, ellipse_radius, gravity_darkening, limb_darkening, kep2car, \
    grav_dark_transit_model, rotation_matrix_x, rotation_matrix_y, vsini2omega



Ms = const.M_sun.value
Ls = const.L_sun.value
Rs = const.R_sun.value
G = const.G.value
h = const.h.value
c = const.c.value
k_B = const.k_B.value



def surface_lines(ax, Re_eq, Rp_polar, lamda, i_s, Nmeridians=12, Nparallels=5):
    R = rotation_matrix_x(-np.deg2rad(lamda)) @ rotation_matrix_y(-np.deg2rad(i_s) + np.pi/2)
    for i in range(Nmeridians):
        phi = 2*np.pi*i/Nmeridians
        thetas = np.linspace(0, np.pi, 100)
        r = Re_eq*np.sqrt(1/(np.sin(thetas)**2 + ((Re_eq/Rp_polar)*np.cos(thetas))**2))
        points = np.array([r*np.sin(thetas)*np.cos(phi), r*np.sin(thetas)*np.sin(phi), r*np.cos(thetas)])
        rot_points = R @ points
        visible_points = rot_points[0] > 0
        if np.any(visible_points):
            diff = np.diff(np.concatenate(([False], visible_points, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                if e - s > 2:
                    ax.plot(rot_points[1, s:e], rot_points[2, s:e], color='black', alpha=0.6, linewidth=0.7)

    for theta in np.linspace(0.15*np.pi, 0.85*np.pi, Nparallels):
        phis = np.linspace(0, 2*np.pi, 200)
        r = Re_eq*np.sqrt(1/(np.sin(theta)**2 + ((Re_eq/Rp_polar)*np.cos(theta))**2))
        points = np.array([r*np.sin(theta)*np.cos(phis), r*np.sin(theta)*np.sin(phis), r*np.cos(theta)*np.ones_like(phis)])
        rot_points = R @ points
        visible_points = rot_points[0] > 0
        if np.sum(visible_points) > 3:
            vy = rot_points[1, visible_points]
            vz = rot_points[2, visible_points]
            vp = phis[visible_points]
            i_sort = np.argsort(vp)
            sorted_phi = vp[i_sort]
            sorted_y = vy[i_sort]
            sorted_z = vz[i_sort]
            gaps = np.where(np.diff(sorted_phi) > np.pi/3)[0]
            i_i = 0
            for gap in np.append(gaps, len(sorted_phi) - 1):
                i_f = gap + 1
                if i_f - i_i > 2:
                    ax.plot(sorted_y[i_i:i_f], sorted_z[i_i:i_f], color='black', alpha=0.6, linewidth=0.7)
                i_i = i_f





def simulation(filename, orbital_period, st_mass, st_mean_radius, st_mean_temperature, beta, lamda, i_s, u1, u2, e, i_0,
               omega_p, raan, t_mid, rp_rs, veq_sini=None, omega=None, obs_wavelength=800e-9, integration_grid_size=1,
               t_start=-0.15, t_end=0.15, time_points=250, save=True):

    if omega is None:
        if veq_sini is None:
            raise ValueError("Provide either omega or veq_sini.")
        omega = vsini2omega(veq_sini, st_mass, st_mean_radius, i_s)


    ts = np.linspace(t_start, t_end, time_points)
    
    print("\n\nComputing light-curve...")
    start1 = time.time()
    flux_values = grav_dark_transit_model(ts, orbital_period, st_mass, st_mean_radius, st_mean_temperature, beta, lamda, i_s,
                                          omega, u1, u2, e, i_0, omega_p, raan, t_mid, rp_rs, obs_wavelength, integration_grid_size)
    end1 = time.time()
    print("   Completed with runtime (sec):", round(end1-start1, 4))


    print("\nMaking the simulation...")
    start2 = time.time()
    st_mass_si, st_mean_radius_si, st_mean_temperature_si = st_mass*Ms, st_mean_radius*Rs, st_mean_temperature*10000
    R_eq = st_mean_radius_si*((2 + omega**2)/2)**(1/3)
    R_polar = R_eq*(2/(2 + omega**2))

    x, y, z, lat = spheroid(R_eq, R_polar, 5000)
    _, y_rot, z_rot, R = rotated_spheroid(x, y, z, np.deg2rad(lamda), np.deg2rad(i_s))
    mask, mu_all = vis_mask(x, y, z, R_eq, R_polar, R)
    y_vis = y_rot[mask]
    z_vis = z_rot[mask]
    mu_vis = mu_all[mask]

    I_grav = gravity_darkening(st_mass_si, st_mean_temperature_si, beta, omega, obs_wavelength, R_eq, R_polar, lat)
    #I_limb = limb_darkening(mu_vis, u1, u2)
    I_vis = I_grav[mask]#*(I_limb)
    Rp_phys = rp_rs*st_mean_radius_si


    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.set_xlim(-1.2*R_eq, 1.2*R_eq)
    ax1.set_ylim(-1.2*R_eq, 1.2*R_eq)
    ax1.set_xlabel(r'$Y$ ($m$)', fontsize = 12)
    ax1.set_ylabel(r'$Z$ ($m$)', fontsize = 12)
    ax1.set_title('Transit Simulation (Y-Z Plane)')
    ax1.set_aspect('equal')

    ax2.set_xlim(np.min(ts), np.max(ts))
    ax2.set_ylim(0.9995*np.min(flux_values), 1.0005*np.max(flux_values))
    ax2.set_xlabel(r'Time ($days$)', fontsize = 12)
    ax2.set_ylabel(r'Normalized Flux ($F/Fs$)', fontsize = 12)
    ax2.set_title('Transit Light-Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    trai_grid = Triangulation(y_vis, z_vis)
    star_surface = ax1.tripcolor(trai_grid, I_vis, shading='gouraud', cmap='YlOrBr_r', vmin=0.95*I_vis.min(), vmax=I_vis.max())
    surface_lines(ax1, R_eq, R_polar, lamda, i_s, Nmeridians=12, Nparallels=5)

    planet_circle = plt.Circle((0, 0), Rp_phys, color='black', alpha=0.9)
    ax1.add_patch(planet_circle)

    light_curve_line, = ax2.plot([], [], color='darkblue', linestyle = '-', linewidth=2)
    current_point, = ax2.plot([], [], 'ro', markersize=8)
    trail_line, = ax1.plot([], [], 'black', linewidth=3, alpha=0.8)
    trail_y, trail_z = [], []


    a_orbit = (G*st_mass_si*((orbital_period*86400)**2)/(4*np.pi**2))**(1/3)

    def _animate(frame):
        if frame == 0:
            trail_y.clear()
            trail_z.clear()
        t = ts[frame]
        x_p, y_p, z_p = kep2car(t, a_orbit, e, i_0, raan, omega_p, t_mid, orbital_period)
        planet_circle.center = (y_p, z_p)
        if x_p >= 0:
            planet_circle.set_alpha(0.95)
            trail_y.append(y_p)
            trail_z.append(z_p)
            if len(trail_y) > len(ts):
                trail_y.pop(0)
                trail_z.pop(0)
        else:
            planet_circle.set_alpha(0.1)
        
        if len(trail_y) > 1:
            trail_line.set_data(trail_y, trail_z)
        else:
            trail_line.set_data([], [])
            
        light_curve_line.set_data(ts[:frame + 1], flux_values[:frame + 1])
        current_point.set_data([t], [flux_values[frame]])
        ax1.set_title(f'Transit Simulation - Time: {t:.3f} days', fontsize=15, fontweight='bold')
        return star_surface, planet_circle, light_curve_line, current_point, trail_line


    cbar = plt.colorbar(star_surface, ax=ax1, shrink=0.8)
    cbar.set_label(r'Surface Intensity ($W/m²/sr$)')
    
    fps = time_points/8.0
    interval_ms = 1000/fps
    anim = FuncAnimation(fig, _animate, frames=len(ts), interval=interval_ms, blit=False, repeat=True)
    end2 = time.time()
    print("   Completed with runtime (sec):", round(end2-start2, 4))
    
    if save == True:
        print("\nSaving please wait...")
        start3 = time.time()
        anim.save(filename, writer='pillow', fps=fps, dpi=120)
        end3 = time.time()
        print("\nFile saved!")
        print("   Completed with runtime (sec):", round(end3-start3, 4))
        plt.close()
    elif save == False:
        plt.show()
    return None




######## example below
simulation(filename='kelt9b_test.gif', orbital_period=1.4811235, st_mass=2.52, st_mean_radius=2.36, st_mean_temperature=1.017,
           beta=0.138, lamda=255.6, i_s=51.6, u1=0.222, u2=0.17, e=0.0, i_0=85.3, omega_p=0.0, raan=0.0, t_p=0.0, rp_rs=0.078, 
           veq_sini=111.4, obs_wavelength=800e-9, integration_grid_size=5, t_start=-0.15, t_end=0.15, time_points=250, save = True)

