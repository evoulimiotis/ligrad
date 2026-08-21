import numpy as np
import matplotlib.pyplot as plt
from ligrad.main import grav_dark_transit_model, vsini2omega

time = np.linspace(-0.1, 0.1, 320)

flux = grav_dark_transit_model(t_vals = time, orbital_period = 5.0, st_mass = 1.0, st_mean_radius = 1.0, 
                               st_mean_temperature = 0.58, beta = 0.2, lamda = 70.0, i_s = 50.0, omega = 0.6, 
                               u1 = 0.5, u2 = -0.1, e = 0.0, i_0 = 88.0, omega_p = 0.0, raan = 0.0, t_mid = 0.0, 
                               rp_rs = 0.1, obs_wavelength=800e-9, integration_grid_size=2)

### in case you want to use real rotational velocities (v*sini in km/s) for the stellar rotation you can use the vsini2omega function,
### to calculate the true dimensionless rotation factor (omega). See the commented example below:


# om = vsini2omega(vsini_kms = 195.4, st_mass_solar = 1.0, R_mean_solar = 1.0, i_s_deg = 50)
# flux2 = grav_dark_transit_model(t_vals = time, orbital_period = 5.0, st_mass = 1.0, st_mean_radius = 1.0, 
#                                st_mean_temperature = 0.58, beta = 0.2, lamda = 70.0, i_s = 50.0, omega = om, 
#                                u1 = 0.5, u2 = -0.1, e = 0.0, i_0 = 88.0, omega_p = 0.0, raan = 0.0, t_p = 0.0, 
#                                rp_rs = 0.1, obs_wavelength=800e-9, integration_grid_size=2)


plt.figure(figsize=(8, 4))
plt.plot(time, flux, color='darkblue', linewidth=2)
#plt.plot(time, flux2, color='darkgreen', linewidth=2, linestyle = '--')
plt.xlabel("Time (days)")
plt.ylabel("Normalized Flux")
plt.title("Transit Light-curve")
plt.grid(True)
plt.show()