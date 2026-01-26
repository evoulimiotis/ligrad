import numpy as np
import matplotlib.pyplot as plt
from ligrad.main import grav_dark_transit_model

time = np.linspace(-0.2, 0.2, 200)

flux = grav_dark_transit_model(t_vals = time, orbital_period = 5.0, st_mass = 1.0, st_mean_radius = 1.0, 
                                    st_mean_temperature = 0.58, beta = 0.2, lamda = 70, i_s = 50, omega = 0.5, 
                                    u1 = 0.5, u2 = -0.1, u3 = 0.05, u4 = -0.02, e = 0.0, i_0 = 88, omega_p = 0, t_p = 0, 
                                    rp_rs = 0.1, obs_wavelength=800e-9, integration_grid_size=45)

plt.figure(figsize=(8, 4))
plt.plot(time, flux, color='darkblue', linewidth=2)
plt.xlabel("Time (days)")
plt.ylabel("Normalized Flux")
plt.title("Transit Light-curve")
plt.grid(True)
plt.show()