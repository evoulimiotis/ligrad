import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from astropy.io import fits


def load_data(filename):
    with fits.open(filename) as hdul:
        data = hdul[1].data
        header = hdul[1].header
        time = data['TIME'] + header['BJDREFI'] + header['BJDREFF']
        flux = data['PDCSAP_FLUX']
        flux_err = data['PDCSAP_FLUX_ERR']
    mask = ~(np.isnan(time) | np.isnan(flux) | np.isnan(flux_err))
    return time[mask], flux[mask], flux_err[mask]


def normalize_flux(time, flux, flux_err):
    err_floor = np.nanmedian(flux_err)*1e-3
    safe_err = np.maximum(flux_err, err_floor)
    w = 1.0/safe_err**2
    c0 = np.polyfit(time, flux, 1, w=w)
    resid = flux - np.polyval(c0, time)
    mad = np.median(np.abs(resid - np.median(resid)))
    inlier = np.abs(resid - np.median(resid)) < 3*1.4826*mad
    coeffs, cov = np.polyfit(time[inlier], flux[inlier], 1, w=1.0/safe_err[inlier]**2, cov=True)
    trend = np.polyval(coeffs, time)
    trend_var = ((time**2)*cov[0, 0] + cov[1, 1] + 2.0*time*cov[0, 1])
    trend_var = np.maximum(trend_var, 0.0)
    norm_flux = flux/trend
    norm_err = np.sqrt(((safe_err/trend)**2) + ((flux / trend**2)**2)*trend_var)
    return norm_flux, norm_err



def normalize_transit_local(time, flux, flux_err, transit_duration_days, degree=2):
    midtime = np.median(time)
    half_dur = transit_duration_days/2.0
    oot_mask = np.abs(time - midtime) > 1.5*half_dur
    err_floor = np.nanmedian(flux_err)
    safe_err = np.maximum(flux_err, err_floor*1e-3)

    if oot_mask.sum() < degree + 2:
        if oot_mask.sum() > 0:
            w_oot = 1.0/np.maximum(flux_err[oot_mask], err_floor)**2
            baseline = np.average(flux[oot_mask], weights=w_oot)
            baseline_var = 1.0/w_oot.sum()
        else:
            baseline = np.median(flux)
            baseline_var = np.nanmedian(flux_err)**2
        norm_flux = flux/baseline
        norm_err = np.sqrt(((safe_err / baseline)**2) + ((flux / baseline**2)**2)*baseline_var)
        return norm_flux, norm_err

    t0 = np.median(time)
    t_norm = time - t0
    safe_oot_err = np.maximum(flux_err[oot_mask], err_floor)
    w_oot = 1.0/safe_oot_err**2
    try:
        coeffs, cov = np.polyfit(t_norm[oot_mask], flux[oot_mask], degree, w=w_oot, cov=True)
    except (np.linalg.LinAlgError, ValueError):
        coeffs = np.polyfit(t_norm[oot_mask], flux[oot_mask], degree, w=w_oot)
        cov = None
    trend = np.polyval(coeffs, t_norm)
    if cov is not None:
        V = np.vander(t_norm, degree + 1)
        trend_var = np.einsum('ij,jk,ik->i', V, cov, V)
        trend_var = np.maximum(trend_var, 0.0)
    else:
        trend_var = np.zeros(len(time))
    norm_flux = flux/trend
    norm_err = np.sqrt(((safe_err/trend)**2) + ((flux/trend**2)**2)*trend_var)
    return norm_flux, norm_err



def detect_transits(time, norm_flux, sigma_threshold=3.0):
    window = min(51, len(norm_flux) // 10)
    if window % 2 == 0:
        window += 1
    smoothed = signal.savgol_filter(norm_flux, window, 3) if len(norm_flux) > window else norm_flux
    baseline = np.median(smoothed)
    residuals = smoothed - baseline
    noise_sigma = 1.4826*np.median(np.abs(residuals - np.median(residuals)))
    dip_threshold = baseline - sigma_threshold*noise_sigma
    dip_mask = smoothed < dip_threshold
    if not np.any(dip_mask):
        print("no transits found")
        return np.array([])
    dip_i = np.where(dip_mask)[0]
    breaks = np.where(np.diff(dip_i) > 5)[0] + 1
    groups = [g for g in np.split(dip_i, breaks) if len(g) >= 3]
    centers = np.array([g[np.argmin(smoothed[g])] for g in groups])
    print("Found", len(centers), "transit candidates")
    return centers


def _update_midtime(time, flux, flux_err, det_i, half_days, dip_threshold):
    det_mid = time[det_i]
    mask = (time >= det_mid - half_days) & (time <= det_mid + half_days)
    t0, f0, e0 = time[mask], flux[mask], flux_err[mask]

    dip_i = np.where(f0 < dip_threshold)[0]
    if len(dip_i) == 0:
        return det_mid, half_days

    segs = np.split(dip_i, np.where(np.diff(dip_i) > 1)[0] + 1)
    seg = max(segs, key=len)
    true_mid = 0.5*(t0[seg[0]] + t0[seg[-1]])

    if len(seg) >= 2:
        cadence = np.median(np.diff(t0))
        safe_e_in = np.maximum(e0[seg], np.nanmedian(e0)*1e-3)
        w_in = 1.0/safe_e_in**2
        mean_in = np.average(f0[seg], weights=w_in)
        sigma_in = np.sqrt(1.0/w_in.sum())
        oot_i = np.setdiff1d(np.arange(len(t0)), seg)
        if len(oot_i) >= 3:
            safe_e_oot = np.maximum(e0[oot_i], np.nanmedian(e0)*1e-3)
            w_oot = 1.0 / safe_e_oot**2
            baseline = np.average(f0[oot_i], weights=w_oot)
            baseline_var = 1.0/w_oot.sum()
        else:
            baseline = np.median(f0)
            baseline_var = np.nanmedian(e0)**2
            
        depth = baseline - mean_in
        if depth > 0:
            depth_sigma = np.sqrt(sigma_in**2 + baseline_var)
            n_in = len(seg)
            mid_err = (depth_sigma/depth)*cadence*np.sqrt(2.0/max(n_in, 2))
        else:
            mid_err = cadence
    else:
        mid_err = half_days
    return true_mid, mid_err



def transit_properties(time, flux, flux_err, midtime_i, window_hours=24):
    midtime = time[midtime_i]
    half = (window_hours/2)/24.0
    mask = (time >= midtime - half) & (time <= midtime + half)
    t, f, e = time[mask], flux[mask], flux_err[mask]
    if len(t) < 5:
        return None, None, None, None
    t_span = t[-1] - t[0]
    oot = (t <= t[0] + 0.3*t_span) | (t >= t[-1] - 0.3*t_span)
    if oot.sum() >= 3:
        w_oot = 1.0/e[oot]**2
        baseline = np.average(f[oot], weights=w_oot)
        baseline_var = 1.0/w_oot.sum()
    else:
        w_all = 1.0/e**2
        baseline = np.average(f, weights=w_all)
        baseline_var = 1.0/w_all.sum()

    min_idx_local = np.argmin(f)
    min_f = f[min_idx_local]
    depth = (baseline - min_f)/baseline
    depth_err = np.sqrt(((e[min_idx_local]/baseline)**2) + ((min_f / baseline**2)**2)*baseline_var)

    below = f < baseline - 0.5*(baseline - min_f)
    if not np.any(below):
        return depth, depth_err, 0.0, 0.0
    idx = np.where(below)[0]
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    seg = max(segs, key=len)
    if len(seg) > 1:
        dur = (t[seg[-1]] - t[seg[0]])*24.0
        cadence_hours = np.median(np.diff(t))*24.0
        dur_err = cadence_hours*np.sqrt(2.0)
    else:
        dur = 0.0
        dur_err = 0.0
    return depth, depth_err, dur, dur_err



def filter_anomalous_transits(time, flux, flux_err, transit_indices, depth_sigma=3.0, duration_sigma=3.0):
    if len(transit_indices) < 3:
        print("too few transits for statistics")
        return transit_indices

    props = [transit_properties(time, flux, flux_err, i) for i in transit_indices]
    valid = [(i, d, de, dur, dure) for i, (d, de, dur, dure) in zip(transit_indices, props) 
             if d is not None and de is not None and d > 0 and dur > 0]
    if len(valid) < 3:
        return np.array([v[0] for v in valid])
    ids = np.array([v[0] for v in valid])
    depths = np.array([v[1] for v in valid])
    depth_errs = np.array([v[2] for v in valid])
    durs = np.array([v[3] for v in valid])
    dur_errs = np.array([v[4] for v in valid])

    def _outlier_mask(vals, errs, sigma):
        w = 1.0/np.maximum(errs, 1e-12)**2
        center = np.average(vals, weights=w)
        residuals = vals - center
        robust_scatter = 1.4826*np.median(np.abs(residuals))
        total_sigma = np.sqrt(robust_scatter**2 + errs**2)
        return np.abs(residuals) > sigma*total_sigma
    
    outlier = (_outlier_mask(depths, depth_errs, depth_sigma)
               | _outlier_mask(durs, dur_errs, duration_sigma))
    good = ids[~outlier]
    rejected = ids[outlier]
    print(f"Transit filtering: {len(transit_indices)} detected -> "
          f"{len(valid)} valid -> {len(good)} kept, {len(rejected)} rejected")
    return good



def period_calc(midtimes, midtime_errs=None):
    midtimes = np.sort(np.asarray(midtimes))
    if len(midtimes) < 2:
        return np.nan, np.nan
    p0 = np.median(np.diff(midtimes))
    cycles = np.round((midtimes - midtimes[0])/p0).astype(int)
    if midtime_errs is not None and len(midtime_errs) == len(midtimes):
        safe_errs = np.maximum(midtime_errs, 1e-10)
        w = 1.0/safe_errs**2
        coeffs, cov = np.polyfit(cycles, midtimes, 1, w=w, cov=True)
    else:
        coeffs, cov = np.polyfit(cycles, midtimes, 1, cov=True)
    return coeffs[0], np.sqrt(cov[0, 0])



def extract_and_save(time, flux, flux_err, norm_flux, norm_err, mid_i, window_hours, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    half_days = (window_hours/2)/24.0
    dip_threshold = np.percentile(norm_flux, 10)
    results = [_update_midtime(time, norm_flux, norm_err, i, half_days, dip_threshold) for i in mid_i]
    midtimes = np.array([r[0] for r in results])
    midtime_errs = np.array([r[1] for r in results])
    period, period_err = period_calc(midtimes, midtime_errs)

    for i, (det_i, true_mid, mid_err) in enumerate(zip(mid_i, midtimes, midtime_errs), start=1):
        mask = (time >= true_mid - half_days) & (time <= true_mid + half_days)
        t1 = time[mask]
        f_raw = flux[mask]
        e_raw = flux_err[mask]
        f_norm = norm_flux[mask]
        e_norm = norm_err[mask]
        _, _, dur_hours, _ = transit_properties(time, norm_flux, norm_err, det_i, window_hours=window_hours)
        dur_days = (dur_hours/24.0 if dur_hours is not None and dur_hours > 0 else window_hours/24.0)
        f_det, e_det = normalize_transit_local(t1, f_raw, e_raw, transit_duration_days=dur_days*1.5, degree=2)

        fname = os.path.join(output_dir, f'transit_{i:02d}.txt')
        with open(fname, 'w') as fh:
            fh.write(f"# Transit {i}\n")
            fh.write(f"# Period (BJD): {period:.8f}\n")
            fh.write(f"# Error Period (BJD): {period_err:.8f}\n")
            fh.write(f"# True midtime (BJD): {true_mid:.8f}\n")
            fh.write(f"# Error midtime (BJD): {mid_err:.8f}\n")
            fh.write(f"# Window ±{window_hours / 2:.2f} hours around true midtime\n")
            fh.write("# Columns: time_BJD flux_norm flux_err_norm flux_detrended flux_err_detrended\n")
            for tt, fn, en, fd, ed in zip(t1, f_norm, e_norm, f_det, e_det):
                fh.write(f"{tt:.8f} {fn:.8f} {en:.8f} {fd:.8f} {ed:.8f}\n")
        print(f"Saved transit {i}: [{t1.min():.8f}, {t1.max():.8f}]"
              f"  mid={true_mid:.8f} ± {mid_err:.6f} d -> {fname}")

    return midtimes, midtime_errs, period, period_err



def plot_timeseries(time, flux, all_transits, good_transits, rejected_transits):
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time, flux, 'k.', alpha=0.3, markersize=1)
    for k, i in enumerate(all_transits):
        plt.axvline(time[i], color='red', alpha=0.5, linestyle=':',
                    label='All detected' if k == 0 else "")
    for k, i in enumerate(good_transits):
        plt.axvline(time[i], color='green', alpha=0.8, linewidth=2,
                    label='Good transits' if k == 0 else "")
    for k, i in enumerate(rejected_transits):
        plt.axvline(time[i], color='red', alpha=0.8, linewidth=2,
                    label='Rejected' if k == 0 else "")
    plt.xlabel('Time (BJD)')
    plt.ylabel('Normalized Flux')
    plt.title('Transit Detection')
    plt.legend()
    plt.grid(alpha=0.3)
    if len(rejected_transits) > 0:
        colors = ['red', 'orange', 'purple', 'brown']
        plt.subplot(2, 1, 2)
        for k, i in enumerate(rejected_transits[:4]):
            mask = (time >= time[i] - 0.1) & (time <= time[i] + 0.1)
            plt.plot(time[mask], flux[mask] - k*0.02, 'o-',
                     color=colors[k % 4], markersize=3, alpha=0.7,
                     label=f'Rejected {k + 1}')
            plt.axvline(time[i], color=colors[k % 4], linestyle='--', alpha=0.7)
        plt.xlabel('Time (BJD)')
        plt.ylabel('Flux (offset)')
        plt.title('Rejected Transits')
        plt.legend()
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def read_transit(filename):
    data = np.loadtxt(filename, comments='#')
    midtime = period = None
    with open(filename) as f:
        for line in f:
            if line.startswith('# True midtime'):
                midtime = float(line.split(':')[1])
            elif line.startswith('# Period (BJD)'):
                period = float(line.split(':')[1])
    return period, midtime, data[:, 0], data[:, 1], data[:, 2]



def plot_transit(filename):
    period, midtime, time, flux, err = read_transit(filename)
    plt.figure(figsize=(14, 5))
    plt.errorbar(time, flux, yerr=err, fmt='o', markersize=4, capsize=2)
    plt.axvline(midtime, color='red', linestyle='--', label='Midtime')
    title = f"Transit at {midtime:.5f}"
    if period:
        title += f" | Period: {period:.5f} d"
    plt.title(title)
    plt.xlabel('Time (BJD)')
    plt.ylabel('Normalized Flux')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()




def main(fits_file, window_hours, directory, depth_sigma=3.0, duration_sigma=3.0, plot_results=True):
    time, flux, flux_err = load_data(fits_file)
    norm_flux, norm_err = normalize_flux(time, flux, flux_err)
    all_transit_ids = detect_transits(time, norm_flux)
    if len(all_transit_ids) == 0:
        print("no transits detected")
        return
    good_transit_ids = filter_anomalous_transits(time, norm_flux, norm_err, all_transit_ids, depth_sigma, duration_sigma)
    if len(good_transit_ids) == 0:
        print("no good transits after filtering")
        return
    if plot_results:
        rejected = np.setdiff1d(all_transit_ids, good_transit_ids)
        plot_timeseries(time, norm_flux, all_transit_ids, good_transit_ids, rejected)
    midtimes, midtime_errs, period, period_err = extract_and_save(
        time, flux, flux_err, norm_flux, norm_err,
        good_transit_ids, window_hours, output_dir=directory)
    print(f"\n\nTransits: {len(all_transit_ids)} detected, "
          f"{len(good_transit_ids)} saved, "
          f"{len(all_transit_ids) - len(good_transit_ids)} rejected")
    if len(good_transit_ids) >= 2:
        print(f"\nORBITAL PERIOD:")
        print(f"  {period:.8f} ± {period_err:.8f} days")
    else:
        print("not enough transits for period calculation (< 2)")





#####   example call below:

main('data\\kelt9\\MAST_2026-03-15T1619\\TESS\\tess2024223182411-s0082-0000000016740101-0278-s\\tess2024223182411-s0082-0000000016740101-0278-s_lc.fits',
     window_hours=11, directory='data\\kelt9\\KELT-9b')


####  you can load the .txt files like this:

# time0, flux0, err0, det_flux, det_flux_err = np.loadtxt('data\\kelt9\\KELT-9b\\transit_04.txt', unpack=True)

# from this you extract: (1) the time array, (2) the PDCSAP_FLUX (scaled close to 1.0 for normalization), (3) the PDCSAP_FLUX error,
## (4) the detrended flux (from 2nd degree polynomial), (5) the detrended flux error


