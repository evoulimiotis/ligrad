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
    def _fit(t, f, e):
        return np.polyfit(t, f, 1, w=1.0 / e)
    c = _fit(time, flux, flux_err)
    resid = flux - np.polyval(c, time)
    mad = np.median(np.abs(resid - np.median(resid)))
    inlier = np.abs(resid - np.median(resid)) < 3 * 1.4826 * mad
    c = _fit(time[inlier], flux[inlier], flux_err[inlier])
    trend = np.polyval(c, time)
    return flux/trend, flux_err/trend


def normalize_transit_local(time, flux, flux_err, transit_duration_days, degree=2):
    midtime = np.median(time)
    half_dur = transit_duration_days/2.0
    oot_mask = np.abs(time - midtime) > 1.5*half_dur

    if oot_mask.sum() < degree + 2:
        baseline = np.median(flux[oot_mask]) if oot_mask.sum() > 0 else np.median(flux)
        return flux/baseline, flux_err/baseline

    t0 = np.median(time)
    t_normalized = time - t0
    
    try:
        coeffs = np.polyfit(t_normalized[oot_mask], flux[oot_mask], degree, w=1.0 / np.maximum(flux_err[oot_mask], np.nanmedian(flux_err)))
        trend = np.polyval(coeffs, t_normalized)
    except:
        coeffs = np.polyfit(t_normalized[oot_mask], flux[oot_mask], degree)
        trend = np.polyval(coeffs, t_normalized)
    return flux/trend, flux_err/trend


# def detect_transits(time, norm_flux, threshold=0.994):
#     window = min(51, len(norm_flux) // 10)
#     if window % 2 == 0:
#         window += 1
#     smoothed = signal.savgol_filter(norm_flux, window, 3) if len(norm_flux) > window else norm_flux
#     baseline = np.median(smoothed[smoothed > threshold])
#     dip_mask = smoothed < baseline * threshold
#     if not np.any(dip_mask):
#         print("no transits found")
#         return np.array([])
#     dip_i = np.where(dip_mask)[0]
#     breaks = np.where(np.diff(dip_i) > 5)[0] + 1
#     groups = [g for g in np.split(dip_i, breaks) if len(g) >= 3]
#     centers = np.array([g[np.argmin(smoothed[g])] for g in groups])
#     print("found", len(centers), "transit candidates")
#     return centers

def detect_transits(time, norm_flux, sigma_threshold=3.0):
    window = min(51, len(norm_flux)//10)
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


def _update_midtime(time, flux, det_i, half_days, dip_threshold):
    det_mid = time[det_i]
    mask = (time >= det_mid - half_days) & (time <= det_mid + half_days)
    t0, f0 = time[mask], flux[mask]
    dip_i = np.where(f0 < dip_threshold)[0]
    if len(dip_i) == 0:
        return det_mid
    segs = np.split(dip_i, np.where(np.diff(dip_i) > 1)[0] + 1)
    seg = max(segs, key=len)
    return 0.5*(t0[seg[0]] + t0[seg[-1]])


def transit_properties(time, flux, midtime_idx, window_hours=24):
    midtime = time[midtime_idx]
    half = (window_hours/2)/24.0
    mask = (time >= midtime - half) & (time <= midtime + half)
    t, f = time[mask], flux[mask]
    if len(t) < 5:
        return None, None
    baseline = np.median(f)
    min_f = np.min(f)
    depth = (baseline - min_f)/baseline
    below = f < baseline - 0.5*(baseline - min_f)
    if not np.any(below):
        return depth, 0
    idx = np.where(below)[0]
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    seg = max(segs, key=len)
    dur = (t[seg[-1]] - t[seg[0]])*24 if len(seg) > 1 else 0
    return depth, dur


def filter_anomalous_transits(time, flux, transit_indices, depth_sigma=3.0, duration_sigma=3.0):
    if len(transit_indices) < 3:
        print("too few transits for statistics")
        return transit_indices
    props = [transit_properties(time, flux, i) for i in transit_indices]
    valid = [(i, d, dur) for i, (d, dur) in zip(transit_indices, props) if d is not None and dur is not None and d > 0 and dur > 0]
    if len(valid) < 3:
        return np.array([v[0] for v in valid])
    ids, depths, durs = map(np.array, zip(*valid))

    def _outlier_mask(vals, sigma):
        med = np.median(vals)
        std = 1.4826*np.median(np.abs(vals - med))
        return np.abs(vals - med) > sigma*std

    outlier = _outlier_mask(depths, depth_sigma) | _outlier_mask(durs, duration_sigma)
    good = ids[~outlier]
    rejected = ids[outlier]
    print(f"Transit filtering: {len(transit_indices)} detected -> "
          f"{len(valid)} valid -> {len(good)} kept, {len(rejected)} rejected")
    return good


def period_calc(midtimes):
    midtimes = np.sort(np.asarray(midtimes))
    if len(midtimes) < 2:
        return np.nan, np.nan
    p0 = np.median(np.diff(midtimes))
    cycles = np.round((midtimes - midtimes[0])/p0).astype(int)
    coeffs, cov = np.polyfit(cycles, midtimes, 1, cov=True)
    return coeffs[0], np.sqrt(cov[0, 0])


def extract_and_save(time, flux, flux_err, norm_flux, norm_err, mid_indices, window_hours, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    half_days = (window_hours/2)/24.0
    dip_threshold = np.percentile(norm_flux, 10)
    midtimes = [_update_midtime(time, norm_flux, i, half_days, dip_threshold) for i in mid_indices]
    period, period_err = period_calc(midtimes)

    for i, (id, true_mid) in enumerate(zip(mid_indices, midtimes), start=1):
        mask = (time >= true_mid - half_days) & (time <= true_mid + half_days)
        t1 = time[mask]
        f_raw = flux[mask]
        e_raw = flux_err[mask]
        f_norm = norm_flux[mask]
        e_norm = norm_err[mask]
        _, dur_hours = transit_properties(time, norm_flux, id, window_hours=window_hours)
        dur_days = dur_hours/24.0 if dur_hours is not None and dur_hours > 0 else window_hours/24.0

        f_det, e_det = normalize_transit_local(t1, f_raw, e_raw, transit_duration_days=dur_days*1.5, degree=2)

        fname = os.path.join(output_dir, f'transit_{i:02d}.txt')
        with open(fname, 'w') as fh:
            fh.write(f"# Transit {i}\n")
            fh.write(f"# Period (BJD): {period:.8f}\n")
            fh.write(f"# Error Period (BJD): {period_err:.8f}\n")
            fh.write(f"# True midtime (BJD): {true_mid:.8f}\n")
            fh.write(f"# Window ±{window_hours/2:.2f} hours around true midtime\n")
            fh.write("# Columns: time_BJD flux_norm flux_err_norm flux_detrended flux_err_detrended\n")
            for tt, fn, en, fd, ed in zip(t1, f_norm, e_norm, f_det, e_det):
                fh.write(f"{tt:.8f} {fn:.8f} {en:.8f} {fd:.8f} {ed:.8f}\n")
        print(f"Saved transit {i}: [{t1.min():.8f}, {t1.max():.8f}]"
              f"  mid={true_mid:.8f} -> {fname}")
    return midtimes, period, period_err


def plot_timeseries(time, flux, all_transits, good_transits, rejected_transits):
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time, flux, 'k.', alpha=0.3, markersize=1)
    for i, id in enumerate(all_transits):
        plt.axvline(time[id], color='red', alpha=0.5, linestyle=':', label='All detected' if i == 0 else "")
    for i, id in enumerate(good_transits):
        plt.axvline(time[id], color='green', alpha=0.8, linewidth=2, label='Good transits' if i == 0 else "")
    for i, id in enumerate(rejected_transits):
        plt.axvline(time[id], color='red', alpha=0.8, linewidth=2, label='Rejected' if i == 0 else "")
    plt.xlabel('Time (BJD)')
    plt.ylabel('Normalized Flux')
    plt.title('Transit Detection and Filtering')
    plt.legend()
    plt.grid(alpha=0.3)
    if len(rejected_transits) > 0:
        colors = ['red', 'orange', 'purple', 'brown']
        plt.subplot(2, 1, 2)
        for i, idx in enumerate(rejected_transits[:4]):
            mask = (time >= time[idx] - 0.1) & (time <= time[idx] + 0.1)
            plt.plot(time[mask], flux[mask] - i * 0.02, 'o-', color=colors[i % 4], markersize=3, alpha=0.7, label=f'Rejected {i+1}')
            plt.axvline(time[idx], color=colors[i % 4], linestyle='--', alpha=0.7)
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
    all_transit_indices = detect_transits(time, norm_flux)
    if len(all_transit_indices) == 0:
        print("no transits detected")
        return
    good_transit_indices = filter_anomalous_transits(time, norm_flux, all_transit_indices, depth_sigma, duration_sigma)
    if len(good_transit_indices) == 0:
        print("no good transits after filtering")
        return
    if plot_results:
        rejected = np.setdiff1d(all_transit_indices, good_transit_indices)
        plot_timeseries(time, norm_flux, all_transit_indices, good_transit_indices, rejected)
    midtimes, period, period_err = extract_and_save(time, flux, flux_err, norm_flux, norm_err, good_transit_indices,
                                                    window_hours, output_dir=directory)
    print(f"\n\nTransits: {len(all_transit_indices)} detected, "
          f"{len(good_transit_indices)} saved, "
          f"{len(all_transit_indices) - len(good_transit_indices)} rejected")
    if len(good_transit_indices) >= 2:
        print(f"\nORBITAL PERIOD:")
        print(f"  {period:.8f} ± {period_err:.8f} days")
    else:
        print("not enough transits for period calculation (< 2)")
        





#####   example call below:

# main('data\\kelt9\\MAST_2026-03-15T1619\\TESS\\tess2024223182411-s0082-0000000016740101-0278-s\\tess2024223182411-s0082-0000000016740101-0278-s_lc.fits', 
#      window_hours=13, directory='data\\kelt9\\KELT-9b')



####  you can load the .txt files like this:

time0, flux0, err0, det_flux, det_flux_err = np.loadtxt('data\\kelt9\\KELT-9b\\transit_04.txt', unpack=True)

# from this you extract: (1) the time array, (2) the PDCSAP_FLUX (scaled close to 1.0 for normalization), (3) the PDCSAP_FLUX error,
## (4) the detrended flux (from 2nd degree polynomial), (5) the detrended flux error


