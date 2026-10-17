"""
SAED Radial Profile & Peak Analysis Module
Extracts azimuthally integrated radial profiles, fits/subtracts background,
detects diffraction rings/peaks, calculates d-spacings, and measures ring/spot metrics.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from scipy import signal, ndimage
import pandas as pd


def compute_radial_profile(
    image: np.ndarray,
    center: Tuple[float, float],
    mask: Optional[np.ndarray] = None,
    max_radius: Optional[float] = None,
    radial_bin_size: float = 1.0,
    sector_angle_range: Optional[Tuple[float, float]] = None,
    *args: Any,
    **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the azimuthally integrated radial intensity profile around the center."""
    h, w = image.shape[:2]
    cx, cy = center

    y_indices, x_indices = np.indices((h, w))
    dx = x_indices - cx
    dy = y_indices - cy
    r = np.sqrt(dx**2 + dy**2)

    valid_mask = np.ones((h, w), dtype=bool)
    if mask is not None:
        valid_mask = valid_mask & (~mask)

    if sector_angle_range is not None:
        theta = np.degrees(np.arctan2(dy, dx)) % 360.0
        start_angle, end_angle = sector_angle_range
        start_angle %= 360.0
        end_angle %= 360.0
        if start_angle <= end_angle:
            angle_mask = (theta >= start_angle) & (theta <= end_angle)
        else:
            angle_mask = (theta >= start_angle) | (theta <= end_angle)
        valid_mask = valid_mask & angle_mask

    if max_radius is None:
        # Distance to farthest corner
        max_r = np.max([
            np.hypot(cx, cy),
            np.hypot(w - cx, cy),
            np.hypot(cx, h - cy),
            np.hypot(w - cx, h - cy)
        ])
    else:
        max_r = max_radius

    # Bin radii
    nbins = int(np.ceil(max_r / radial_bin_size))
    bin_edges = np.linspace(0, max_r, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    r_flat = r[valid_mask]
    val_flat = image[valid_mask]

    # Digitise
    bin_indices = np.digitize(r_flat, bin_edges) - 1
    # Only keep points inside bins
    in_bounds = (bin_indices >= 0) & (bin_indices < nbins)
    bin_indices = bin_indices[in_bounds]
    val_flat = val_flat[in_bounds]

    count = np.bincount(bin_indices, minlength=nbins)

    # Integration mode: "mean", "max" (ideal for single-crystal/spotty SAED), or "percentile"
    integration_mode = kwargs.get("integration_mode", "Mean (Powder Rings)")
    if "Max" in integration_mode:
        profile = np.zeros(nbins, dtype=np.float32)
        # Use maximum projection across azimuth for each radial bin
        for b_idx in range(nbins):
            vals_in_bin = val_flat[bin_indices == b_idx]
            if len(vals_in_bin) > 0:
                profile[b_idx] = np.max(vals_in_bin)
    elif "Percentile" in integration_mode or "90th" in integration_mode:
        profile = np.zeros(nbins, dtype=np.float32)
        for b_idx in range(nbins):
            vals_in_bin = val_flat[bin_indices == b_idx]
            if len(vals_in_bin) > 0:
                profile[b_idx] = np.percentile(vals_in_bin, 90.0)
    else:
        # Standard azimuthal mean integration
        sum_vals = np.bincount(bin_indices, weights=val_flat, minlength=nbins)
        with np.errstate(divide='ignore', invalid='ignore'):
            profile = np.where(count > 0, sum_vals / count, 0.0)

    # Standard error of the mean in each bin
    variance_sum = np.bincount(bin_indices, weights=val_flat**2, minlength=nbins)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = profile
        var = np.maximum(0.0, np.where(count > 1, (variance_sum / count) - mean**2, 0.0))
        std_err = np.where(count > 1, np.sqrt(var / count), 0.0)

    return bin_centers, profile, std_err


def estimate_background_rolling_ball(
    profile: np.ndarray,
    radius: int = 40
) -> np.ndarray:
    """Estimate smooth diffuse background using 1D morphological opening."""
    kernel = np.ones(max(3, int(radius)))
    bg = ndimage.grey_opening(profile, structure=kernel)
    # Gaussian smooth the background
    bg = ndimage.gaussian_filter1d(bg, sigma=max(2.0, radius / 5.0))
    return bg


def find_diffraction_peaks(
    radii_pixels: np.ndarray,
    intensities: np.ndarray,
    prominence: float = 0.01,
    min_distance_px: float = 4.0,
    width_range_px: Optional[Tuple[float, float]] = None,
    rel_height: float = 0.5,
    min_radius_px: float = 18.0,
    max_radius_px: Optional[float] = None,
    *args: Any,
    **kwargs: Any
) -> Dict[str, Any]:
    """Detect peaks corresponding to diffraction rings in radial profile."""
    # Light 1D Gaussian smoothing to suppress pixel noise and connect discrete spot peaks
    smoothed = ndimage.gaussian_filter1d(intensities, sigma=1.2)

    find_kwargs: Dict[str, Any] = {
        "prominence": prominence,
        "distance": min_distance_px,
        "rel_height": rel_height
    }
    if width_range_px is not None:
        find_kwargs["width"] = width_range_px

    # Find peaks on smoothed profile
    peaks_idx, properties = signal.find_peaks(smoothed, **find_kwargs)

    if len(peaks_idx) == 0:
        return {
            "indices": np.array([], dtype=int),
            "peak_radii_px": np.array([], dtype=float),
            "intensities": np.array([], dtype=float),
            "fwhm_px": np.array([], dtype=float),
            "prominences": np.array([], dtype=float)
        }

    # Filter out peaks located inside direct beam or beyond maximum radius
    valid_mask = np.ones(len(peaks_idx), dtype=bool)
    if min_radius_px is not None:
        valid_mask = valid_mask & (radii_pixels[peaks_idx] >= min_radius_px)
    if max_radius_px is not None:
        valid_mask = valid_mask & (radii_pixels[peaks_idx] <= max_radius_px)

    peaks_idx = peaks_idx[valid_mask]
    for key in list(properties.keys()):
        val = properties[key]
        if isinstance(val, np.ndarray) and len(val) == len(valid_mask):
            properties[key] = val[valid_mask]

    if len(peaks_idx) == 0:
        return {
            "indices": np.array([], dtype=int),
            "peak_radii_px": np.array([], dtype=float),
            "intensities": np.array([], dtype=float),
            "fwhm_px": np.array([], dtype=float),
            "prominences": np.array([], dtype=float)
        }

    # Interpolate peak center for sub-pixel accuracy using parabolic fit
    subpix_radii = []
    for idx in peaks_idx:
        if 0 < idx < len(intensities) - 1:
            y0 = intensities[idx - 1]
            y1 = intensities[idx]
            y2 = intensities[idx + 1]
            denom = 2 * (2 * y1 - y0 - y2)
            if abs(denom) > 1e-6:
                delta = (y0 - y2) / denom
                delta = np.clip(delta, -0.5, 0.5)
            else:
                delta = 0.0
            r_bin = radii_pixels[1] - radii_pixels[0] if len(radii_pixels) > 1 else 1.0
            subpix_radii.append(radii_pixels[idx] + delta * r_bin)
        else:
            subpix_radii.append(radii_pixels[idx])

    subpix_radii = np.array(subpix_radii, dtype=float)
    widths = properties.get("widths", np.zeros(len(peaks_idx)))
    r_step = (radii_pixels[1] - radii_pixels[0]) if len(radii_pixels) > 1 else 1.0
    fwhm = widths * r_step

    return {
        "indices": peaks_idx,
        "peak_radii_px": subpix_radii,
        "intensities": intensities[peaks_idx],
        "fwhm_px": fwhm,
        "prominences": properties.get("prominences", np.zeros(len(peaks_idx)))
    }


def compute_spacings_and_metrics(
    peak_results: Dict[str, Any],
    calibration_scale: float,
    scale_type: str = "d_inv_nm"
) -> pd.DataFrame:
    """Calculate d-spacings (Å or nm), reciprocal radii, and peak metrics into a DataFrame."""
    radii_px = peak_results.get("peak_radii_px", np.array([]))
    if len(radii_px) == 0:
        return pd.DataFrame(columns=[
            "Ring #", "Radius (px)", "FWHM (px)", "Intensity (a.u.)",
            "q (1/nm)", "d-spacing (nm)", "d-spacing (Å)"
        ])

    intensities = peak_results.get("intensities", np.zeros(len(radii_px)))
    fwhm_px = peak_results.get("fwhm_px", np.zeros(len(radii_px)))
    prom = peak_results.get("prominences", np.zeros(len(radii_px)))

    # Compute reciprocal space radius q (in 1/nm)
    if scale_type == "d_inv_nm":
        # calibration_scale is directly nm^-1 per pixel
        q_nm_inv = radii_px * calibration_scale
    elif scale_type == "d_inv_angstrom":
        # calibration_scale is Å^-1 per pixel
        q_nm_inv = (radii_px * calibration_scale) * 10.0
    elif scale_type == "pixel_size_nm":
        # real space calibration: r_nm = radii_px * calibration_scale
        # In diffraction, this represents reciprocal distance
        q_nm_inv = radii_px * calibration_scale
    else:
        q_nm_inv = radii_px * calibration_scale

    # d = 1 / q (nm)
    with np.errstate(divide='ignore'):
        d_nm = np.where(q_nm_inv > 1e-9, 1.0 / q_nm_inv, np.nan)
        d_angstrom = d_nm * 10.0

    df = pd.DataFrame({
        "Ring #": [f"Ring {i+1}" for i in range(len(radii_px))],
        "Radius (px)": np.round(radii_px, 2),
        "FWHM (px)": np.round(fwhm_px, 2),
        "Peak Height": np.round(intensities, 4),
        "Prominence": np.round(prom, 4),
        "q (1/nm)": np.round(q_nm_inv, 4),
        "d-spacing (nm)": np.round(d_nm, 4),
        "d-spacing (Å)": np.round(d_angstrom, 4)
    })

    return df
