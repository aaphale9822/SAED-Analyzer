"""
Script to generate a clean synthetic polycrystalline SAED pattern (Gold Au FCC).
Useful for instant demonstration, automated unit tests, and interactive experimentation.
"""

import numpy as np
import cv2


def generate_synthetic_saed(
    size: int = 800,
    center: tuple = (400, 400),
    scale_nm_inv_per_px: float = 0.025,
    noise_level: float = 0.04
) -> np.ndarray:
    """Generate a realistic synthetic SAED diffraction pattern with central beam and Au FCC rings."""
    img = np.zeros((size, size), dtype=np.float32)
    y, x = np.indices((size, size))
    cx, cy = center
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # 1. Central direct beam (bright Gaussian peak)
    direct_beam = np.exp(-(r**2) / (2 * (16.0**2))) * 1.0
    diffuse_bg = 0.35 * np.exp(-r / 180.0)
    img += direct_beam + diffuse_bg

    # 2. Au FCC d-spacings in Angstroms: (111), (200), (220), (311), (222), (400)
    # d_nm = d_ang / 10.0; q = 1 / d_nm; r_px = q / scale_nm_inv_per_px
    au_rings = [
        {"d": 2.355, "intensity": 0.65, "fwhm": 4.5},
        {"d": 2.039, "intensity": 0.40, "fwhm": 4.5},
        {"d": 1.442, "intensity": 0.30, "fwhm": 5.0},
        {"d": 1.230, "intensity": 0.32, "fwhm": 5.2},
        {"d": 1.177, "intensity": 0.15, "fwhm": 5.5},
        {"d": 1.020, "intensity": 0.12, "fwhm": 5.8},
    ]

    for ring in au_rings:
        d_nm = ring["d"] / 10.0
        q = 1.0 / d_nm  # nm^-1
        r_ring = q / scale_nm_inv_per_px  # pixels
        sigma = ring["fwhm"] / 2.355

        # Ring profile
        ring_profile = ring["intensity"] * np.exp(-((r - r_ring)**2) / (2 * (sigma**2)))

        # Mild azimuthal texture / grain speckles
        theta = np.arctan2(y - cy, x - cx)
        speckle = 1.0 + 0.15 * np.sin(12 * theta) * np.cos(8 * theta)
        img += ring_profile * speckle

    # 3. Add realistic Poisson / Gaussian noise
    noise = np.random.normal(0, noise_level, (size, size))
    img = np.clip(img + noise, 0.0, 1.0)
    return img


if __name__ == "__main__":
    synthetic = generate_synthetic_saed()
    cv2.imwrite("synthetic_au_sample.png", (synthetic * 255).astype(np.uint8))
    print("Saved synthetic_au_sample.png")
