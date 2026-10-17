"""
Comprehensive Unit Tests for SAED Analyzer Modules
Tests center detection, radial profile integration, peak extraction,
CIF parsing, and synthetic pattern generation.
"""

import unittest
import numpy as np
from modules.image_processing import (
    crop_image, adjust_contrast, detect_center_centroid,
    detect_center_hough, detect_center_minmax, auto_detect_center
)
from modules.radial_profile import (
    compute_radial_profile, estimate_background_rolling_ball,
    find_diffraction_peaks, compute_spacings_and_metrics
)
from modules.cif_database import (
    calculate_d_cubic, calculate_d_general, load_cif,
    match_experimental_to_phase, get_standard_materials, create_phase_from_standard
)
from modules.synthetic_data import generate_synthetic_saed


class TestSAEDModules(unittest.TestCase):

    def setUp(self):
        self.img_size = 300
        self.true_center = (150.0, 150.0)
        self.test_img = generate_synthetic_saed(
            size=self.img_size,
            center=self.true_center,
            scale_nm_inv_per_px=0.03,
            noise_level=0.01
        )

    def test_crop_image(self):
        cropped = crop_image(self.test_img, 50, 50, 250, 250)
        self.assertEqual(cropped.shape, (200, 200))

    def test_contrast_adjustment(self):
        adj = adjust_contrast(self.test_img, contrast_low=1.0, contrast_high=99.0, gamma=1.2)
        self.assertTrue(0.0 <= np.min(adj) <= np.max(adj) <= 1.0)

    def test_center_detection(self):
        # 1. Centroid
        cx, cy, _ = auto_detect_center(self.test_img, method="Intensity Centroid")
        self.assertAlmostEqual(cx, self.true_center[0], delta=5.0)
        self.assertAlmostEqual(cy, self.true_center[1], delta=5.0)

        # 2. Max Peak
        cx_m, cy_m, _ = auto_detect_center(self.test_img, method="Max Intensity Peak")
        self.assertAlmostEqual(cx_m, self.true_center[0], delta=2.0)
        self.assertAlmostEqual(cy_m, self.true_center[1], delta=2.0)

    def test_radial_profile_and_peaks(self):
        radii, profile, stderr = compute_radial_profile(
            self.test_img,
            center=self.true_center,
            radial_bin_size=1.0
        )
        self.assertTrue(len(radii) > 0)
        self.assertEqual(len(radii), len(profile))

        # Detect peaks
        peaks = find_diffraction_peaks(radii, profile, prominence=0.03, min_distance_px=10.0)
        self.assertTrue(len(peaks["indices"]) >= 3)

        # Compute d-spacings
        df = compute_spacings_and_metrics(peaks, calibration_scale=0.03, scale_type="d_inv_nm")
        self.assertIn("d-spacing (Å)", df.columns)
        self.assertTrue(len(df) >= 3)

    def test_cif_parser_and_matching(self):
        # Cubic d-spacing test
        # For Au a=4.078, (1 1 1) -> 4.078 / sqrt(3) = 2.3544 A
        d111 = calculate_d_cubic(4.078, 1, 1, 1)
        self.assertAlmostEqual(d111, 2.3544, places=3)

        # General triclinic/cubic matching test
        d_gen = calculate_d_general(4.078, 4.078, 4.078, 90, 90, 90, 1, 1, 1)
        self.assertAlmostEqual(d_gen, 2.3544, places=3)

        # Match Gold standard against experimental d-spacing
        phase = create_phase_from_standard("Gold (Au - FCC, a=4.078 Å)")
        exp_d = [2.35, 2.04, 1.44]
        match_df = match_experimental_to_phase(exp_d, phase, tolerance_pct=3.0)
        self.assertEqual(len(match_df), 3)
        self.assertTrue(all(status == "Matched" for status in match_df["Status"]))


if __name__ == "__main__":
    unittest.main()
