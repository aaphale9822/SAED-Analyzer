# app.py
"""
SAED Ring & Spot Diffraction Pattern Analyzer
A production-ready Streamlit application for transmission electron microscopy (TEM)
Selected Area Electron Diffraction (SAED) quantitative pattern analysis.

Features:
1. Image Upload & Crop: Upload diffraction image with interactive cropping tools & contrast adjustment.
2. Auto-Center: Automatically detect direct beam/center using circular Hough transform or intensity centroid.
3. Full Pattern Analysis: Azimuthally integrated radial profile extraction, d-spacing measurement, and ring metrics.
4. CIF Database Matching: Parse local CIF files (using pymatgen/gemmi with robust fallback) to calculate theoretical peaks and match experimental d-spacings for phase identification.
"""

import sys
import os
import io
import json
from typing import Optional, List, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
import cv2

# Add current directory to path for modular imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from modules.image_processing import (
    load_image_bytes,
    adjust_contrast,
    crop_image,
    auto_detect_center,
    detect_2d_diffraction_spots
)
from modules.radial_profile import (
    compute_radial_profile,
    estimate_background_rolling_ball,
    find_diffraction_peaks,
    compute_spacings_and_metrics
)
from modules.cif_database import (
    load_cif,
    match_experimental_to_phase,
    get_standard_materials,
    create_phase_from_standard,
    CIFPhase
)
from modules.plotting import (
    plot_saed_pattern_with_rings,
    plot_radial_profile_with_peaks
)
from modules.synthetic_data import generate_synthetic_saed
from modules.report_generator import (
    generate_publication_markdown_table,
    generate_publication_latex_table,
    generate_publication_narrative
)
from modules.mixture_analysis import assign_rings_to_phases
from modules.theory import render_theory
from modules.phase_analysis import (
    run_all_phases_comparison,
    check_systematic_absences,
    find_discriminating_rings,
    generate_phase_narrative,
)



# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SAED Diffraction Pattern Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "raw_image": None,
        "processed_image": None,
        "cropped_image": None,
        "image_filename": "sample_saed.png",
        "crop_box": None,
        "center_x": 0.0,
        "center_y": 0.0,
        "center_method": "Intensity Centroid",
        "calibration_scale": 0.025,  # nm^-1 per pixel
        "calibration_unit": "nm⁻¹ / pixel",
        "radial_profile_radii": None,
        "radial_profile_intensity": None,
        "radial_profile_bg": None,
        "peak_results": None,
        "metrics_df": None,
        "cif_phases": {},
        "selected_phase_name": None,
        "matching_df": None,
        "leaderboard_df": None,
        "discriminating_rings": [],

        # Microscope & Publication Metadata
        "microscope_voltage_kv": 200.0,
        "microscope_camera_length_mm": 500.0,
        "microscope_cal_reference": "Standard Polycrystalline Au (Gold)",
        "microscope_zone_axis": "Polycrystalline ring pattern (Debye-Scherrer rings)",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# -----------------------------------------------------------------------------
# Sidebar: Global Calibration & Quick Presets
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/atom.png", width=64)
    st.markdown("### ⚙️ Calibration & Controls")

    cal_mode = st.radio(
        "Reciprocal Space Calibration",
        ["Direct Scale (nm⁻¹/px)", "TEM Accelerating Voltage & Camera Length", "Known Standard Ring"],
        index=0
    )

    if cal_mode == "Direct Scale (nm⁻¹/px)":
        scale = st.number_input(
            "Scale Factor (nm⁻¹ / pixel)",
            min_value=0.0001,
            max_value=5.0,
            value=float(st.session_state["calibration_scale"]),
            step=0.001,
            format="%.5f"
        )
        st.session_state["calibration_scale"] = scale
        st.session_state["calibration_unit"] = "nm⁻¹ / pixel"

    elif cal_mode == "TEM Accelerating Voltage & Camera Length":
        ht_kv = st.number_input("Accelerating Voltage (kV)", min_value=20.0, max_value=1000.0, value=float(st.session_state.get("microscope_voltage_kv", 200.0)), step=10.0)
        camera_length_mm = st.number_input("Camera Length L (mm)", min_value=50.0, max_value=3000.0, value=float(st.session_state.get("microscope_camera_length_mm", 500.0)), step=25.0)
        pixel_size_um = st.number_input("Camera Pixel Size (µm)", min_value=1.0, max_value=50.0, value=14.0, step=0.5)

        st.session_state["microscope_voltage_kv"] = ht_kv
        st.session_state["microscope_camera_length_mm"] = camera_length_mm

        # Relativistic de Broglie wavelength: lambda = h / sqrt(2*m0*e*V*(1 + e*V/(2*m0*c^2)))
        # lambda (Angstroms) approx: 12.2643 / sqrt(V*(1 + 0.978466e-6 * V))
        wl_angstrom = 12.2643 / np.sqrt(ht_kv * 1000.0 * (1.0 + 0.978466e-6 * ht_kv * 1000.0))
        wl_nm = wl_angstrom / 10.0

        # R = L * lambda / d  =>  (1/d) [nm^-1] = R [pixels] * (pixel_size_um * 1e-3 mm) / (lambda_nm * L_mm)
        # scale [nm^-1 / px] = (pixel_size_um * 1e-3) / (wl_nm * camera_length_mm)
        derived_scale = (pixel_size_um * 1e-3) / (wl_nm * camera_length_mm)
        st.session_state["calibration_scale"] = derived_scale
        st.session_state["calibration_unit"] = "nm⁻¹ / pixel"

        st.caption(f"⚡ Electron Wavelength: **{wl_angstrom:.4f} Å** ({wl_nm:.5f} nm)")
        st.success(f"Derived Scale: **{derived_scale:.5f} nm⁻¹ / px**")

    else:  # Known Standard Ring
        std_dict = get_standard_materials()
        selected_std = st.selectbox("Select Standard", list(std_dict.keys()))
        std_d_val = st.number_input("Known Ring d-spacing (Å)", min_value=0.5, max_value=10.0, value=2.355, step=0.01)
        measured_r_px = st.number_input("Measured Ring Radius (pixels)", min_value=1.0, max_value=2000.0, value=170.0, step=1.0)
        # d_nm = d_ang / 10.0; q = 1 / d_nm; scale = q / r_px
        std_scale = (10.0 / std_d_val) / measured_r_px
        st.session_state["calibration_scale"] = std_scale
        st.session_state["calibration_unit"] = "nm⁻¹ / pixel"
        st.success(f"Calibrated Scale: **{std_scale:.5f} nm⁻¹ / px**")

    st.markdown("---")
    st.markdown("### 🧪 Quick Demonstration")
    if st.button("Load Synthetic Au (Gold) Pattern", use_container_width=True):
        syn_img = generate_synthetic_saed(size=700, center=(350, 350), scale_nm_inv_per_px=0.025)
        st.session_state["raw_image"] = syn_img
        st.session_state["processed_image"] = syn_img.copy()
        st.session_state["cropped_image"] = syn_img.copy()
        st.session_state["image_filename"] = "synthetic_gold_fcc.png"
        st.session_state["center_x"] = 350.0
        st.session_state["center_y"] = 350.0
        st.session_state["calibration_scale"] = 0.025
        st.success("Loaded synthetic Gold (Au) test pattern!")
        st.rerun()


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🔬 SAED Diffraction Pattern Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Quantitative Ring & Spot Profiling, Sub-pixel Auto-centering, and CIF Crystal Phase Identification</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab_upload, tab_center, tab_profile, tab_cif, tab_report, tab_export, tab_theory = st.tabs([
    "1️⃣ Image Upload & Crop",
    "2️⃣ Auto-Center Beam",
    "3️⃣ Full Pattern & Radial Profile",
    "4️⃣ CIF Database & Phase Matching",
    "5️⃣ 📄 Publication Report Generator",
    "6️⃣ Export Results",
    "7️⃣ 📚 Theory & Equations"
])


def _filter_spots(img, spots, center, max_spots=60, min_sigma=5.0, min_sep_px=7.0):
    """Keep only genuine spots: pixels that stand out from the rest of their own ring.

    The raw detector keeps every pixel above a global threshold, which on a ring
    pattern means thousands of points lying on the rings. Here every candidate is
    scored by how far it rises above the typical level (median) of its own radius,
    in units of that radius' robust spread (MAD). Weak ones are dropped, neighbours of
    an already-kept spot (within `min_sep_px`) are merged away, and only the strongest
    `max_spots` are returned. A smooth ring therefore yields no spots at all.
    """
    if not spots:
        return []
    g = np.asarray(img, dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    h, w = g.shape
    cx, cy = center
    yy, xx = np.indices((h, w))
    rb = np.hypot(xx - cx, yy - cy).astype(int).ravel()
    df = pd.DataFrame({"r": rb, "v": g.ravel()})
    # leave the dark beam stop / masked areas out of the ring statistics
    df = df[df["v"] > float(g.min()) + 0.02 * float(g.max() - g.min())].copy()
    med = df.groupby("r")["v"].median()
    df["dev"] = (df["v"] - df["r"].map(med)).abs()
    mad = df.groupby("r")["dev"].median() * 1.4826
    floor = 0.03 * float(g.max() - g.min())          # ignore tiny absolute excursions
    scored = []
    for sp in spots:
        ix = int(np.clip(round(sp["x"]), 0, w - 1))
        iy = int(np.clip(round(sp["y"]), 0, h - 1))
        rr = int(np.hypot(ix - cx, iy - cy))
        if rr not in med.index or rr not in mad.index:
            continue
        excess = float(g[iy, ix] - med[rr])
        z = excess / (float(mad[rr]) + 1e-6)
        if z > min_sigma and excess > floor:
            scored.append((z, sp))
    scored.sort(key=lambda t: t[0], reverse=True)
    kept, kept_xy = [], np.empty((0, 2))
    for _, sp in scored:
        if len(kept) >= max_spots:
            break
        p = np.array([[sp["x"], sp["y"]]])
        if len(kept_xy) and np.min(np.hypot(*(kept_xy - p).T)) < min_sep_px:
            continue
        kept.append(sp)
        kept_xy = np.vstack([kept_xy, p])
    return kept


# =============================================================================
# TAB 1: Image Upload & Crop
# =============================================================================
with tab_upload:
    st.subheader("📁 Upload SAED Image & Interactive Adjustments")

    col_up1, col_up2 = st.columns([1, 1])

    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload TEM Diffraction Pattern",
            type=["png", "jpg", "jpeg", "tif", "tiff", "dm3", "dm4"],
            help="Supports standard formats as well as DigitalMicrograph .dm3/.dm4 files."
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            img = load_image_bytes(file_bytes, uploaded_file.name)
            st.session_state["raw_image"] = img
            st.session_state["image_filename"] = uploaded_file.name
            if st.session_state["center_x"] == 0 and st.session_state["center_y"] == 0:
                h, w = img.shape[:2]
                st.session_state["center_x"] = w / 2.0
                st.session_state["center_y"] = h / 2.0

        st.markdown("#### Contrast & Enhancement")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            p_low = st.slider("Min Percentile Cutoff", 0.0, 10.0, 0.5, step=0.1)
        with col_c2:
            p_high = st.slider("Max Percentile Cutoff", 90.0, 100.0, 99.8, step=0.1)

        gamma = st.slider("Gamma Correction (Enhance Weak Rings)", 0.1, 3.0, 1.0, step=0.05)
        invert_col = st.checkbox("Invert Color (Dark on Light)", value=False)

        if st.session_state["raw_image"] is not None:
            adj = adjust_contrast(
                st.session_state["raw_image"],
                contrast_low=p_low,
                contrast_high=p_high,
                gamma=gamma,
                invert=invert_col
            )
            st.session_state["processed_image"] = adj

            st.markdown("#### Interactive Crop")
            h, w = adj.shape[:2]
            enable_crop = st.checkbox("Enable Region-of-Interest (ROI) Cropping", value=False)
            if enable_crop:
                col_cr1, col_cr2 = st.columns(2)
                with col_cr1:
                    crop_x = st.slider("X Range (Left to Right)", 0, w, (0, w), step=1)
                with col_cr2:
                    crop_y = st.slider("Y Range (Top to Bottom)", 0, h, (0, h), step=1)

                cropped = crop_image(adj, crop_x[0], crop_y[0], crop_x[1], crop_y[1])
                st.session_state["cropped_image"] = cropped
                st.session_state["crop_box"] = (crop_x[0], crop_y[0], crop_x[1], crop_y[1])
                # Shift center if previously within bounds
                if st.button("Apply Crop as Active Pattern"):
                    st.session_state["processed_image"] = cropped
                    st.session_state["center_x"] = max(0.0, st.session_state["center_x"] - crop_x[0])
                    st.session_state["center_y"] = max(0.0, st.session_state["center_y"] - crop_y[0])
                    st.success("Cropped region applied!")
                    st.rerun()
            else:
                st.session_state["cropped_image"] = adj
                st.session_state["crop_box"] = None

    with col_up2:
        if st.session_state["processed_image"] is not None:
            active_img = st.session_state["cropped_image"] if st.session_state["cropped_image"] is not None else st.session_state["processed_image"]
            h, w = active_img.shape[:2]
            st.caption(f"Image Dimensions: **{w} × {h} pixels** | Dynamic Range: [{np.min(active_img):.2f}, {np.max(active_img):.2f}]")

            fig_img = plot_saed_pattern_with_rings(
                active_img,
                center=None,
                crop_box=st.session_state["crop_box"],
                color_scale="Viridis" if not invert_col else "gray"
            )
            fig_img.update_layout(title="Active Pattern Preview")
            st.plotly_chart(fig_img, use_container_width=True)
        else:
            st.info("👈 Please upload an image or click **'Load Synthetic Au (Gold) Pattern'** in the sidebar to begin.")


# =============================================================================
# TAB 2: Auto-Center Beam
# =============================================================================
with tab_center:
    st.subheader("🎯 Direct Beam & Pattern Center Detection")
    st.markdown("Accurate determination of the unscattered central beam is critical for radial integration.")

    if st.session_state["processed_image"] is None:
        st.warning("Please upload or load an image in Tab 1 first.")
    else:
        active_img = st.session_state["processed_image"]
        h, w = active_img.shape[:2]

        col_ct1, col_ct2 = st.columns([1, 2])

        with col_ct1:
            st.markdown("#### Detection Algorithm")
            method = st.selectbox(
                "Center Detection Method",
                ["Intensity Centroid", "Circular Hough Transform", "Max Intensity Peak", "Manual Coords"],
                index=0
            )

            if method == "Intensity Centroid":
                thresh_pct = st.slider("Intensity Percentile Cutoff", 80.0, 99.9, 98.0, step=0.1)
                sigma = st.slider("Gaussian Pre-smoothing (σ)", 0.5, 10.0, 2.0, step=0.5)
                if st.button("Auto-Detect Center (Centroid)", use_container_width=True, type="primary"):
                    cx, cy, details = auto_detect_center(
                        active_img,
                        method="Intensity Centroid",
                        threshold_percentile=thresh_pct,
                        sigma=sigma
                    )
                    st.session_state["center_x"] = cx
                    st.session_state["center_y"] = cy
                    st.session_state["center_method"] = method
                    st.success(f"Detected Center: ({cx:.2f}, {cy:.2f})")

            elif method == "Circular Hough Transform":
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    min_r = st.number_input("Min Radius (px)", min_value=5, max_value=300, value=20)
                with col_h2:
                    max_r = st.number_input("Max Radius (px)", min_value=10, max_value=500, value=min(h, w) // 4)

                param1 = st.slider("Canny Edge Upper Threshold", 20.0, 200.0, 50.0)
                param2 = st.slider("Hough Accumulator Threshold", 10.0, 100.0, 30.0)

                if st.button("Auto-Detect Center (Hough)", use_container_width=True, type="primary"):
                    cx, cy, details = auto_detect_center(
                        active_img,
                        method="Circular Hough Transform",
                        min_radius=min_r,
                        max_radius=max_r,
                        param1=param1,
                        param2=param2
                    )
                    st.session_state["center_x"] = cx
                    st.session_state["center_y"] = cy
                    st.session_state["center_method"] = method
                    if details.get("hough_fallback"):
                        st.warning(f"No distinct circle found with Hough; fell back to centroid: ({cx:.2f}, {cy:.2f})")
                    else:
                        st.success(f"Hough Ring Center Detected: ({cx:.2f}, {cy:.2f}) with radius {details.get('detected_radius', 0):.1f} px")

            elif method == "Max Intensity Peak":
                smooth_s = st.slider("Gaussian Smoothing σ", 0.5, 15.0, 3.0)
                if st.button("Auto-Detect (Max Peak)", use_container_width=True, type="primary"):
                    cx, cy, details = auto_detect_center(active_img, method="Max Intensity Peak", sigma=smooth_s)
                    st.session_state["center_x"] = cx
                    st.session_state["center_y"] = cy
                    st.session_state["center_method"] = method
                    st.success(f"Max Peak Center Detected: ({cx:.2f}, {cy:.2f})")

            st.markdown("#### Fine-tune Center Coordinates")
            curr_cx = float(st.session_state["center_x"])
            curr_cy = float(st.session_state["center_y"])

            col_fx, col_fy = st.columns(2)
            with col_fx:
                new_cx = st.number_input("X Center (px)", min_value=0.0, max_value=float(w), value=curr_cx, step=0.5)
            with col_fy:
                new_cy = st.number_input("Y Center (px)", min_value=0.0, max_value=float(h), value=curr_cy, step=0.5)

            if new_cx != curr_cx or new_cy != curr_cy:
                st.session_state["center_x"] = new_cx
                st.session_state["center_y"] = new_cy

            st.info(f"Active Center: **X = {st.session_state['center_x']:.2f} px, Y = {st.session_state['center_y']:.2f} px**")

        with col_ct2:
            fig_center = plot_saed_pattern_with_rings(
                active_img,
                center=(st.session_state["center_x"], st.session_state["center_y"]),
                ring_radii_px=[50, 100, 150],  # alignment guide circles
                ring_labels=["Guide 50px", "Guide 100px", "Guide 150px"],
                color_scale="Viridis"
            )
            fig_center.update_layout(title="Center Verification & Alignment Crosshair")
            st.plotly_chart(fig_center, use_container_width=True)


# =============================================================================
# TAB 3: Full Pattern & Radial Profile
# =============================================================================
with tab_profile:
    st.subheader("📈 Radial Intensity Integration & d-Spacing Measurement")

    if st.session_state["processed_image"] is None:
        st.warning("Please upload or load an image first.")
    else:
        active_img = st.session_state["processed_image"]
        cx = float(st.session_state["center_x"])
        cy = float(st.session_state["center_y"])
        cal_scale = float(st.session_state["calibration_scale"])

        col_pr1, col_pr2 = st.columns([1, 2])

        with col_pr1:
            st.markdown("#### Integration Settings")
            bin_size = st.slider("Radial Bin Step (pixels)", 0.5, 3.0, 1.0, step=0.5)
            # Default max radius to distance to nearest/farthest image border
            default_max_r = int(min(cx, cy, w - cx, h - cy)) if min(cx, cy, w - cx, h - cy) > 20 else int(max(h, w) // 2)
            max_r = st.slider("Max Integration Radius (pixels)", 30, int(max(h, w)), min(default_max_r, 350), step=10)

            st.markdown("#### Azimuthal Sector Masking (Optional)")
            use_sector = st.checkbox("Sector / Wedge Integration (Filter beamstop)", value=False)
            if use_sector:
                angles = st.slider("Angular Range (degrees)", 0, 360, (0, 360), step=5)
                sector_range = (angles[0], angles[1])
            else:
                sector_range = None

            st.markdown("#### Integration Method (Crucial for Spotty Patterns)")
            integ_mode = st.radio(
                "Profile Projection Type",
                [
                    "Maximum Azimuthal Intensity (Best for Spotty / Single-Crystal Patterns)",
                    "Mean Azimuthal Intensity (Best for Continuous Powder Rings)",
                    "90th Percentile Intensity (Noise-Robust Spot Projection)"
                ],
                index=2,
                help="Mean integration dilutes isolated bright diffraction spots into background noise across 360°. Maximum projection preserves sharp spot intensities!"
            )

            st.markdown("#### Direct Beam Exclusion & Peak Criteria")
            direct_beam_mask_r = st.slider(
                "Exclude Central Direct Beam / Halo Radius (pixels)",
                min_value=0.0,
                max_value=float(min(80, max_r - 10)),
                value=30.0,
                step=1.0,
                help="Diffraction rings cannot exist inside the unscattered direct beam. Exclude this region to prevent matching the beam halo."
            )
            prominence = st.slider(
                "Peak Prominence Sensitivity (Lower = Detect Weak Rings)",
                0.001, 0.100, 0.010, step=0.001, format="%.3f",
                help="Lower = more (weaker) rings detected. Raise it if noise bumps are marked as rings; lower it (0.004-0.008) if real faint rings are missing."
            )
            min_dist = st.slider("Min Peak Separation (pixels)", 2.0, 30.0, 6.0, step=1.0)
            sub_bg = st.checkbox("Subtract Diffuse Background (Rolling Ball)", value=True)
            bg_radius = st.slider("Background Rolling Window", 10, 100, 35, step=5)

            detect_spots_2d = st.checkbox("Also Run 2D Discrete Spot Detection (Mark Individual Reflections)", value=True)
            spot_thresh = st.slider("2D Spot Threshold (% of intensity)", 2.0, 90.0, 30.0, step=1.0) if detect_spots_2d else 30.0
            max_spots_show = st.slider("Max spots to display (strongest first)", 10, 300, 60, step=10) if detect_spots_2d else 60
            spot_sigma = st.slider("Spot significance (σ above its ring)", 2.0, 15.0, 5.0, step=0.5) if detect_spots_2d else 5.0

            if st.button("Run Radial Integration & Peak Detection", type="primary", use_container_width=True):
                radii, profile, stderr = compute_radial_profile(
                    active_img,
                    center=(cx, cy),
                    max_radius=float(max_r),
                    radial_bin_size=bin_size,
                    sector_angle_range=sector_range,
                    integration_mode=integ_mode
                )
                st.session_state["radial_profile_radii"] = radii
                st.session_state["radial_profile_intensity"] = profile

                if sub_bg:
                    bg = estimate_background_rolling_ball(profile, radius=bg_radius)
                    st.session_state["radial_profile_bg"] = bg
                    target_profile = np.maximum(0.0, profile - bg)
                else:
                    st.session_state["radial_profile_bg"] = None
                    target_profile = profile

                peaks = find_diffraction_peaks(
                    radii,
                    target_profile,
                    prominence=prominence,
                    min_distance_px=min_dist,
                    min_radius_px=float(direct_beam_mask_r),
                    max_radius_px=float(max_r)
                )
                st.session_state["peak_results"] = peaks

                # 2D Spot Detection
                if detect_spots_2d:
                    spots_2d = detect_2d_diffraction_spots(
                        active_img,
                        center=(cx, cy),
                        min_radius_px=float(direct_beam_mask_r),
                        max_radius_px=float(max_r),
                        threshold_rel=spot_thresh / 100.0,
                        min_distance_px=int(min_dist)
                    )
                    st.session_state["spots_raw_count"] = len(spots_2d)
                    spots_2d = _filter_spots(active_img, spots_2d, (cx, cy), max_spots=int(max_spots_show), min_sigma=float(spot_sigma))
                    st.session_state["detected_spots_2d"] = spots_2d
                else:
                    st.session_state["detected_spots_2d"] = []

                # Calculate metrics DataFrame
                metrics_df = compute_spacings_and_metrics(peaks, calibration_scale=cal_scale, scale_type="d_inv_nm")
                st.session_state["metrics_df"] = metrics_df
                st.success(f"Detected {len(peaks['indices'])} diffraction rings and {len(st.session_state.get('detected_spots_2d', []))} individual spots (strongest of {st.session_state.get('spots_raw_count', 0)} candidates)!")

        with col_pr2:
            radii = st.session_state["radial_profile_radii"]
            profile = st.session_state["radial_profile_intensity"]
            bg = st.session_state["radial_profile_bg"]
            peaks = st.session_state["peak_results"]

            if radii is not None and profile is not None:
                fig_prof = plot_radial_profile_with_peaks(
                    radii=radii,
                    profile=profile,
                    background=bg,
                    peaks_dict=peaks,
                    x_units="Radius (pixels)",
                    calibration_scale=cal_scale
                )
                st.plotly_chart(fig_prof, use_container_width=True)
            else:
                st.info("Click **'Run Radial Integration & Peak Detection'** to compute the 1D diffraction profile.")

        # Show Table of detected rings and metrics
        st.markdown("---")
        st.subheader("📋 Experimental Diffraction Ring Metrics")
        if st.session_state["metrics_df"] is not None and not st.session_state["metrics_df"].empty:
            df = st.session_state["metrics_df"]
            st.dataframe(df, use_container_width=True)

            # Overlay rings on 2D pattern
            col_ov1, col_ov2 = st.columns([1, 1])
            with col_ov1:
                st.markdown("#### 2D Pattern Overlay (Rings & Spots)")
                ring_radii = peaks["peak_radii_px"] if peaks is not None else []
                ring_names = [f"{row['Ring #']}: d={row['d-spacing (Å)']}Å" for _, row in df.iterrows()]
                fig_overlay = plot_saed_pattern_with_rings(
                    active_img,
                    center=(cx, cy),
                    ring_radii_px=list(ring_radii),
                    ring_labels=ring_names,
                    spot_markers=st.session_state.get("detected_spots_2d", []),
                    color_scale="gray"
                )
                st.plotly_chart(fig_overlay, use_container_width=True)

            with col_ov2:
                st.markdown("#### Summary Ring Statistics")
                for _, row in df.iterrows():
                    st.markdown(
                        f"""
                        <div class="metric-card" style="margin-bottom: 8px;">
                            <strong>{row['Ring #']}</strong>: Radius = <code>{row['Radius (px)']} px</code> |
                            <strong>d-spacing = {row['d-spacing (Å)']} Å</strong> ({row['d-spacing (nm)']} nm) |
                            FWHM = {row['FWHM (px)']} px | Intensity = {row['Peak Height']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Show 2D individual spot coordinates if detected
                spots_list = st.session_state.get("detected_spots_2d", [])
                if spots_list:
                    with st.expander(f"📍 View Individual Spot Coordinates ({len(spots_list)} spots detected)"):
                        sp_df = pd.DataFrame(spots_list)
                        sp_df["d-spacing (Å)"] = np.round((10.0 / (sp_df["radius_px"] * cal_scale)), 4)
                        st.dataframe(sp_df[["radius_px", "angle_deg", "d-spacing (Å)", "intensity", "x", "y"]].rename(columns={
                            "radius_px": "Radius (px)",
                            "angle_deg": "Angle θ (°)",
                            "intensity": "Intensity (a.u.)",
                            "x": "X (px)",
                            "y": "Y (px)"
                        }), height=200, use_container_width=True)
        else:
            st.caption("No peaks currently identified. Run integration above.")


# =============================================================================
# TAB 4: CIF Database & Phase Matching
# =============================================================================
with tab_cif:
    st.subheader("💎 Crystallographic Information File (CIF) Database & Phase Identification")
    st.markdown("Load crystal structures via CIF (using pymatgen/gemmi) or choose from standard TEM reference libraries to index rings.")

    col_cif1, col_cif2 = st.columns([1, 1])

    with col_cif1:
        st.markdown("#### 1. Load Reference Phase")
        cif_source = st.radio("CIF Input Source", ["Built-in TEM Standards", "Upload Local CIF File", "Select from Local ./cifs Directory"])

        active_phase: Optional[CIFPhase] = None

        if cif_source == "Built-in TEM Standards":
            std_library = get_standard_materials()
            selected_std_key = st.selectbox("Choose Standard Phase", list(std_library.keys()))
            if st.button("Load Standard Phase", key="load_std_btn"):
                active_phase = create_phase_from_standard(selected_std_key)
                st.session_state["cif_phases"][active_phase.name] = active_phase
                st.session_state["selected_phase_name"] = active_phase.name
                st.success(f"Loaded '{active_phase.name}' into active phases!")

        elif cif_source == "Upload Local CIF File":
            uploaded_cifs = st.file_uploader(
                "Upload one or more CIF files (.cif)",
                type=["cif"],
                accept_multiple_files=True,
                help="Drag-and-drop multiple .cif files at once to load them all into the phase library."
            )
            if uploaded_cifs:
                st.caption(f"📎 {len(uploaded_cifs)} file(s) selected: {', '.join(f.name for f in uploaded_cifs)}")
                if st.button(f"Parse & Load All {len(uploaded_cifs)} CIF File(s)", type="primary"):
                    loaded_names = []
                    errors = []
                    for cif_file in uploaded_cifs:
                        try:
                            cif_text = cif_file.read().decode("utf-8", errors="ignore")
                            phase = load_cif(cif_text)
                            # Use filename as phase name (strip .cif)
                            phase.name = os.path.splitext(cif_file.name)[0]
                            st.session_state["cif_phases"][phase.name] = phase
                            loaded_names.append(f"{phase.name} ({len(phase.reflections)} reflections)")
                        except Exception as e:
                            errors.append(f"{cif_file.name}: {e}")
                    if loaded_names:
                        st.session_state["selected_phase_name"] = list(st.session_state["cif_phases"].keys())[-1]
                        st.success(f"✅ Loaded {len(loaded_names)} phase(s):\n" + "\n".join(f"• {n}" for n in loaded_names))
                    if errors:
                        st.error("❌ Failed to parse:\n" + "\n".join(errors))

        else:  # Select from Local ./cifs Directory
            # Search the whole app folder (any sub-folder name, .cif or .CIF) so files committed to
            # GitHub are found wherever they were placed. Paths are kept relative to the app folder.
            cif_dir = CURRENT_DIR
            cif_files = []
            for _root, _dirs, _files in os.walk(CURRENT_DIR):
                _dirs[:] = [d for d in _dirs if not d.startswith((".", "__")) and d not in ("venv", "node_modules", "site-packages")]
                for _f in _files:
                    if _f.lower().endswith(".cif"):
                        cif_files.append(os.path.relpath(os.path.join(_root, _f), CURRENT_DIR))
            cif_files.sort()

            if cif_files:
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    sel_local_cif = st.selectbox("Select CIF from the app folder (searched recursively)", cif_files)
                with col_c2:
                    st.write("")
                    st.write("")
                    if st.button("Load Selected CIF", use_container_width=True):
                        filepath = os.path.join(cif_dir, sel_local_cif)
                        phase = load_cif(filepath)
                        st.session_state["cif_phases"][phase.name] = phase
                        st.session_state["selected_phase_name"] = phase.name
                        st.success(f"Parsed '{phase.name}' with {len(phase.reflections)} reflections!")

                if st.button("📂 Load All Local CIF Files into Library", use_container_width=True):
                    loaded_count = 0
                    for cf in cif_files:
                        fp = os.path.join(cif_dir, cf)
                        ph = load_cif(fp)
                        st.session_state["cif_phases"][ph.name] = ph
                        loaded_count += 1
                    if cif_files:
                        st.session_state["selected_phase_name"] = os.path.splitext(os.path.basename(cif_files[0]))[0]
                    st.success(f"Successfully loaded all {loaded_count} CIF crystal structures into active library!")
            else:
                st.warning(
                    "No .cif files found. Put them in a folder named `cifs` at the top level of your GitHub "
                    "repository (next to app.py), commit, and reboot the app."
                )
                st.caption(f"Searched: {CURRENT_DIR}  |  top-level contents: " + ", ".join(sorted(os.listdir(CURRENT_DIR))[:30]))

        # Show currently loaded phases
        if st.session_state["cif_phases"]:
            st.markdown("#### Loaded Phases in Library")
            phase_keys = list(st.session_state["cif_phases"].keys())
            st.session_state["selected_phase_name"] = st.selectbox(
                "Active Candidate Phase for Matching",
                phase_keys,
                index=0
            )

    with col_cif2:
        st.markdown("#### Theoretical Reflections")
        if st.session_state["selected_phase_name"] in st.session_state["cif_phases"]:
            sel_phase = st.session_state["cif_phases"][st.session_state["selected_phase_name"]]
            st.write(f"**Phase:** {sel_phase.name} | **Crystal System:** {sel_phase.crystal_system or 'N/A'}")
            if sel_phase.cell_params:
                p_text = ", ".join([f"{k}={v:.3f}" for k, v in sel_phase.cell_params.items()])
                st.caption(f"Cell Parameters: {p_text}")

            theo_df = sel_phase.get_reflections_df()
            st.dataframe(theo_df[["hkl", "d_spacing_angstrom", "intensity"]], height=240, use_container_width=True)
        else:
            st.info("No candidate phase selected. Load one from the left panel.")

    # Match experimental to theoretical
    st.markdown("---")
    st.markdown("#### 2. Match Experimental SAED Rings to Candidate Phase")

    if st.session_state["metrics_df"] is not None and not st.session_state["metrics_df"].empty:
        exp_d_list = st.session_state["metrics_df"]["d-spacing (Å)"].tolist()

        col_m1, col_m2 = st.columns([1, 3])
        with col_m1:
            tolerance = st.slider("Tolerance Matching Error (%)", 0.5, 10.0, 3.5, step=0.1)
            run_match = st.button("Perform Phase Identification Matching", type="primary", use_container_width=True)

        with col_m2:
            if run_match and st.session_state["selected_phase_name"] in st.session_state["cif_phases"]:
                active_phase = st.session_state["cif_phases"][st.session_state["selected_phase_name"]]
                match_results = match_experimental_to_phase(exp_d_list, active_phase, tolerance_pct=tolerance)
                st.session_state["matching_df"] = match_results

            if st.session_state["matching_df"] is not None:
                m_df = st.session_state["matching_df"]
                st.dataframe(m_df, use_container_width=True)

                matched_count = (m_df["Status"] == "Matched").sum()
                total_count = len(m_df)
                st.success(f"Matched **{matched_count} / {total_count}** experimental rings within ±{tolerance}% tolerance!")

                # Joint Stick Spectrum Visualization
                active_phase = st.session_state["cif_phases"].get(st.session_state["selected_phase_name"])
                if active_phase is not None and st.session_state["radial_profile_radii"] is not None:
                    fig_matched = plot_radial_profile_with_peaks(
                        radii=st.session_state["radial_profile_radii"],
                        profile=st.session_state["radial_profile_intensity"],
                        background=st.session_state["radial_profile_bg"],
                        peaks_dict=st.session_state["peak_results"],
                        x_units="Radius (pixels)",
                        calibration_scale=float(st.session_state["calibration_scale"]),
                        matched_phases_info=[{
                            "name": active_phase.name,
                            "reflections": active_phase.reflections
                        }]
                    )
                    fig_matched.update_layout(title=f"Radial Profile with Theoretical {active_phase.name} Sticks")
                    st.plotly_chart(fig_matched, use_container_width=True)
    else:
        st.info("Extract radial profile and peaks in **Tab 3** before performing CIF phase identification.")

    # =========================================================================
    # MULTI-PHASE ANALYSIS SECTION
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🏆 Multi-Phase Comparison & Phase Discrimination")
    st.markdown(
        "Compare **all loaded phases simultaneously** to determine which best explains your "
        "diffraction data. Includes confidence scoring, systematic absence validation, "
        "fingerprint ring identification, and an auto-generated scientific narrative."
    )

    loaded_phases = st.session_state.get("cif_phases", {})
    exp_df = st.session_state.get("metrics_df")

    if not loaded_phases:
        st.info("💡 Load 2+ phases from the panel above, then run analysis here.")
    elif exp_df is None or exp_df.empty:
        st.info("💡 Run radial profile extraction in **Tab 3** first, then load phases and compare here.")
    else:
        exp_d_list_all = exp_df["d-spacing (Å)"].tolist()

        col_mp1, col_mp2 = st.columns([1, 3])
        with col_mp1:
            tol_multi = st.slider(
                "Matching Tolerance (%)", 0.5, 10.0, 3.5, step=0.1,
                key="multi_tol_slider",
                help="d-spacing error threshold for a ring to count as 'matched'"
            )
            run_multi = st.button(
                "🔬 Run Multi-Phase Comparison",
                type="primary", use_container_width=True,
                key="run_multi_btn"
            )

        if run_multi or st.session_state.get("leaderboard_df") is not None:
            if run_multi:
                leaderboard = run_all_phases_comparison(exp_d_list_all, loaded_phases, tolerance_pct=tol_multi)
                st.session_state["leaderboard_df"] = leaderboard
                st.session_state["leaderboard_tol"] = tol_multi

            leaderboard = st.session_state.get("leaderboard_df")
            tol_used = st.session_state.get("leaderboard_tol", tol_multi)

            if leaderboard is not None and not leaderboard.empty:

                # ── LEADERBOARD TABLE ─────────────────────────────────────
                st.markdown("#### 📊 Phase Leaderboard")

                def _color_confidence(val):
                    if isinstance(val, (int, float)):
                        if val >= 75:
                            return "background-color: #14532d; color: #bbf7d0"
                        elif val >= 50:
                            return "background-color: #713f12; color: #fef9c3"
                        else:
                            return "background-color: #7f1d1d; color: #fecaca"
                    return ""

                _styler = leaderboard.style
                _apply = getattr(_styler, "map", None) or _styler.applymap  # pandas >=2.1 renamed applymap -> map
                styled = _apply(_color_confidence, subset=["Confidence Score"])
                st.dataframe(styled, use_container_width=True, height=min(400, 80 + 40 * len(leaderboard)))

                # Winner callout
                best = leaderboard.iloc[0]
                best_name = best["Phase"]
                st.success(
                    f"🏆 **Best Match: {best_name}** — "
                    f"{best['Rings Matched']}/{best['Total Rings']} rings matched, "
                    f"Confidence = {best['Confidence Score']}/100"
                )

                # ── SYSTEMATIC ABSENCE CHECKER ────────────────────────────
                st.markdown("---")
                st.markdown("#### 🔬 Systematic Absence Validator (Best Phase)")

                best_phase_obj = loaded_phases.get(best_name)
                if best_phase_obj is not None:
                    best_matching = match_experimental_to_phase(exp_d_list_all, best_phase_obj, tolerance_pct=tol_used)
                    crystal_sys = getattr(best_phase_obj, "crystal_system", "")
                    sa_df = check_systematic_absences(best_matching, crystal_sys)

                    display_cols = ["Experimental Ring", "Exp d (Å)", "Theo (hkl)", "Error (%)", "Status", "SA Warning"]
                    avail = [c for c in display_cols if c in sa_df.columns]
                    st.dataframe(sa_df[avail], use_container_width=True)

                    forbidden = sa_df[sa_df["SA Warning"].str.startswith("⚠️", na=False)]
                    if forbidden.empty:
                        st.success("✅ All matched reflections are crystallographically allowed. No systematic absence violations.")
                    else:
                        st.warning(
                            f"⚠️ {len(forbidden)} matched reflection(s) appear to violate systematic absence rules — "
                            "these may be super-lattice reflections, mis-assignments, or database errors."
                        )

                    # ── DISCRIMINATING RINGS ──────────────────────────────
                    st.markdown("---")
                    st.markdown("#### 🎯 Discriminating Reflections (Unique Fingerprints)")
                    st.markdown(
                        "Rings that match the best phase **and cannot be explained by any other loaded phase** "
                        "— these are the definitive proof of phase identification."
                    )

                    disc_rings = find_discriminating_rings(
                        exp_d_list_all, loaded_phases, best_name, tolerance_pct=tol_used
                    )
                    st.session_state["discriminating_rings"] = disc_rings

                    if disc_rings:
                        disc_df = pd.DataFrame(disc_rings)
                        disc_df = disc_df.rename(columns={
                            "exp_d": "Exp d (Å)",
                            "hkl": "Best-Phase hkl",
                            "best_error": "Error (%)",
                            "unique": "Unique to Best Phase?",
                            "competing_phases": "Also matched by",
                        })
                        disc_df["Unique to Best Phase?"] = disc_df["Unique to Best Phase?"].map(
                            {True: "✅ YES — fingerprint ring", False: "⚠️ Shared with other phases"}
                        )
                        disc_df["Also matched by"] = disc_df["Also matched by"].apply(
                            lambda x: ", ".join(x) if x else "—"
                        )
                        st.dataframe(disc_df, use_container_width=True)

                        n_unique = sum(1 for r in disc_rings if r["unique"])
                        if n_unique > 0:
                            unique_hkls = [r["hkl"] for r in disc_rings if r["unique"]]
                            st.success(
                                f"✅ **{n_unique} fingerprint ring(s)** uniquely confirm **{best_name}**: "
                                f"{', '.join(unique_hkls[:5])}"
                            )
                        else:
                            st.warning(
                                "⚠️ No rings are exclusively unique to the best phase — "
                                "all matched rings have overlapping d-spacings with other candidate phases. "
                                "Consider loading more reference phases to improve discrimination."
                            )
                    else:
                        st.info("No discriminating ring data. Run matching in the section above first.")

                    # ── AUTO-NARRATIVE ────────────────────────────────────
                    st.markdown("---")
                    st.markdown("#### 📝 Auto-Generated Scientific Narrative")

                    narrative = generate_phase_narrative(
                        leaderboard_df=leaderboard,
                        discriminating_rings=disc_rings,
                        best_phase_name=best_name,
                        crystal_system=crystal_sys,
                        space_group=getattr(best_phase_obj, "space_group", ""),
                    )
                    st.info(narrative)
                    st.download_button(
                        "📥 Download Narrative (.txt)",
                        data=narrative,
                        file_name=f"{st.session_state.get('image_filename', 'saed')}_phase_narrative.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="download_narrative_btn"
                    )



    # -------------------------------------------------------------------------
    # Mixed-phase ring assignment (which ring belongs to which phase?)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧪 Mixed-Phase Ring Assignment")
    st.caption(
        "For samples containing several phases (e.g. Fe₃O₄ + Fe₂O₃ + steel). Every ring is assigned to the "
        "phases that can explain it: **unique** (one phase only), **shared**, or **unexplained**. "
        "Phases SAED cannot separate (such as magnetite / maghemite) are merged into one group. "
        "Ring brightness is not used to estimate phase fractions."
    )
    if loaded_phases and exp_df is not None and not exp_df.empty:
        mix_names = st.multiselect(
            "Candidate phases to include (fewer, chemically plausible candidates give clearer results)",
            list(loaded_phases), default=list(loaded_phases)[:6], key="mix_phase_select",
        )
        tol_mix = st.slider(
            "Mixture tolerance (%)", 0.5, 6.0, 2.5, step=0.1, key="mix_tol_slider",
            help="Lower values separate close phases (e.g. hematite vs Cr2O3) but only if your calibration is accurate.",
        )
        if mix_names:
            ring_df_mix, group_df_mix, mix_warnings = assign_rings_to_phases(
                exp_df["d-spacing (Å)"].tolist(), {n: loaded_phases[n] for n in mix_names}, tol_pct=tol_mix
            )
            if not group_df_mix.empty:
                st.markdown("**Evidence per phase group**")
                st.dataframe(group_df_mix, use_container_width=True, hide_index=True)
                st.markdown("**Ring-by-ring assignment**")
                st.dataframe(ring_df_mix, use_container_width=True, hide_index=True)
                for msg in mix_warnings:
                    st.warning(msg)
                st.download_button(
                    "📥 Download assignment (CSV)", data=ring_df_mix.to_csv(index=False),
                    file_name=f"{st.session_state.get('image_filename', 'saed')}_mixed_phase.csv",
                    mime="text/csv", use_container_width=True, key="download_mix_btn",
                )
        else:
            st.info("Select at least one phase above.")
    else:
        st.info("💡 Detect rings in Tab 3 and load two or more phases to use mixed-phase assignment.")


# =============================================================================
# TAB 5: Publication Report Generator
# =============================================================================
with tab_report:
    st.subheader("📄 Publication-Ready Report & Manuscript Compiler")
    st.markdown("Automatically compiles electron diffraction analysis into standard publication formats: summary comparison tables, formal LaTeX tables, and manuscript methods/results narrative.")

    # 1. Microscope Metadata Inputs Section
    st.markdown("#### 🔬 Microscope & Experimental Calibration Metadata")
    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)

    with col_meta1:
        pub_voltage = st.number_input(
            "Accelerating Voltage (kV)",
            min_value=20.0,
            max_value=1200.0,
            value=float(st.session_state.get("microscope_voltage_kv", 200.0)),
            step=10.0,
            key="pub_voltage_input"
        )
        st.session_state["microscope_voltage_kv"] = pub_voltage

    with col_meta2:
        pub_cam_len = st.number_input(
            "Camera Length (mm)",
            min_value=50.0,
            max_value=3500.0,
            value=float(st.session_state.get("microscope_camera_length_mm", 500.0)),
            step=25.0,
            key="pub_cam_len_input"
        )
        st.session_state["microscope_camera_length_mm"] = pub_cam_len

    with col_meta3:
        pub_cal_ref = st.text_input(
            "Calibration Reference Material",
            value=st.session_state.get("microscope_cal_reference", "Standard Polycrystalline Au (Gold)"),
            key="pub_cal_ref_input"
        )
        st.session_state["microscope_cal_reference"] = pub_cal_ref

    with col_meta4:
        pub_zone_axis = st.text_input(
            "Zone Axis / Pattern Type",
            value=st.session_state.get("microscope_zone_axis", "Polycrystalline ring pattern (Debye-Scherrer rings)"),
            key="pub_zone_axis_input"
        )
        st.session_state["microscope_zone_axis"] = pub_zone_axis

    st.markdown("---")

    # Check whether matching results exist
    m_df = st.session_state.get("matching_df")
    active_phase_name = st.session_state.get("selected_phase_name", "Identified Crystal Phase")
    active_phase_obj = st.session_state.get("cif_phases", {}).get(active_phase_name)
    c_sys = getattr(active_phase_obj, "crystal_system", "") if active_phase_obj else ""
    s_grp = getattr(active_phase_obj, "space_group", "") if active_phase_obj else ""

    if m_df is not None and not m_df.empty:
        # 2. Publication Summary Table
        st.markdown(f"#### 📊 Publication Summary Table: *{active_phase_name}*")
        
        # Display clean styled dataframe
        summary_view_cols = ["Experimental Ring", "Exp d (Å)", "Theo d (Å)", "Error (%)", "Theo (hkl)", "Theo Intensity", "Status"]
        avail_cols = [c for c in summary_view_cols if c in m_df.columns]
        st.dataframe(
            m_df[avail_cols].rename(columns={
                "Experimental Ring": "Ring #",
                "Exp d (Å)": "Measured d (Å)",
                "Theo d (Å)": "Calculated d (Å)",
                "Error (%)": "Error (%)",
                "Theo (hkl)": "Indexed Plane (hkl)",
                "Theo Intensity": "Theoretical Intensity",
                "Status": "Status"
            }),
            use_container_width=True
        )

        st.markdown("---")

        # 3. Formatted Paragraph Summarizing Phase Identification & Zone Axis
        st.markdown("#### 📝 Formatted Publication Narrative")
        narrative_text = generate_publication_narrative(
            matching_df=m_df,
            phase_name=active_phase_name,
            crystal_system=c_sys,
            space_group=s_grp,
            voltage_kv=pub_voltage,
            camera_length_mm=pub_cam_len,
            cal_ref_material=pub_cal_ref,
            zone_axis=pub_zone_axis
        )
        st.text_area(
            "Copy-ready Methods/Results Paragraph (for Manuscript, Thesis, or Report)",
            value=narrative_text,
            height=140
        )

        st.markdown("---")

        # 4. Export Formats: Markdown & LaTeX Code
        st.markdown("#### 📋 Copy-Pasteable Table Codes")
        col_tab1, col_tab2 = st.columns(2)

        with col_tab1:
            st.markdown("**Markdown Table Format**")
            md_table_code = generate_publication_markdown_table(m_df, phase_name=active_phase_name)
            st.code(md_table_code, language="markdown")
            st.download_button(
                "📥 Download Markdown Table (.md)",
                data=md_table_code,
                file_name=f"{st.session_state['image_filename']}_table.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col_tab2:
            st.markdown("**LaTeX Table Code (booktabs)**")
            latex_table_code = generate_publication_latex_table(
                matching_df=m_df,
                phase_name=active_phase_name,
                voltage_kv=pub_voltage,
                camera_length_mm=pub_cam_len
            )
            st.code(latex_table_code, language="latex")
            st.download_button(
                "📥 Download LaTeX Table (.tex)",
                data=latex_table_code,
                file_name=f"{st.session_state['image_filename']}_table.tex",
                mime="text/x-tex",
                use_container_width=True
            )

    else:
        st.info("💡 To generate a publication report: extract radial profile peaks in **Tab 3** and run phase matching in **Tab 4** (or click 'Load Synthetic Au Pattern' from sidebar for instant demo).")


# =============================================================================
# TAB 6: Export Results
# =============================================================================
with tab_export:
    st.subheader("💾 Export Quantitative SAED Analysis")

    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        st.markdown("#### Export Radial Profile & Metrics Data")
        if st.session_state["radial_profile_radii"] is not None:
            prof_df = pd.DataFrame({
                "Radius_px": st.session_state["radial_profile_radii"],
                "Intensity_raw": st.session_state["radial_profile_intensity"],
                "Intensity_bg": st.session_state["radial_profile_bg"] if st.session_state["radial_profile_bg"] is not None else np.zeros_like(st.session_state["radial_profile_radii"]),
                "q_nm_inv": st.session_state["radial_profile_radii"] * float(st.session_state["calibration_scale"])
            })
            csv_prof = prof_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Radial Profile (CSV)",
                data=csv_prof,
                file_name=f"{st.session_state['image_filename']}_radial_profile.csv",
                mime="text/csv",
                use_container_width=True
            )

        if st.session_state["metrics_df"] is not None:
            csv_metrics = st.session_state["metrics_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Ring Metrics & d-Spacings (CSV)",
                data=csv_metrics,
                file_name=f"{st.session_state['image_filename']}_ring_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )

        if st.session_state["matching_df"] is not None:
            csv_match = st.session_state["matching_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Phase Identification Results (CSV)",
                data=csv_match,
                file_name=f"{st.session_state['image_filename']}_phase_matches.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col_ex2:
        st.markdown("#### Export Comprehensive Report (JSON)")
        report_data = {
            "image_filename": st.session_state["image_filename"],
            "center_coordinates": {
                "x_px": float(st.session_state["center_x"]),
                "y_px": float(st.session_state["center_y"]),
                "method": st.session_state["center_method"]
            },
            "calibration": {
                "scale_factor": float(st.session_state["calibration_scale"]),
                "unit": st.session_state["calibration_unit"]
            },
            "rings_detected": st.session_state["metrics_df"].to_dict(orient="records") if st.session_state["metrics_df"] is not None else [],
            "phase_matches": st.session_state["matching_df"].to_dict(orient="records") if st.session_state["matching_df"] is not None else []
        }
        json_str = json.dumps(report_data, indent=2).encode("utf-8")
        st.download_button(
            "📥 Download Analysis Report (JSON)",
            data=json_str,
            file_name=f"{st.session_state['image_filename']}_analysis_report.json",
            mime="application/json",
            use_container_width=True
        )

        st.markdown("#### Analysis Metadata Summary")
        st.json(report_data)


# =============================================================================
# TAB 7: Theory & Equations
# =============================================================================
with tab_theory:
    render_theory()
