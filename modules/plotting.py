"""
Interactive Plotting and Visualization for SAED Pattern Analysis
Generates interactive 2D pattern views with detected center & ring overlays,
radial profiles with peak markers, and theoretical reflection stick patterns.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Ring colours (bright, readable on dark / grey images) and label angles (degrees).
# Labels are spread around the pattern so neighbouring rings never collide.
_RING_COLORS = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#ff9f1c", "#c77dff", "#80ed99", "#4cc9f0", "#f72585"]
_LABEL_ANGLES = [-60, -120, 120, 60, -90, 90, 150, -150, 135, -135, -45, -135]


def _short_label(label: str, idx: int) -> str:
    """'R3: d=2.35Å' -> 'R3'. Full text stays available in the hover tooltip."""
    short = str(label).split(":")[0].strip() if label else f"R{idx + 1}"
    return short if len(short) <= 12 else short[:11] + "…"


def plot_saed_pattern_with_rings(
    image: np.ndarray,
    center: Optional[Tuple[float, float]] = None,
    ring_radii_px: Optional[List[float]] = None,
    ring_labels: Optional[List[str]] = None,
    beam_stop_mask: Optional[np.ndarray] = None,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    color_scale: str = "gray",
    *args: Any,
    **kwargs: Any
) -> go.Figure:
    """2D diffraction pattern with a clean ring overlay.

    - one thin, semi-transparent circle per ring, each in its own colour
    - short labels (R1, R2, ...) staggered around the pattern; full text on hover
    - a single compact legend (centre / spots) instead of one entry per ring
    - optional beam-stop mask shaded red
    kwargs: spot_markers (list of dicts with x, y, radius_px, angle_deg),
            show_labels (bool, default True), max_spots (int, default 400)
    """
    h, w = image.shape[:2]
    show_labels = kwargs.get("show_labels", True)
    max_spots = kwargs.get("max_spots", 400)

    # Downsample large images for a snappy browser / mobile UI
    step = max(1, int(max(h, w) / 1024))
    disp_img = image[::step, ::step]
    xs, ys = np.arange(0, w, step), np.arange(0, h, step)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=disp_img, x=xs, y=ys,
            colorscale=color_scale,
            showscale=False,
            hovertemplate="x=%{x}, y=%{y}<br>I=%{z:.3g}<extra></extra>",
            showlegend=False,
        )
    )

    # Beam-stop / excluded area (True = masked)
    if beam_stop_mask is not None and beam_stop_mask.shape[:2] == (h, w):
        m = np.where(beam_stop_mask[::step, ::step] > 0, 1.0, np.nan)
        fig.add_trace(
            go.Heatmap(
                z=m, x=xs, y=ys,
                colorscale=[[0, "rgba(255,0,0,0.35)"], [1, "rgba(255,0,0,0.35)"]],
                showscale=False, hoverinfo="skip", showlegend=False,
            )
        )

    if center is not None:
        cx, cy = center

        # ---- rings -----------------------------------------------------------
        if ring_radii_px is not None and len(ring_radii_px) > 0:
            theta = np.linspace(0, 2 * np.pi, 180)
            lab_x, lab_y, lab_txt, lab_col = [], [], [], []
            for idx, r in enumerate(ring_radii_px):
                if r <= 0:
                    continue
                color = _RING_COLORS[idx % len(_RING_COLORS)]
                full = ring_labels[idx] if ring_labels and idx < len(ring_labels) \
                    else f"Ring {idx + 1} ({r:.1f}px)"
                fig.add_trace(
                    go.Scatter(
                        x=cx + r * np.cos(theta), y=cy + r * np.sin(theta),
                        mode="lines",
                        line=dict(width=1.2, color=color),
                        opacity=0.75,
                        text=[full] * len(theta), hoverinfo="text",
                        showlegend=False,
                    )
                )
                ang = np.radians(_LABEL_ANGLES[idx % len(_LABEL_ANGLES)])
                lx, ly = cx + r * np.cos(ang), cy + r * np.sin(ang)
                if 0 <= lx <= w and 0 <= ly <= h:          # keep labels inside the image
                    lab_x.append(lx); lab_y.append(ly)
                    lab_txt.append(_short_label(full, idx)); lab_col.append(color)
            if show_labels and lab_x:
                fig.add_trace(
                    go.Scatter(
                        x=lab_x, y=lab_y, mode="text", text=lab_txt,
                        textfont=dict(size=12, color=lab_col),
                        hoverinfo="skip", showlegend=False,
                    )
                )

        # ---- discrete spots --------------------------------------------------
        spot_markers = kwargs.get("spot_markers")
        if spot_markers and len(spot_markers) > 0:
            shown = spot_markers[:max_spots]
            fig.add_trace(
                go.Scatter(
                    x=[s["x"] for s in shown], y=[s["y"] for s in shown],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=6, color="#f59e0b",
                                line=dict(width=1), opacity=0.7),
                    text=[f"Spot: r={s['radius_px']:.1f}px (θ={s['angle_deg']:.1f}°)" for s in shown],
                    hoverinfo="text",
                    name=f"Spots ({len(shown)}{'+' if len(spot_markers) > max_spots else ''})",
                )
            )

        # ---- centre cross (drawn last so it is on top) -------------------------
        fig.add_trace(
            go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(symbol="cross-thin", size=16, color="cyan",
                            line=dict(width=2, color="cyan")),
                name="Centre", hovertemplate="Centre<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
            )
        )

    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="#00ffcc", width=2, dash="dash"))

    fig.update_layout(
        title=dict(text="Diffraction pattern", font=dict(size=15)),
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(range=[0, w], constrain="domain", showgrid=False, title=None),
        yaxis=dict(range=[h, 0], constrain="domain", scaleanchor="x", scaleratio=1,
                   showgrid=False, title=None),
    )
    return fig


def plot_radial_profile_with_peaks(
    radii: np.ndarray,
    profile: np.ndarray,
    background: Optional[np.ndarray] = None,
    peaks_dict: Optional[Dict[str, Any]] = None,
    x_units: str = "Radius (px)",
    calibration_scale: float = 1.0,
    matched_phases_info: Optional[List[Dict[str, Any]]] = None
) -> go.Figure:
    """Plot azimuthally integrated 1D profile with detected peaks and CIF sticks."""
    fig = go.Figure()

    # Raw / integrated profile
    fig.add_trace(
        go.Scatter(
            x=radii,
            y=profile,
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            name="Radial Profile I(r)"
        )
    )

    # Background line if present
    if background is not None:
        fig.add_trace(
            go.Scatter(
                x=radii,
                y=background,
                mode="lines",
                line=dict(color="#94a3b8", dash="dash", width=1.5),
                name="Estimated Background"
            )
        )
        # Background-subtracted profile
        sub = np.maximum(0.0, profile - background)
        fig.add_trace(
            go.Scatter(
                x=radii,
                y=sub,
                mode="lines",
                line=dict(color="#10b981", width=1.5),
                name="Net Profile (I - Bg)"
            )
        )

    # Detected peaks
    if peaks_dict is not None and len(peaks_dict.get("peak_radii_px", [])) > 0:
        pk_radii = peaks_dict["peak_radii_px"]
        pk_intensities = peaks_dict["intensities"]
        fwhms = peaks_dict.get("fwhm_px", np.zeros(len(pk_radii)))

        hover_texts = [
            f"Ring {i+1}<br>Radius: {r:.2f} px<br>Intensity: {y:.3f}<br>FWHM: {w:.2f} px"
            for i, (r, y, w) in enumerate(zip(pk_radii, pk_intensities, fwhms))
        ]

        fig.add_trace(
            go.Scatter(
                x=pk_radii,
                y=pk_intensities,
                mode="markers+text",
                marker=dict(color="#ef4444", size=10, symbol="triangle-down"),
                text=[f"R{i+1}" for i in range(len(pk_radii))],
                textposition="top center",
                hovertext=hover_texts,
                hoverinfo="text",
                name="Detected Rings / Peaks"
            )
        )

    # CIF Theoretical reflection stick markers
    if matched_phases_info is not None:
        colors = ["#eab308", "#a855f7", "#06b6d4", "#f97316"]
        for p_idx, phase_data in enumerate(matched_phases_info):
            p_name = phase_data.get("name", "CIF Phase")
            reflections = phase_data.get("reflections", [])
            color = colors[p_idx % len(colors)]

            # Convert d to radius px: r_px = 1 / (d * calibration)
            stick_x = []
            stick_y = []
            stick_labels = []
            max_intensity = max(profile) if len(profile) > 0 else 1.0

            for ref in reflections:
                d_val = ref.get("d_spacing_angstrom", ref.get("d", 0.0))
                if d_val <= 0:
                    continue
                # If calibration_scale is nm^-1/px: d (nm) = 1 / (r * cal) => r = 1 / (d_nm * cal) = 10 / (d_A * cal)
                r_calc = 10.0 / (d_val * calibration_scale) if calibration_scale > 0 else 0
                if 0 <= r_calc <= np.max(radii):
                    stick_height = (ref.get("intensity", 50.0) / 100.0) * max_intensity * 0.75
                    stick_x.extend([r_calc, r_calc, None])
                    stick_y.extend([0, stick_height, None])
                    stick_labels.append(f"{p_name} {ref.get('hkl', '')}: d={d_val:.3f}Å")

            if stick_x:
                fig.add_trace(
                    go.Scatter(
                        x=stick_x,
                        y=stick_y,
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=f"Theory: {p_name}"
                    )
                )

    fig.update_layout(
        title="Radial Intensity Profile & Ring Identification",
        xaxis_title=x_units,
        yaxis_title="Integrated Intensity (arbitrary units)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
