"""
SAED Publication Report Generator Module
Compiles diffraction analysis, d-spacing metrics, and CIF phase matches
into publication-ready formats:
- Formatted Markdown summary tables
- Clean LaTeX table code
- Formatted publication paragraph summarizing phase identification and zone axis indexing
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def generate_publication_markdown_table(
    matching_df: pd.DataFrame,
    phase_name: str = ""
) -> str:
    """Generate a clean GitHub-Flavored Markdown table of matched rings."""
    if matching_df is None or matching_df.empty:
        return "*No diffraction rings indexed or matched yet.*"

    md_lines = [
        f"| Ring # | Measured $d$ (Å) | Calculated $d$ (Å) | Error (%) | Indexed Plane $(hkl)$ | Intensity (a.u.) | Status |",
        f"|:------:|:----------------:|:------------------:|:---------:|:---------------------:|:----------------:|:------:|"
    ]

    for _, row in matching_df.iterrows():
        ring = str(row.get("Experimental Ring", "Ring"))
        exp_d = f"{float(row.get('Exp d (Å)', 0.0)):.4f}" if pd.notnull(row.get("Exp d (Å)")) else "—"
        theo_d = f"{float(row.get('Theo d (Å)', 0.0)):.4f}" if pd.notnull(row.get("Theo d (Å)")) else "—"
        err = f"{float(row.get('Error (%)', 0.0)):.2f}%" if pd.notnull(row.get("Error (%)")) else "—"
        hkl = str(row.get("Theo (hkl)", "Unindexed"))
        intensity = f"{float(row.get('Theo Intensity', 0.0)):.1f}" if pd.notnull(row.get("Theo Intensity")) else "—"
        status = str(row.get("Status", "—"))

        md_lines.append(f"| {ring} | {exp_d} | {theo_d} | {err} | {hkl} | {intensity} | {status} |")

    return "\n".join(md_lines)


def generate_publication_latex_table(
    matching_df: pd.DataFrame,
    phase_name: str = "Candidate Phase",
    voltage_kv: float = 200.0,
    camera_length_mm: float = 500.0
) -> str:
    """Generate publication-ready LaTeX table code (booktabs format)."""
    if matching_df is None or matching_df.empty:
        return "% No diffraction rings available for LaTeX table."

    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Indexed Selected Area Electron Diffraction (SAED) ring patterns matched to {phase_name}. TEM operating conditions: {voltage_kv:.0f}~kV, camera length $L = {camera_length_mm:.0f}$~mm.}}",
        r"\label{tab:saed_indexing}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"\textbf{Ring \#} & \textbf{$d_{\text{meas}}$ (\AA)} & \textbf{$d_{\text{calc}}$ (\AA)} & \textbf{Error (\%)} & \textbf{Plane $(hkl)$} & \textbf{Rel. Int. (\%)} \\",
        r"\midrule"
    ]

    for _, row in matching_df.iterrows():
        ring = str(row.get("Experimental Ring", "Ring")).replace("#", r"\#")
        exp_d = f"{float(row.get('Exp d (Å)', 0.0)):.4f}" if pd.notnull(row.get("Exp d (Å)")) else "--"
        theo_d = f"{float(row.get('Theo d (Å)', 0.0)):.4f}" if pd.notnull(row.get("Theo d (Å)")) else "--"
        err = f"{float(row.get('Error (%)', 0.0)):.2f}" if pd.notnull(row.get("Error (%)")) else "--"
        hkl = str(row.get("Theo (hkl)", "Unindexed")).replace("(", "").replace(")", "").strip()
        hkl_tex = f"({hkl})" if hkl != "Unindexed" else "Unindexed"
        intensity = f"{float(row.get('Theo Intensity', 0.0)):.1f}" if pd.notnull(row.get("Theo Intensity")) else "--"

        latex.append(f"{ring} & {exp_d} & {theo_d} & {err}\\% & {hkl_tex} & {intensity} \\\\")

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(latex)


def generate_publication_narrative(
    matching_df: pd.DataFrame,
    phase_name: str,
    crystal_system: str = "",
    space_group: str = "",
    voltage_kv: float = 200.0,
    camera_length_mm: float = 500.0,
    cal_ref_material: str = "Standard Au reference",
    zone_axis: str = "Polycrystalline Ring Pattern (All allowed reflections)"
) -> str:
    """Generate a formal manuscript methods/results paragraph for TEM/SAED phase identification."""
    if matching_df is None or matching_df.empty:
        return "Diffraction analysis pending: Please execute radial integration and phase identification."

    matched_rows = matching_df[matching_df["Status"] == "Matched"]
    total_rings = len(matching_df)
    matched_count = len(matched_rows)

    planes_list = []
    d_comp_list = []
    errors = []

    for _, row in matched_rows.iterrows():
        hkl = str(row.get("Theo (hkl)", ""))
        exp_d = float(row.get("Exp d (Å)", 0.0))
        theo_d = float(row.get("Theo d (Å)", 0.0))
        err = float(row.get("Error (%)", 0.0))
        errors.append(err)
        if hkl and hkl != "Unindexed":
            planes_list.append(hkl)
            d_comp_list.append(f"{exp_d:.3f} Å (theo. {theo_d:.3f} Å)")

    mean_err = np.mean(errors) if errors else 0.0
    max_err = np.max(errors) if errors else 0.0

    planes_str = ", ".join(planes_list[:5])
    if len(planes_list) > 5:
        planes_str += f", and {len(planes_list) - 5} higher-order reflections"

    struct_info = f"{phase_name}"
    if crystal_system:
        struct_info += f" ({crystal_system}"
        if space_group:
            struct_info += f", space group {space_group}"
        struct_info += ")"

    paragraph = (
        f"Selected area electron diffraction (SAED) analysis was performed to determine the crystal phase "
        f"and structural orientation of the specimen. The diffraction patterns were recorded on a transmission "
        f"electron microscope operated at an accelerating voltage of {voltage_kv:.0f} kV with a nominal camera length "
        f"of {camera_length_mm:.0f} mm, calibrated against a {cal_ref_material}. Azimuthally integrated radial intensity "
        f"profiles were extracted following sub-pixel direct-beam centering and diffuse background subtraction. "
        f"The experimental interplanar spacings were indexed against standard reference reflections for {struct_info}. "
        f"A total of {matched_count} of the {total_rings} primary diffraction rings were unambiguously indexed to the "
        f"{planes_str} crystallographic planes, corresponding to a mean relative deviation of {mean_err:.2f}% "
        f"(maximum deviation {max_err:.2f}%). {zone_axis}."
    )

    return paragraph
