"""
SAED CIF Database & Phase Identification Module
Parses CIF structures (via pymatgen or gemmi with pure-Python fallback),
computes theoretical interplanar d-spacings and structure factors / intensities,
and matches experimental SAED d-spacings against candidate crystal phases.
"""

import os
import math
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


def calculate_d_cubic(a: float, h: int, k: int, l: int) -> float:
    denom = math.sqrt(h**2 + k**2 + l**2)
    return a / denom if denom > 0 else 0.0

def calculate_d_tetragonal(a: float, c: float, h: int, k: int, l: int) -> float:
    val = (h**2 + k**2) / (a**2) + (l**2) / (c**2)
    return 1.0 / math.sqrt(val) if val > 0 else 0.0

def calculate_d_hexagonal(a: float, c: float, h: int, k: int, l: int) -> float:
    val = (4.0 / 3.0) * (h**2 + h*k + k**2) / (a**2) + (l**2) / (c**2)
    return 1.0 / math.sqrt(val) if val > 0 else 0.0

def calculate_d_orthorhombic(a: float, b: float, c: float, h: int, k: int, l: int) -> float:
    val = (h**2 / a**2) + (k**2 / b**2) + (l**2 / c**2)
    return 1.0 / math.sqrt(val) if val > 0 else 0.0

def calculate_d_general(a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float, h: int, k: int, l: int) -> float:
    """General triclinic/monoclinic reciprocal lattice metric tensor d-spacing calculation."""
    ra = math.radians(alpha_deg)
    rb = math.radians(beta_deg)
    rg = math.radians(gamma_deg)

    # Unit cell volume
    cos_a, cos_b, cos_g = math.cos(ra), math.cos(rb), math.cos(rg)
    sin_a, sin_b, sin_g = math.sin(ra), math.sin(rb), math.sin(rg)

    term = 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    if term <= 0:
        return 0.0
    V = a * b * c * math.sqrt(term)

    # Reciprocal cell parameters
    astar = (b * c * sin_a) / V
    bstar = (a * c * sin_b) / V
    cstar = (a * b * sin_g) / V

    cos_astar = (cos_b * cos_g - cos_a) / (sin_b * sin_g)
    cos_bstar = (cos_a * cos_g - cos_b) / (sin_a * sin_g)
    cos_cstar = (cos_a * cos_b - cos_g) / (sin_a * sin_b)

    # 1 / d^2
    one_over_d2 = (
        (h * astar)**2 + (k * bstar)**2 + (l * cstar)**2 +
        2 * h * k * astar * bstar * cos_cstar +
        2 * k * l * bstar * cstar * cos_astar +
        2 * h * l * astar * cstar * cos_bstar
    )

    if one_over_d2 <= 0:
        return 0.0
    return 1.0 / math.sqrt(one_over_d2)


class CIFPhase:
    """Represents a crystal phase with calculated reflections."""
    def __init__(self, name: str, source: str = "custom"):
        self.name = name
        self.source = source
        self.formula = ""
        self.crystal_system = ""
        self.space_group = ""
        self.cell_params: Dict[str, float] = {}
        self.reflections: List[Dict[str, Any]] = []

    def get_reflections_df(self) -> pd.DataFrame:
        if not self.reflections:
            return pd.DataFrame(columns=["h", "k", "l", "hkl", "d_spacing_angstrom", "intensity"])
        df = pd.DataFrame(self.reflections)
        return df.sort_values(by="d_spacing_angstrom", ascending=False).reset_index(drop=True)


def parse_cif_pymatgen(file_path_or_str: str) -> Optional[CIFPhase]:
    """Parse CIF using pymatgen if installed."""
    try:
        from pymatgen.core.structure import Structure
        from pymatgen.analysis.diffraction.tem import TEMCalculator

        if os.path.exists(file_path_or_str):
            struct = Structure.from_file(file_path_or_str)
            name = os.path.splitext(os.path.basename(file_path_or_str))[0]
        else:
            struct = Structure.from_str(file_path_or_str, fmt="cif")
            name = "CIF Structure"

        phase = CIFPhase(name=name, source="pymatgen")
        phase.formula = struct.composition.reduced_formula
        phase.crystal_system = struct.get_space_group_info()[0] if hasattr(struct, "get_space_group_info") else ""
        lat = struct.lattice
        phase.cell_params = {
            "a": lat.a, "b": lat.b, "c": lat.c,
            "alpha": lat.alpha, "beta": lat.beta, "gamma": lat.gamma
        }

        # Calculate electron diffraction pattern
        tem_calc = TEMCalculator()
        pattern = tem_calc.get_pattern(struct)
        reflections = []
        for d, hkl, intensity in zip(pattern.d_hkls, pattern.hkls, pattern.intensities):
            hkl_tuple = hkl[0]["hkl"] if isinstance(hkl[0], dict) else hkl[0]
            reflections.append({
                "h": int(hkl_tuple[0]),
                "k": int(hkl_tuple[1]),
                "l": int(hkl_tuple[2]),
                "hkl": f"({hkl_tuple[0]} {hkl_tuple[1]} {hkl_tuple[2]})",
                "d_spacing_angstrom": float(d),
                "intensity": float(intensity)
            })
        phase.reflections = reflections
        return phase
    except Exception:
        return None


def parse_cif_gemmi(file_path_or_str: str) -> Optional[CIFPhase]:
    """Parse CIF using gemmi if installed."""
    try:
        import gemmi

        if os.path.exists(file_path_or_str):
            doc = gemmi.cif.read_file(file_path_or_str)
            name = os.path.splitext(os.path.basename(file_path_or_str))[0]
        else:
            doc = gemmi.cif.read_string(file_path_or_str)
            name = "CIF Structure"

        block = doc.sole_block()
        cell = gemmi.UnitCell(
            float(block.find_value("_cell_length_a") or 1.0),
            float(block.find_value("_cell_length_b") or 1.0),
            float(block.find_value("_cell_length_c") or 1.0),
            float(block.find_value("_cell_angle_alpha") or 90.0),
            float(block.find_value("_cell_angle_beta") or 90.0),
            float(block.find_value("_cell_angle_gamma") or 90.0)
        )

        phase = CIFPhase(name=name, source="gemmi")
        phase.cell_params = {
            "a": cell.a, "b": cell.b, "c": cell.c,
            "alpha": cell.alpha, "beta": cell.beta, "gamma": cell.gamma
        }

        reflections = []
        max_index = 5
        seen_d = set()
        for h in range(max_index + 1):
            for k in range(max_index + 1):
                for l in range(max_index + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    d = cell.calculate_d(gemmi.Miller(h, k, l))
                    if 0.5 <= d <= 5.0:
                        d_round = round(d, 4)
                        if d_round not in seen_d:
                            seen_d.add(d_round)
                            reflections.append({
                                "h": h, "k": k, "l": l,
                                "hkl": f"({h} {k} {l})",
                                "d_spacing_angstrom": d,
                                "intensity": 100.0 / (h**2 + k**2 + l**2)
                            })
        phase.reflections = reflections
        return phase
    except Exception:
        return None


def parse_cif_pure_python(file_path_or_str: str) -> CIFPhase:
    """Robust regex-based pure Python fallback CIF parser."""
    if os.path.exists(file_path_or_str):
        with open(file_path_or_str, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        name = os.path.splitext(os.path.basename(file_path_or_str))[0]
    else:
        content = file_path_or_str
        name = "CIF Structure"

    phase = CIFPhase(name=name, source="internal_parser")

    # Extract cell parameters
    def get_tag_num(tag: str, default: float) -> float:
        pattern = rf"{tag}\s+([0-9]+\.?[0-9]*)"
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        return default

    a = get_tag_num(r"_cell_length_a", 4.0)
    b = get_tag_num(r"_cell_length_b", a)
    c = get_tag_num(r"_cell_length_c", a)
    alpha = get_tag_num(r"_cell_angle_alpha", 90.0)
    beta = get_tag_num(r"_cell_angle_beta", 90.0)
    gamma = get_tag_num(r"_cell_angle_gamma", 90.0)

    phase.cell_params = {
        "a": a, "b": b, "c": c,
        "alpha": alpha, "beta": beta, "gamma": gamma
    }

    # Extract name/formula if available
    name_match = re.search(r"_chemical_name_common\s+['\"]?([^'\"\n]+)['\"]?", content, re.IGNORECASE)
    if name_match:
        phase.name = name_match.group(1).strip()
    formula_match = re.search(r"_chemical_formula_sum\s+['\"]?([^'\"\n]+)['\"]?", content, re.IGNORECASE)
    if formula_match:
        phase.formula = formula_match.group(1).strip()

    # Generate theoretical reflections up to max hkl index 6
    reflections = []
    seen_d = {}
    for h in range(7):
        for k in range(7):
            for l in range(7):
                if h == 0 and k == 0 and l == 0:
                    continue
                d = calculate_d_general(a, b, c, alpha, beta, gamma, h, k, l)
                if 0.5 <= d <= 5.0:
                    d_key = round(d, 3)
                    # Simple multiplicity tracking
                    if d_key not in seen_d:
                        seen_d[d_key] = {
                            "h": h, "k": k, "l": l,
                            "hkl": f"({h} {k} {l})",
                            "d_spacing_angstrom": d,
                            "multiplicity": 1
                        }
                    else:
                        seen_d[d_key]["multiplicity"] += 1

    for item in seen_d.values():
        mult = item.pop("multiplicity")
        h, k, l = item["h"], item["k"], item["l"]
        est_intensity = float(mult * 100.0 / (h**2 + k**2 + l**2 + 1e-4))
        item["intensity"] = round(est_intensity, 1)
        reflections.append(item)

    reflections.sort(key=lambda x: x["d_spacing_angstrom"], reverse=True)
    phase.reflections = reflections
    return phase


def load_cif(file_path_or_str: str) -> CIFPhase:
    """Unified entry to load CIF: tries pymatgen -> gemmi -> internal parser."""
    phase = parse_cif_pymatgen(file_path_or_str)
    if phase is not None:
        return phase
    phase = parse_cif_gemmi(file_path_or_str)
    if phase is not None:
        return phase
    return parse_cif_pure_python(file_path_or_str)


def match_experimental_to_phase(
    exp_d_spacings: List[float],
    phase: CIFPhase,
    tolerance_pct: float = 3.0
) -> pd.DataFrame:
    """Match a list of experimental d-spacings (in Angstroms) against theoretical phase reflections."""
    if not phase.reflections or len(exp_d_spacings) == 0:
        return pd.DataFrame()

    theo_df = phase.get_reflections_df()
    theo_d = theo_df["d_spacing_angstrom"].to_numpy()

    matches = []
    for i, exp_d in enumerate(exp_d_spacings):
        if np.isnan(exp_d) or exp_d <= 0:
            continue

        pct_errors = np.abs(theo_d - exp_d) / theo_d * 100.0
        best_idx = np.argmin(pct_errors)
        best_err = pct_errors[best_idx]

        if best_err <= tolerance_pct:
            matched_row = theo_df.iloc[best_idx]
            matches.append({
                "Experimental Ring": f"Ring {i+1}",
                "Exp d (Å)": round(exp_d, 4),
                "Matched Phase": phase.name,
                "Theo (hkl)": matched_row["hkl"],
                "Theo d (Å)": round(float(matched_row["d_spacing_angstrom"]), 4),
                "Error (%)": round(float(best_err), 2),
                "Theo Intensity": round(float(matched_row.get("intensity", 100.0)), 1),
                "Status": "Matched"
            })
        else:
            matches.append({
                "Experimental Ring": f"Ring {i+1}",
                "Exp d (Å)": round(exp_d, 4),
                "Matched Phase": phase.name,
                "Theo (hkl)": "Unmatched",
                "Theo d (Å)": np.nan,
                "Error (%)": round(float(best_err), 2),
                "Theo Intensity": 0.0,
                "Status": "Unmatched"
            })

    return pd.DataFrame(matches)


def get_standard_materials() -> Dict[str, Dict[str, Any]]:
    """Built-in library of common TEM calibration standard materials."""
    return {
        "Gold (Au - FCC, a=4.078 Å)": {
            "name": "Gold (Au)",
            "system": "FCC",
            "lattice_a": 4.078,
            "reflections": [
                {"hkl": "(1 1 1)", "d": 2.355, "intensity": 100.0},
                {"hkl": "(2 0 0)", "d": 2.039, "intensity": 52.0},
                {"hkl": "(2 2 0)", "d": 1.442, "intensity": 32.0},
                {"hkl": "(3 1 1)", "d": 1.230, "intensity": 36.0},
                {"hkl": "(2 2 2)", "d": 1.177, "intensity": 12.0},
                {"hkl": "(4 0 0)", "d": 1.020, "intensity": 6.0},
                {"hkl": "(3 3 1)", "d": 0.935, "intensity": 21.0},
                {"hkl": "(4 2 0)", "d": 0.912, "intensity": 20.0},
                {"hkl": "(4 2 2)", "d": 0.832, "intensity": 17.0}
            ]
        },
        "Aluminum (Al - FCC, a=4.049 Å)": {
            "name": "Aluminum (Al)",
            "system": "FCC",
            "lattice_a": 4.049,
            "reflections": [
                {"hkl": "(1 1 1)", "d": 2.338, "intensity": 100.0},
                {"hkl": "(2 0 0)", "d": 2.025, "intensity": 47.0},
                {"hkl": "(2 2 0)", "d": 1.432, "intensity": 22.0},
                {"hkl": "(3 1 1)", "d": 1.221, "intensity": 24.0},
                {"hkl": "(2 2 2)", "d": 1.169, "intensity": 7.0},
                {"hkl": "(4 0 0)", "d": 1.012, "intensity": 3.0}
            ]
        },
        "Silver (Ag - FCC, a=4.086 Å)": {
            "name": "Silver (Ag)",
            "system": "FCC",
            "lattice_a": 4.086,
            "reflections": [
                {"hkl": "(1 1 1)", "d": 2.359, "intensity": 100.0},
                {"hkl": "(2 0 0)", "d": 2.043, "intensity": 40.0},
                {"hkl": "(2 2 0)", "d": 1.445, "intensity": 25.0},
                {"hkl": "(3 1 1)", "d": 1.232, "intensity": 26.0},
                {"hkl": "(2 2 2)", "d": 1.179, "intensity": 9.0}
            ]
        },
        "Silicon (Si - Diamond Cubic, a=5.431 Å)": {
            "name": "Silicon (Si)",
            "system": "Diamond Cubic",
            "lattice_a": 5.431,
            "reflections": [
                {"hkl": "(1 1 1)", "d": 3.136, "intensity": 100.0},
                {"hkl": "(2 2 0)", "d": 1.920, "intensity": 55.0},
                {"hkl": "(3 1 1)", "d": 1.638, "intensity": 30.0},
                {"hkl": "(4 0 0)", "d": 1.358, "intensity": 6.0},
                {"hkl": "(3 3 1)", "d": 1.246, "intensity": 11.0},
                {"hkl": "(4 2 2)", "d": 1.109, "intensity": 12.0}
            ]
        },
        "Titanium Dioxide (TiO2 Anatase - Tetragonal)": {
            "name": "TiO2 (Anatase)",
            "system": "Tetragonal",
            "lattice_a": 3.784,
            "lattice_c": 9.515,
            "reflections": [
                {"hkl": "(1 0 1)", "d": 3.520, "intensity": 100.0},
                {"hkl": "(1 0 3)", "d": 2.431, "intensity": 10.0},
                {"hkl": "(0 0 4)", "d": 2.378, "intensity": 20.0},
                {"hkl": "(1 1 2)", "d": 2.332, "intensity": 10.0},
                {"hkl": "(2 0 0)", "d": 1.892, "intensity": 35.0},
                {"hkl": "(1 0 5)", "d": 1.699, "intensity": 20.0},
                {"hkl": "(2 1 1)", "d": 1.666, "intensity": 20.0}
            ]
        },
        "Iron (alpha-Fe Ferrite - BCC, a=2.866 Å)": {
            "name": "alpha-Fe (Ferrite)",
            "system": "BCC",
            "lattice_a": 2.866,
            "reflections": [
                {"hkl": "(1 1 0)", "d": 2.027, "intensity": 100.0},
                {"hkl": "(2 0 0)", "d": 1.433, "intensity": 19.0},
                {"hkl": "(2 1 1)", "d": 1.170, "intensity": 30.0},
                {"hkl": "(2 2 0)", "d": 1.013, "intensity": 9.0},
                {"hkl": "(3 1 0)", "d": 0.906, "intensity": 12.0},
                {"hkl": "(2 2 2)", "d": 0.827, "intensity": 3.0},
                {"hkl": "(3 2 1)", "d": 0.766, "intensity": 22.0}
            ]
        },
        "Iron Oxide (alpha-Fe2O3 Hematite - Rhombohedral/Hexagonal)": {
            "name": "alpha-Fe2O3 (Hematite)",
            "system": "Hexagonal / Rhombohedral",
            "lattice_a": 5.035,
            "lattice_c": 13.750,
            "reflections": [
                # R-3c reflections regenerated from the crystal structure (forbidden / mislabelled
                # entries removed). Intensities are X-ray style, relative to the strongest line.
                {"hkl": "(0 1 2)", "d": 3.682, "intensity": 26.9},
                {"hkl": "(1 0 4)", "d": 2.700, "intensity": 100.0},
                {"hkl": "(1 1 0)", "d": 2.518, "intensity": 75.3},
                {"hkl": "(0 0 6)", "d": 2.292, "intensity": 2.1},
                {"hkl": "(1 1 3)", "d": 2.207, "intensity": 19.2},
                {"hkl": "(2 0 2)", "d": 2.078, "intensity": 2.1},
                {"hkl": "(0 2 4)", "d": 1.841, "intensity": 43.1},
                {"hkl": "(1 1 6)", "d": 1.695, "intensity": 54.3},
                {"hkl": "(1 2 2)", "d": 1.603, "intensity": 2.9},
                {"hkl": "(0 1 8)", "d": 1.599, "intensity": 10.6},
                {"hkl": "(2 1 4)", "d": 1.486, "intensity": 40.0},
                {"hkl": "(0 3 0)", "d": 1.453, "intensity": 39.0},
                {"hkl": "(2 0 8)", "d": 1.350, "intensity": 4.2},
                {"hkl": "(1 0 10)", "d": 1.311, "intensity": 15.2},
                {"hkl": "(1 1 9)", "d": 1.306, "intensity": 2.4},
                {"hkl": "(2 2 0)", "d": 1.259, "intensity": 10.0},
            ]
        },
        "Iron Oxide (gamma-Fe2O3 Maghemite - Cubic Spinel, a=8.351 Å)": {
            "name": "gamma-Fe2O3 (Maghemite)",
            "system": "Cubic Spinel",
            "lattice_a": 8.351,
            "reflections": [
                {"hkl": "(2 2 0)", "d": 2.953, "intensity": 32.0},
                {"hkl": "(3 1 1)", "d": 2.518, "intensity": 100.0},
                {"hkl": "(2 2 2)", "d": 2.411, "intensity": 8.0},
                {"hkl": "(4 0 0)", "d": 2.088, "intensity": 22.0},
                {"hkl": "(4 2 2)", "d": 1.705, "intensity": 12.0},
                {"hkl": "(5 1 1)", "d": 1.607, "intensity": 30.0},
                {"hkl": "(4 4 0)", "d": 1.476, "intensity": 40.0},
                {"hkl": "(5 3 3)", "d": 1.274, "intensity": 10.0},
                {"hkl": "(6 2 2)", "d": 1.259, "intensity": 8.0}
            ]
        },
        "Iron Oxide (Fe3O4 Magnetite - Cubic Inverse Spinel, a=8.396 Å)": {
            "name": "Fe3O4 (Magnetite)",
            "system": "Cubic Inverse Spinel",
            "lattice_a": 8.396,
            "reflections": [
                {"hkl": "(1 1 1)", "d": 4.847, "intensity": 8.0},
                {"hkl": "(2 2 0)", "d": 2.968, "intensity": 30.0},
                {"hkl": "(3 1 1)", "d": 2.531, "intensity": 100.0},
                {"hkl": "(2 2 2)", "d": 2.424, "intensity": 8.0},
                {"hkl": "(4 0 0)", "d": 2.099, "intensity": 20.0},
                {"hkl": "(4 2 2)", "d": 1.714, "intensity": 10.0},
                {"hkl": "(5 1 1)", "d": 1.616, "intensity": 30.0},
                {"hkl": "(4 4 0)", "d": 1.484, "intensity": 40.0},
                {"hkl": "(5 3 1)", "d": 1.419, "intensity": 5.0},
                {"hkl": "(6 2 0)", "d": 1.328, "intensity": 4.0},
                {"hkl": "(5 3 3)", "d": 1.280, "intensity": 10.0}
            ]
        },
        "Mixed Oxide (FeCr2O4 Chromite Spinel, a=8.379 Å)": {
            "name": "FeCr2O4 (Chromite Spinel)",
            "system": "Cubic Spinel",
            "lattice_a": 8.379,
            "reflections": [
                {"hkl": "(1 1 1)", "d": 4.838, "intensity": 10.0},
                {"hkl": "(2 2 0)", "d": 2.962, "intensity": 35.0},
                {"hkl": "(3 1 1)", "d": 2.526, "intensity": 100.0},
                {"hkl": "(2 2 2)", "d": 2.419, "intensity": 6.0},
                {"hkl": "(4 0 0)", "d": 2.095, "intensity": 22.0},
                {"hkl": "(4 2 2)", "d": 1.710, "intensity": 12.0},
                {"hkl": "(5 1 1)", "d": 1.613, "intensity": 28.0},
                {"hkl": "(4 4 0)", "d": 1.481, "intensity": 42.0}
            ]
        },
        "Mixed Oxide (NiFe2O4 Trevorite Spinel, a=8.339 Å)": {
            "name": "NiFe2O4 (Nickel Ferrite)",
            "system": "Cubic Inverse Spinel",
            "lattice_a": 8.339,
            "reflections": [
                {"hkl": "(1 1 1)", "d": 4.815, "intensity": 9.0},
                {"hkl": "(2 2 0)", "d": 2.948, "intensity": 32.0},
                {"hkl": "(3 1 1)", "d": 2.514, "intensity": 100.0},
                {"hkl": "(2 2 2)", "d": 2.407, "intensity": 7.0},
                {"hkl": "(4 0 0)", "d": 2.085, "intensity": 21.0},
                {"hkl": "(4 2 2)", "d": 1.702, "intensity": 11.0},
                {"hkl": "(5 1 1)", "d": 1.605, "intensity": 29.0},
                {"hkl": "(4 4 0)", "d": 1.474, "intensity": 40.0}
            ]
        },
        "Mixed Oxide (CoFe2O4 Cobalt Ferrite Spinel, a=8.391 Å)": {
            "name": "CoFe2O4 (Cobalt Ferrite)",
            "system": "Cubic Inverse Spinel",
            "lattice_a": 8.391,
            "reflections": [
                {"hkl": "(1 1 1)", "d": 4.845, "intensity": 9.0},
                {"hkl": "(2 2 0)", "d": 2.967, "intensity": 31.0},
                {"hkl": "(3 1 1)", "d": 2.530, "intensity": 100.0},
                {"hkl": "(2 2 2)", "d": 2.422, "intensity": 7.0},
                {"hkl": "(4 0 0)", "d": 2.098, "intensity": 21.0},
                {"hkl": "(4 2 2)", "d": 1.713, "intensity": 11.0},
                {"hkl": "(5 1 1)", "d": 1.615, "intensity": 30.0},
                {"hkl": "(4 4 0)", "d": 1.483, "intensity": 41.0}
            ]
        },
        "Iron-Chromium (403 / 410 Martensitic Steel - BCC, a=2.870 Å)": {
            "name": "403 Steel (bcc Fe-Cr)",
            "system": "BCC",
            "lattice_a": 2.87,
            "reflections": [
                # Martensite / ferrite are both bcc with a ~ 2.86-2.88 A: SAED cannot separate them.
                {"hkl": "(1 1 0)", "d": 2.029, "intensity": 100.0},
                {"hkl": "(2 0 0)", "d": 1.435, "intensity": 16.7},
                {"hkl": "(2 1 1)", "d": 1.172, "intensity": 32.8},
                {"hkl": "(2 2 0)", "d": 1.015, "intensity": 9.4},
                {"hkl": "(3 1 0)", "d": 0.908, "intensity": 11.9},
                {"hkl": "(2 2 2)", "d": 0.828, "intensity": 2.7},
            ]
        },
        "Iron (gamma-Fe Austenite - FCC, a=3.590 Å)": {
            "name": "Austenite (fcc Fe-Cr-Ni)",
            "system": "FCC",
            "lattice_a": 3.59,
            "reflections": [
                # Retained austenite. Its (111) at 2.07 A is within ~2% of bcc (110) at 2.03 A: use (200) at 1.79 A to tell them apart.
                {"hkl": "(1 1 1)", "d": 2.073, "intensity": 100.0},
                {"hkl": "(2 0 0)", "d": 1.795, "intensity": 48.6},
                {"hkl": "(2 2 0)", "d": 1.269, "intensity": 30.9},
                {"hkl": "(3 1 1)", "d": 1.082, "intensity": 34.3},
                {"hkl": "(2 2 2)", "d": 1.036, "intensity": 9.6},
                {"hkl": "(4 0 0)", "d": 0.897, "intensity": 4.0},
                {"hkl": "(3 3 1)", "d": 0.824, "intensity": 11.1},
                {"hkl": "(4 2 0)", "d": 0.803, "intensity": 9.9},
            ]
        },
        "Iron Oxide (FeO Wustite - Rock salt, a=4.300 Å)": {
            "name": "FeO (Wustite)",
            "system": "FCC",
            "lattice_a": 4.3,
            "reflections": [
                # Fe(1-x)O: the lattice parameter changes with vacancy content (about 4.28-4.31 A).
                {"hkl": "(1 1 1)", "d": 2.483, "intensity": 64.3},
                {"hkl": "(2 0 0)", "d": 2.150, "intensity": 100.0},
                {"hkl": "(2 2 0)", "d": 1.520, "intensity": 65.0},
                {"hkl": "(3 1 1)", "d": 1.296, "intensity": 30.2},
                {"hkl": "(2 2 2)", "d": 1.241, "intensity": 20.9},
                {"hkl": "(4 0 0)", "d": 1.075, "intensity": 9.0},
                {"hkl": "(3 3 1)", "d": 0.986, "intensity": 11.2},
                {"hkl": "(4 2 0)", "d": 0.962, "intensity": 23.0},
                {"hkl": "(4 2 2)", "d": 0.878, "intensity": 15.7},
                {"hkl": "(5 1 1)", "d": 0.828, "intensity": 7.1},
            ]
        },
        "Chromium Oxide (Cr2O3 Eskolaite - Rhombohedral/Hexagonal)": {
            "name": "Cr2O3 (Eskolaite)",
            "system": "Hexagonal / Rhombohedral",
            "lattice_a": 4.959,
            "lattice_c": 13.594,
            "reflections": [
                # R-3c, corundum type. Close to hematite: separate with (116) 1.67 vs 1.70 A and (104) 2.67 vs 2.70 A.
                {"hkl": "(0 1 2)", "d": 3.631, "intensity": 61.2},
                {"hkl": "(1 0 4)", "d": 2.665, "intensity": 93.9},
                {"hkl": "(1 1 0)", "d": 2.479, "intensity": 89.0},
                {"hkl": "(0 0 6)", "d": 2.266, "intensity": 6.4},
                {"hkl": "(1 1 3)", "d": 2.175, "intensity": 28.6},
                {"hkl": "(2 0 2)", "d": 2.048, "intensity": 5.8},
                {"hkl": "(0 2 4)", "d": 1.815, "intensity": 41.4},
                {"hkl": "(1 1 6)", "d": 1.673, "intensity": 100.0},
                {"hkl": "(0 1 8)", "d": 1.580, "intensity": 1.9},
                {"hkl": "(1 2 2)", "d": 1.579, "intensity": 6.9},
                {"hkl": "(2 1 4)", "d": 1.465, "intensity": 36.7},
                {"hkl": "(0 3 0)", "d": 1.432, "intensity": 48.4},
                {"hkl": "(1 0 10)", "d": 1.296, "intensity": 19.7},
                {"hkl": "(1 1 9)", "d": 1.290, "intensity": 3.6},
                {"hkl": "(2 2 0)", "d": 1.240, "intensity": 11.1},
                {"hkl": "(0 3 6)", "d": 1.210, "intensity": 8.2},
            ]
        }
    }


def create_phase_from_standard(std_key: str) -> CIFPhase:
    """Create a CIFPhase object from a standard material definition."""
    standards = get_standard_materials()
    data = standards[std_key]
    phase = CIFPhase(name=data["name"], source="standard_library")
    phase.crystal_system = data.get("system", "")
    phase.cell_params = {k: v for k, v in (("a", data.get("lattice_a")), ("c", data.get("lattice_c"))) if v}
    phase.reflections = [
        {
            "h": 0, "k": 0, "l": 0,
            "hkl": item["hkl"],
            "d_spacing_angstrom": item["d"],
            "intensity": item["intensity"]
        }
        for item in data["reflections"]
    ]
    return phase
