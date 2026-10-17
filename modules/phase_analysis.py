"""
phase_analysis.py
=================
Advanced multi-phase SAED analysis utilities:

1. run_all_phases_comparison()  — Match experimental d-spacings against ALL loaded
   phases simultaneously and return a ranked leaderboard DataFrame.

2. compute_phase_confidence()   — Single 0-100 confidence score combining:
      • Matched fraction (most important, 50%)
      • Average error of matched rings (25%)
      • Strong-line coverage: did the highest-intensity reflections match? (25%)

3. check_systematic_absences()  — Validates each matched hkl against known space-group
   extinction rules and flags forbidden reflections with a warning string.

4. find_discriminating_rings()  — Identifies rings that are UNIQUE to one phase
   (no other loaded phase has a reflection within tolerance at that d-spacing).

5. generate_phase_narrative()   — Auto-generates a scientific explanation paragraph
   explaining WHY the best-matched phase wins over all competitors.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Systematic absence rules by crystal / space-group family
# ---------------------------------------------------------------------------
# Each entry maps a short tag to a callable(h, k, l) -> bool where True means
# the reflection IS allowed (present).  False = systematically absent.

def _bcc_allowed(h: int, k: int, l: int) -> bool:
    """BCC: h+k+l must be even."""
    return (h + k + l) % 2 == 0

def _fcc_allowed(h: int, k: int, l: int) -> bool:
    """FCC: h, k, l must be all odd or all even."""
    parities = {abs(h) % 2, abs(k) % 2, abs(l) % 2}
    return len(parities) == 1

def _diamond_cubic_allowed(h: int, k: int, l: int) -> bool:
    """Diamond cubic: FCC conditions + h+k+l != 4n+2 when unmixed."""
    if not _fcc_allowed(h, k, l):
        return False
    s = h + k + l
    return not (s % 4 == 2)

def _hexagonal_rhombohedral_allowed(h: int, k: int, l: int) -> bool:
    """Hexagonal R-lattice (rhombohedral, R3c type): -h+k+l divisible by 3."""
    # R-centering: -h+k+l = 3n
    return (-h + k + l) % 3 == 0

def _spinel_allowed(h: int, k: int, l: int) -> bool:
    """
    Cubic spinel, space group Fd-3m (No. 227) reflection conditions:
      hkl : h, k, l all odd or all even   (F-centring)
      0kl : k + l = 4n                    (d-glide)
    so (222), (442), (622) ARE allowed, while (200), (420), (600) are not.
    """
    if not _fcc_allowed(h, k, l):
        return False
    nz = [abs(x) for x in (h, k, l)]
    if 0 in nz:                       # 0kl-type reflections
        return sum(nz) % 4 == 0
    return True


def _always_allowed(h: int, k: int, l: int) -> bool:
    return True


SYSTEMATIC_ABSENCE_RULES: Dict[str, Any] = {
    "BCC":                     _bcc_allowed,
    "FCC":                     _fcc_allowed,
    "Diamond Cubic":           _diamond_cubic_allowed,
    "Hexagonal / Rhombohedral": _hexagonal_rhombohedral_allowed,
    "Hexagonal":               _always_allowed,   # P63/mmc — only some conditions
    "Cubic Spinel":            _spinel_allowed,
    "Cubic Inverse Spinel":    _spinel_allowed,
    "Tetragonal":              _always_allowed,
    "Orthorhombic":            _always_allowed,
}


def check_systematic_absences(
    matching_df: pd.DataFrame,
    crystal_system: str
) -> pd.DataFrame:
    """
    Add a 'Systematic Absence Warning' column to a matching DataFrame.

    Parameters
    ----------
    matching_df : pd.DataFrame
        Output of match_experimental_to_phase().
    crystal_system : str
        Crystal system string from CIFPhase.crystal_system.

    Returns
    -------
    pd.DataFrame with new column 'SA Warning'.
    """
    rule = SYSTEMATIC_ABSENCE_RULES.get(crystal_system, _always_allowed)
    warnings = []

    for _, row in matching_df.iterrows():
        hkl_str = str(row.get("Theo (hkl)", ""))
        if row.get("Status", "") != "Matched" or hkl_str in ("Unmatched", "", "nan"):
            warnings.append("")
            continue

        # Parse hkl string like "(0 1 1)" or "(011)"
        nums = [int(x) for x in hkl_str.replace("(", "").replace(")", "").split()]
        if len(nums) == 3:
            h, k, l_val = nums
            if not rule(h, k, l_val):
                warnings.append(
                    f"⚠️ ({h} {k} {l_val}) is FORBIDDEN in {crystal_system} — "
                    f"possible mis-assignment or super-lattice"
                )
            else:
                warnings.append("✅ Allowed")
        else:
            warnings.append("")

    result = matching_df.copy()
    result["SA Warning"] = warnings
    return result


# ---------------------------------------------------------------------------
# Multi-phase comparison
# ---------------------------------------------------------------------------

def match_single_phase_quiet(
    exp_d_spacings: List[float],
    phase,  # CIFPhase
    tolerance_pct: float = 3.5,
) -> Tuple[int, int, float, float]:
    """
    Quick match returning (n_matched, n_total, avg_error, strong_line_coverage).
    Does NOT build a full DataFrame for speed.
    """
    if not phase.reflections or not exp_d_spacings:
        return 0, len(exp_d_spacings), 999.0, 0.0

    theo_d   = np.array([r["d_spacing_angstrom"] for r in phase.reflections], dtype=float)
    theo_int = np.array([r.get("intensity", 50.0) for r in phase.reflections], dtype=float)

    n_matched = 0
    errors = []
    matched_intensities = []

    for exp_d in exp_d_spacings:
        if np.isnan(exp_d) or exp_d <= 0:
            continue
        pct_errors = np.abs(theo_d - exp_d) / theo_d * 100.0
        best_idx = int(np.argmin(pct_errors))
        best_err = float(pct_errors[best_idx])
        if best_err <= tolerance_pct:
            n_matched += 1
            errors.append(best_err)
            matched_intensities.append(float(theo_int[best_idx]))

    avg_err = float(np.mean(errors)) if errors else 999.0

    # Strong-line coverage: fraction of top-3 strongest theoretical lines that matched
    top3_idx = np.argsort(theo_int)[-3:]
    top3_d   = theo_d[top3_idx]
    strong_matched = 0
    for td in top3_d:
        for ed in exp_d_spacings:
            if abs(td - ed) / td * 100.0 <= tolerance_pct:
                strong_matched += 1
                break
    strong_coverage = strong_matched / max(1, len(top3_idx))

    return n_matched, len(exp_d_spacings), avg_err, strong_coverage


def compute_phase_confidence(
    n_matched: int,
    n_total: int,
    avg_error: float,
    strong_coverage: float,
    max_error: float = 5.0,
) -> float:
    """
    Compute a single phase confidence score (0–100).

    Weights:
      - Match fraction   50%   (rings matched / total rings)
      - Error quality    25%   (1 – avg_error / max_error), clipped to [0,1]
      - Strong coverage  25%   (fraction of strongest lines matched)
    """
    match_frac = n_matched / max(1, n_total)
    error_score = max(0.0, 1.0 - avg_error / max_error) if avg_error < 999.0 else 0.0
    score = 0.50 * match_frac + 0.25 * error_score + 0.25 * strong_coverage
    return round(float(score * 100.0), 1)


def run_all_phases_comparison(
    exp_d_spacings: List[float],
    phases: Dict[str, Any],   # {name: CIFPhase}
    tolerance_pct: float = 3.5,
) -> pd.DataFrame:
    """
    Run multi-phase comparison and return a ranked leaderboard DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        Rank, Phase, Rings Matched, Total Rings, Match Fraction (%),
        Avg Error (%), Strong-Line Coverage (%), Confidence Score (0-100),
        Verdict
    """
    rows = []
    for phase_name, phase in phases.items():
        n_matched, n_total, avg_err, strong_cov = match_single_phase_quiet(
            exp_d_spacings, phase, tolerance_pct
        )
        confidence = compute_phase_confidence(n_matched, n_total, avg_err, strong_cov)
        match_frac = n_matched / max(1, n_total) * 100.0

        rows.append({
            "Phase": phase_name,
            "Rings Matched": n_matched,
            "Total Rings": n_total,
            "Match Fraction (%)": round(match_frac, 1),
            "Avg Error (%)": round(avg_err, 2) if avg_err < 999.0 else "—",
            "Strong-Line Coverage (%)": round(strong_cov * 100.0, 1),
            "Confidence Score": confidence,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("Confidence Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", ["🥇", "🥈", "🥉"] + [""] * max(0, len(df) - 3))

    # Verdict column
    verdicts = []
    for i, row in df.iterrows():
        c = row["Confidence Score"]
        if c >= 75:
            verdicts.append("✅ Strong match — confirmed phase")
        elif c >= 50:
            verdicts.append("⚠️ Moderate — possible secondary phase")
        elif c >= 25:
            verdicts.append("❌ Weak — coincidental only")
        else:
            verdicts.append("❌ Ruled out")
    df["Verdict"] = verdicts

    return df


# ---------------------------------------------------------------------------
# Discriminating reflections
# ---------------------------------------------------------------------------

def find_discriminating_rings(
    exp_d_spacings: List[float],
    phases: Dict[str, Any],
    best_phase_name: str,
    tolerance_pct: float = 3.5,
) -> List[Dict[str, Any]]:
    """
    Find experimental rings that:
    (a) Match the best phase, AND
    (b) Are NOT matched by any other loaded phase within tolerance.

    These are the "fingerprint" rings that uniquely confirm the best phase.

    Returns
    -------
    List of dicts with keys: exp_d, hkl (from best phase), unique (bool),
    competing_phases (list of phase names that also match).
    """
    best_phase = phases.get(best_phase_name)
    if best_phase is None:
        return []

    best_theo_d   = np.array([r["d_spacing_angstrom"] for r in best_phase.reflections])
    best_theo_hkl = [r["hkl"] for r in best_phase.reflections]

    other_phases = {k: v for k, v in phases.items() if k != best_phase_name}

    results = []
    for exp_d in exp_d_spacings:
        if np.isnan(exp_d) or exp_d <= 0:
            continue

        # Check match to best phase
        pct_errors_best = np.abs(best_theo_d - exp_d) / best_theo_d * 100.0
        best_idx = int(np.argmin(pct_errors_best))
        best_err = float(pct_errors_best[best_idx])

        if best_err > tolerance_pct:
            continue  # not matched by best phase — skip

        hkl_label = best_theo_hkl[best_idx]

        # Check which other phases ALSO match this d-spacing
        competing = []
        for other_name, other_phase in other_phases.items():
            other_d = np.array([r["d_spacing_angstrom"] for r in other_phase.reflections])
            if len(other_d) == 0:
                continue
            errs = np.abs(other_d - exp_d) / other_d * 100.0
            if np.min(errs) <= tolerance_pct:
                competing.append(other_name)

        results.append({
            "exp_d": round(exp_d, 4),
            "hkl": hkl_label,
            "best_error": round(best_err, 2),
            "unique": len(competing) == 0,
            "competing_phases": competing,
        })

    return results


# ---------------------------------------------------------------------------
# Auto-narrative generator
# ---------------------------------------------------------------------------

def generate_phase_narrative(
    leaderboard_df: pd.DataFrame,
    discriminating_rings: List[Dict[str, Any]],
    best_phase_name: str,
    crystal_system: str = "",
    space_group: str = "",
) -> str:
    """
    Generate a scientific narrative paragraph explaining the phase identification
    result, why the best phase wins, and why alternatives are ruled out.
    """
    if leaderboard_df.empty:
        return "No phase comparison data available."

    best_row = leaderboard_df.iloc[0]
    n_matched = best_row["Rings Matched"]
    n_total   = best_row["Total Rings"]
    conf      = best_row["Confidence Score"]
    avg_err_raw = best_row["Avg Error (%)"]
    avg_err   = f"{avg_err_raw}%" if avg_err_raw != "—" else "N/A"

    # Unique fingerprint rings
    unique_rings = [r for r in discriminating_rings if r["unique"]]
    unique_hkl   = ", ".join([r["hkl"] for r in unique_rings[:5]])
    unique_d     = ", ".join([f"{r['exp_d']:.2f} Å" for r in unique_rings[:5]])

    # Runner-up phases
    runner_ups = []
    for _, row in leaderboard_df.iloc[1:].iterrows():
        runner_ups.append(
            f"{row['Phase']} ({row['Rings Matched']}/{row['Total Rings']} rings, "
            f"score={row['Confidence Score']})"
        )

    sys_info = f" ({crystal_system}, {space_group})" if crystal_system or space_group else ""

    lines = [
        f"SAED phase analysis identifies **{best_phase_name}**{sys_info} as the dominant "
        f"crystalline phase, achieving a confidence score of {conf}/100 with "
        f"{n_matched} of {n_total} experimental diffraction rings matched within "
        f"the specified tolerance (average error: {avg_err}).",
    ]

    if unique_rings:
        lines.append(
            f"\n\nThe identification is supported by {len(unique_rings)} discriminating "
            f"reflection(s) unique to this phase: planes {unique_hkl} at measured "
            f"d-spacings of {unique_d}. These reflections are absent in all other "
            f"candidate phases evaluated and serve as unambiguous crystallographic "
            f"fingerprints."
        )

    if runner_ups:
        lines.append(
            f"\n\nAlternative phases were evaluated and ranked lower: "
            + "; ".join(runner_ups[:3])
            + ". These phases either failed to account for the majority of observed "
            "rings or produced unacceptably high matching errors, ruling them out as "
            "primary phases. Some coincidental d-spacing overlaps may still occur "
            "but do not constitute a phase identification."
        )

    return "".join(lines)
