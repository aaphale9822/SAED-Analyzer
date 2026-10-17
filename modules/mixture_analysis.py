"""
mixture_analysis.py
===================
Mixed-phase ring assignment for SAED ring patterns.

Answers: "which measured ring belongs to which phase?" when several candidate phases
are loaded (e.g. Fe3O4 + Fe2O3 + bcc steel), and states honestly what SAED can and
cannot separate.

Key ideas
---------
* Every ring is matched against every phase (nearest reflection within the tolerance).
* Phases whose reflection lists overlap almost completely (e.g. magnetite, maghemite,
  chromite: same spinel lattice within ~0.5 %) are merged into one INDISTINGUISHABLE
  GROUP. Evidence is then reported per group, not per phase.
* A ring is "unique" if only one group matches it, "shared" if several groups do, and
  "unexplained" if none does.
* A group is only called "supported" if it has at least two unique rings and its
  strongest expected reflections are not missing.
* Ring intensity is NOT used to estimate phase fractions (texture, thickness and
  multiple scattering make that unreliable in SAED).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _ref_arrays(phase, min_intensity: float):
    """Reflection arrays (d, intensity, hkl) kept above a minimum relative intensity."""
    d, inten, hkl = [], [], []
    for r in phase.reflections:
        if r.get("intensity", 100.0) >= min_intensity and r["d_spacing_angstrom"] > 0:
            d.append(float(r["d_spacing_angstrom"]))
            inten.append(float(r.get("intensity", 100.0)))
            hkl.append(str(r["hkl"]))
    return np.array(d), np.array(inten), hkl


def _overlap_fraction(d_a: np.ndarray, d_b: np.ndarray, tol_pct: float) -> float:
    """Fraction of reflections in A that have a reflection in B within tol_pct."""
    if len(d_a) == 0 or len(d_b) == 0:
        return 0.0
    hits = [np.min(np.abs(d_b - d) / d * 100.0) <= tol_pct for d in d_a]
    return float(np.mean(hits))


def build_groups(
    phases: Dict[str, Any], group_tol_pct: float = 1.0, min_intensity: float = 10.0
) -> Dict[str, int]:
    """Union phases whose reflections overlap (both ways >= 80 %) within group_tol_pct."""
    names = list(phases)
    refs = {n: _ref_arrays(phases[n], min_intensity)[0] for n in names}
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (_overlap_fraction(refs[a], refs[b], group_tol_pct) >= 0.8
                    and _overlap_fraction(refs[b], refs[a], group_tol_pct) >= 0.8):
                parent[find(a)] = find(b)

    roots = {}
    gid = {}
    for n in names:
        r = find(n)
        roots.setdefault(r, len(roots))
        gid[n] = roots[r]
    return gid


def assign_rings_to_phases(
    exp_d: List[float],
    phases: Dict[str, Any],
    tol_pct: float = 3.0,
    min_intensity: float = 5.0,
    strong_intensity: float = 30.0,
    group_tol_pct: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Assign every measured ring to the loaded phases.

    Returns
    -------
    ring_df   : one row per ring (matches, assignment: Unique / Shared / Unexplained)
    group_df  : one row per indistinguishable phase group (evidence summary)
    warnings  : plain-language cautions
    """
    names = list(phases)
    exp = [float(x) for x in exp_d if x is not None and np.isfinite(x) and x > 0]
    warnings: List[str] = []
    if not names or not exp:
        return pd.DataFrame(), pd.DataFrame(), ["Load at least one phase and detect rings first."]

    refs = {n: _ref_arrays(phases[n], min_intensity) for n in names}
    gid = build_groups(phases, group_tol_pct)
    members: Dict[int, List[str]] = {}
    for n, g in gid.items():
        members.setdefault(g, []).append(n)
    glabel = {g: " / ".join(m) for g, m in members.items()}

    # ---- per-ring matching -------------------------------------------------
    ring_rows = []
    ring_groups: List[set] = []
    for i, d in enumerate(exp, 1):
        matches = {}
        for n in names:
            td, ti, th = refs[n]
            if len(td) == 0:
                continue
            err = (d - td) / td * 100.0
            j = int(np.argmin(np.abs(err)))
            if abs(err[j]) <= tol_pct:
                matches[n] = (th[j], float(err[j]))
        groups_hit = {gid[n] for n in matches}
        ring_groups.append(groups_hit)

        if not groups_hit:
            status = "Unexplained"
        elif len(groups_hit) == 1:
            g = next(iter(groups_hit))
            status = f"Unique: {glabel[g]}"
        else:
            status = "Shared: " + " + ".join(glabel[g] for g in sorted(groups_hit))
        ring_rows.append({
            "Ring": f"Ring {i}",
            "d exp (Å)": round(d, 4),
            "Assignment": status,
            "Matches (hkl, error %)": "; ".join(
                f"{n} {h} {e:+.1f}%" for n, (h, e) in matches.items()) or "—",
        })
    ring_df = pd.DataFrame(ring_rows)

    # ---- per-group evidence ------------------------------------------------
    lo, hi = min(exp) * (1 - tol_pct / 100), max(exp) * (1 + tol_pct / 100)
    group_rows = []
    for g, mem in sorted(members.items()):
        matched = [k for k, gs in enumerate(ring_groups) if g in gs]
        unique = [k for k in matched if ring_groups[k] == {g}]
        shared = [k for k in matched if len(ring_groups[k]) > 1]

        # strong reflections expected inside the measured d-range (union over members)
        strong_d = []
        for n in mem:
            td, ti, _ = refs[n]
            for dd, ii in zip(td, ti):
                if ii >= strong_intensity and lo <= dd <= hi:
                    if not any(abs(dd - s) / s * 100 <= group_tol_pct for s in strong_d):
                        strong_d.append(dd)
        missing = [s for s in strong_d
                   if not any(abs(e - s) / s * 100 <= tol_pct for e in exp)]

        n_exp = len(strong_d)
        if not matched:
            verdict = "Not supported (no rings match)"
        elif n_exp and len(missing) / n_exp > 0.5:
            verdict = "Weak (most strong lines are absent)"
        elif len(unique) >= 2 and len(missing) <= 1:
            verdict = "Supported"
        elif len(unique) == 1:
            verdict = "Weak support (only 1 unique ring)"
        else:
            verdict = "Not demonstrated (all its rings are shared)"

        group_rows.append({
            "Phase group": glabel[g],
            "Rings matched": len(matched),
            "Unique rings": len(unique),
            "Shared rings": len(shared),
            "Strong lines expected": n_exp,
            "Strong lines missing": ", ".join(f"{m:.2f} Å" for m in missing) or "none",
            "Evidence": verdict,
        })
        if len(mem) > 1 and matched:
            warnings.append(
                f"{' / '.join(mem)} cannot be told apart by SAED: their reflections agree "
                f"within {group_tol_pct:g}%. Report them as one group, not as separate phases.")
        if matched and not unique and len(matched) > 0:
            warnings.append(
                f"{glabel[g]}: no ring is unique to this phase, so its presence is not demonstrated.")

    group_df = pd.DataFrame(group_rows)

    # ---- global warnings ---------------------------------------------------
    n_unexp = int((ring_df["Assignment"] == "Unexplained").sum())
    if n_unexp:
        warnings.append(
            f"{n_unexp} of {len(exp)} rings are unexplained: check for a missing phase "
            f"(e.g. FeO, Cr2O3, carbide), a superlattice, the carbon support, or a calibration error.")

    # near-overlapping groups: shared rings caused only by a wide tolerance
    glist = sorted(members)
    for ai, a in enumerate(glist):
        for b in glist[ai + 1:]:
            dA = np.concatenate([refs[n][0] for n in members[a]])
            dB = np.concatenate([refs[n][0] for n in members[b]])
            if (_overlap_fraction(dA, dB, tol_pct) >= 0.8
                    and _overlap_fraction(dB, dA, tol_pct) >= 0.8
                    and any({a, b} <= gs for gs in ring_groups)):
                warnings.append(
                    f"{glabel[a]} and {glabel[b]} overlap within the {tol_pct:g}% tolerance. "
                    f"Lowering the tolerance only helps if your calibration is accurate to better than that.")
    return ring_df, group_df, list(dict.fromkeys(warnings))
