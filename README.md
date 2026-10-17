# SAED Ring & Spot Pattern Analyzer

A modular, production-ready Streamlit application for TEM Selected Area Electron Diffraction (SAED) quantitative pattern analysis.

## Key Features

1. **Image Upload & Interactive Cropping**:
   - Supports PNG, JPG, TIFF, and DigitalMicrograph `.dm3`/`.dm4` file reading.
   - Interactive percentile contrast stretching, gamma correction, color inversion, and ROI cropping.
2. **Sub-pixel Auto-Centering**:
   - Direct beam center detection using circular Hough transform, intensity centroid (weighted center of mass), and smoothed maximum intensity peak.
   - Interactive coordinate fine-tuning with concentric guide overlays.
3. **Full Pattern Analysis**:
   - Azimuthally integrated radial profile $I(r)$ calculation.
   - Sector/wedge integration to omit beam-stop artifacts.
   - Rolling ball diffuse background subtraction.
   - Peak detection with sub-pixel parabolic peak interpolation.
   - Accurate calculation of reciprocal radii $q$ ($\text{nm}^{-1}$), interplanar $d$-spacings ($\text{Å}$ and $\text{nm}$), FWHM, and intensities.
4. **CIF Database & Crystal Phase Matching**:
   - Parses local `.cif` crystallographic files using `pymatgen` or `gemmi`, with a robust built-in pure Python fallback parser.
   - Comprehensive reference library for standard materials:
     - Metals & Semiconductors: Gold (Au), Aluminum (Al), Silver (Ag), Silicon (Si).
     - Iron & Iron Oxides: $\alpha$-Fe (Ferrite BCC), Hematite ($\alpha$-$\text{Fe}_2\text{O}_3$), Maghemite ($\gamma$-$\text{Fe}_2\text{O}_3$), Magnetite ($\text{Fe}_3\text{O}_4$).
     - Mixed Spinel Oxides: Chromite spinel ($\text{FeCr}_2\text{O}_4$), Trevorite/Nickel ferrite ($\text{NiFe}_2\text{O}_4$), Cobalt ferrite ($\text{CoFe}_2\text{O}_4$), and Anatase ($\text{TiO}_2$).
   - Computes theoretical reflections $(hkl)$, $d$-spacings, and relative intensities.
   - Automated matching of experimental rings to candidate crystal structures within customizable percentage error tolerances.
   - Theoretical reflection stick-pattern overlay on 1D radial profiles.
5. **Publication Report Generator**:
   - Compiles analysis into a publication-ready summary table showing measured $d$-spacings, calculated $d$-spacings, relative error (%), indexed $(hkl)$ planes, and theoretical reflection intensities.
   - Interactive microscope metadata inputs: accelerating voltage ($\text{kV}$), camera length ($\text{mm}$), calibration reference standard, and zone axis / pattern geometry.
   - Copy-pasteable Markdown table and LaTeX table (`booktabs` syntax) generation and one-click download.
   - Formatted publication paragraph summarizing phase identification and zone axis indexing ready for manuscript results or methodology sections.
6. **Data & Report Export**:
   - Download radial profiles (CSV), ring metrics (CSV), phase identification matches (CSV), and comprehensive summary reports (JSON).

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```
