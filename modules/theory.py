"""
theory.py
=========
"Theory & Equations" tab for the SAED analyser.

Every section is tagged:
    ✅ used by the app          - the app computes exactly this
    📘 reference                - background theory, not computed by the app
Sections are collapsed expanders so the tab stays short on a phone.
"""

import streamlit as st


def render_theory():
    st.subheader("📚 Theory & Equations")
    st.caption(
        "Short reference for how a ring pattern becomes d-spacings and phases.  "
        "✅ = used by this app   |   📘 = background, not computed by the app."
    )

    # ------------------------------------------------------------------ 1
    with st.expander("1. Why rings form: Bragg's law and the pattern geometry  📘", expanded=True):
        st.markdown(
            "Electrons are diffracted by atomic planes of spacing **d** at angles satisfying Bragg's law. "
            "In a polycrystalline sample every orientation is present, so each (hkl) plane family "
            "gives a full ring."
        )
        st.latex(r"\lambda = 2\,d\,\sin\theta")
        st.markdown(
            "At TEM voltages λ is tiny (about 0.02 Å at 200 kV), so θ is only a few milliradians and "
            "sin θ ≈ θ. The ring radius **R** on a detector at camera length **L** is then"
        )
        st.latex(r"R = L\tan 2\theta \;\approx\; \frac{\lambda L}{d}\qquad\Longrightarrow\qquad d=\frac{\lambda L}{R}")
        st.markdown(
            "So **large d gives a small ring, small d gives a large ring**: R is proportional to 1/d, "
            "which is why SAED patterns live in reciprocal space.  \n"
            "The quantity **q = 1/d** (in nm⁻¹) is the distance from the centre in reciprocal space."
        )

    # ------------------------------------------------------------------ 2
    with st.expander("2. Electron wavelength  ✅"):
        st.markdown("Relativistic de Broglie wavelength for accelerating voltage V (volts), as used in the camera-length calibration:")
        st.latex(r"\lambda\,[\text{Å}] = \frac{12.2643}{\sqrt{V\,(1 + 0.978466\times10^{-6}\,V)}}")
        st.table({
            "Voltage": ["80 kV", "120 kV", "200 kV", "300 kV"],
            "λ (Å)": ["0.0418", "0.0335", "0.0251", "0.0197"],
        })

    # ------------------------------------------------------------------ 3
    with st.expander("3. Calibration: pixels to nm⁻¹  ✅"):
        st.markdown(
            "The app stores the calibration as a **scale s in nm⁻¹ per pixel**. Three ways to obtain it:"
        )
        st.markdown("**a) Direct / scale bar** (a bar of known reciprocal length covering N pixels):")
        st.latex(r"s = \frac{\text{bar value (nm}^{-1})}{\text{bar length (px)}}")
        st.markdown("**b) Voltage, camera length and pixel size** (p = detector pixel size):")
        st.latex(r"s = \frac{p}{\lambda\,L}")
        st.markdown("**c) Known standard ring** (e.g. Au (111), d = 2.355 Å, measured at R px):")
        st.latex(r"s = \frac{1/d_{\text{std}}}{R_{\text{px}}} = \frac{10/d_{\text{std}}[\text{Å}]}{R_{\text{px}}}")
        st.info(
            "**Converting from a scale bar.** A bar labelled 5 nm⁻¹ that measures 105 px gives "
            "s = 5/105 = **0.0476 nm⁻¹/px** (equivalently 21 px per nm⁻¹). "
            "Always measure the bar length on the *same* (cropped or resized) image you analyse."
        )
        st.warning(
            "The default scale in the sidebar is only a placeholder. If you leave it unchanged, every "
            "d-spacing is wrong by the same factor. Method (c), a gold or aluminium pattern recorded at the "
            "same settings, is the most accurate."
        )

    # ------------------------------------------------------------------ 4
    with st.expander("4. From ring radius to d-spacing  ✅"):
        st.latex(r"q = R_{\text{px}}\cdot s\quad[\text{nm}^{-1}]\qquad d\,[\text{nm}] = \frac{1}{q}\qquad d\,[\text{Å}] = \frac{10}{q}")
        st.markdown(
            "This is the **crystallographic convention** (q = 1/d). Some software uses q = 2π/d; "
            "then d = 2π/q, which changes every d-spacing by a factor of 2π. Do not mix the two."
        )
        st.markdown("**Worked example:** s = 0.0476 nm⁻¹/px, ring at R = 58 px:")
        st.latex(r"q = 58\times0.0476 = 2.76\ \text{nm}^{-1}\;\;\Rightarrow\;\; d = \frac{1}{2.76} = 0.362\ \text{nm} = 3.62\ \text{Å}")

    # ------------------------------------------------------------------ 5
    with st.expander("5. Peak position, width and background  📘"):
        st.markdown(
            "The azimuthally averaged intensity I(r) is background-corrected (a smooth diffuse "
            "component from the amorphous support and inelastic scattering is removed), and each ring "
            "appears as a peak. A common model for a peak is a Gaussian on a local background:"
        )
        st.latex(r"I(r) = A\,\exp\!\left[-\frac{(r-r_0)^2}{2\sigma^2}\right] + c\qquad \text{FWHM} = 2\sqrt{2\ln 2}\,\sigma \approx 2.355\,\sigma")
        st.markdown(
            "The ring radius is r₀, and the width (FWHM) is reported in pixels in the results table. "
            "Which exact detection and fitting routine is used is defined in `modules/radial_profile.py`."
        )
        st.markdown(
            "**Which azimuthal average?**  \n"
            "- *Mean*: robust for smooth rings, but dilutes isolated bright spots.  \n"
            "- *90th percentile / Maximum*: keeps spots, but noisier. Maximum can create false rings on ring patterns."
        )

    # ------------------------------------------------------------------ 6
    with st.expander("6. Uncertainty of a d-spacing  📘"):
        st.latex(r"\frac{\delta d}{d} = \sqrt{\left(\frac{\delta R}{R}\right)^2 + \left(\frac{\delta s}{s}\right)^2 + \left(\frac{\delta c}{R}\right)^2}")
        st.markdown(
            "δR = error in measuring the ring radius, δs = calibration error, δc = error in the centre position. "
            "Because d ∝ 1/R, a ring at R = 100 px measured to ±1 px already has a 1 % uncertainty, and "
            "a centre error of a few pixels or a 3 % scale-bar error dominates it.  \n"
            "**Typical SAED accuracy is 1 to 3 %**, much poorer than XRD. A shift of less than this is not evidence of strain, "
            "vacancies or a different phase."
        )

    # ------------------------------------------------------------------ 7
    with st.expander("7. Cubic ring ratios and lattice type  ✅"):
        st.markdown("For a cubic crystal with lattice parameter a:")
        st.latex(r"d_{hkl} = \frac{a}{\sqrt{h^2+k^2+l^2}} = \frac{a}{\sqrt{N}}\qquad\Longrightarrow\qquad \left(\frac{R_n}{R_1}\right)^2 = \frac{N_n}{N_1}")
        st.markdown(
            "The ratios of squared ring radii are the ratios of the integers N = h²+k²+l². Which values of N "
            "occur depends on the lattice:"
        )
        st.table({
            "Lattice": ["Simple cubic (SC)", "Body-centred (BCC)", "Face-centred (FCC)", "Diamond cubic"],
            "Allowed N (first rings)": [
                "1, 2, 3, 4, 5, 6, 8, 9, 10",
                "2, 4, 6, 8, 10, 12, 14, 16",
                "3, 4, 8, 11, 12, 16, 19, 20",
                "3, 8, 11, 16, 19, 24, 27",
            ],
        })
        st.markdown("Once rings are indexed, each gives an estimate of the lattice parameter:")
        st.latex(r"a = d_{hkl}\sqrt{h^2+k^2+l^2}")
        st.caption(
            "SC and BCC give identical ratio sequences (1 : 2 : 3 : 4 : 5 : 6) for their first six rings. "
            "The 7th ring separates them: (R/R₁)² = 8 for SC, 7 for BCC."
        )

    # ------------------------------------------------------------------ 8
    with st.expander("8. d-spacing for other crystal systems  ✅"):
        st.markdown("Used to calculate the theoretical rings of a CIF or library phase:")
        st.markdown("**Tetragonal** (a, c):")
        st.latex(r"\frac{1}{d^2} = \frac{h^2+k^2}{a^2} + \frac{l^2}{c^2}")
        st.markdown("**Hexagonal / rhombohedral in hexagonal axes** (a, c):")
        st.latex(r"\frac{1}{d^2} = \frac{4}{3}\,\frac{h^2+hk+k^2}{a^2} + \frac{l^2}{c^2}")
        st.markdown("**Orthorhombic** (a, b, c):")
        st.latex(r"\frac{1}{d^2} = \frac{h^2}{a^2} + \frac{k^2}{b^2} + \frac{l^2}{c^2}")
        st.markdown("Monoclinic and triclinic phases use the general metric-tensor form; the app evaluates it numerically.")

    # ------------------------------------------------------------------ 9
    with st.expander("9. Systematic absences (which rings can exist)  ✅"):
        st.markdown(
            "Lattice centring and glide planes cancel some reflections. A reflection that is "
            "forbidden for a phase should not be used to support it."
        )
        st.table({
            "Structure type": ["BCC", "FCC", "Diamond cubic", "Spinel (Fd-3m)", "Rhombohedral (R-3c, hexagonal axes)"],
            "Condition for an allowed reflection": [
                "h + k + l even",
                "h, k, l all odd or all even",
                "FCC rule, and h + k + l ≠ 4n + 2 when all even",
                "h, k, l unmixed; and for 0kl: k + l = 4n  ((222) is allowed, (200) is not)",
                "−h + k + l = 3n (plus glide-plane conditions such as 00l: l = 6n)",
            ],
        })
        st.caption(
            "Forbidden rings can still appear weakly in practice (double diffraction, defects, a superlattice), "
            "so the app flags them as warnings instead of removing them."
        )

    # ------------------------------------------------------------------ 10
    with st.expander("10. Matching rings to phases  ✅"):
        st.markdown("The error between a measured ring and the nearest theoretical reflection:")
        st.latex(r"\text{Error}\,(\%) = \frac{d_{\text{exp}} - d_{\text{theo}}}{d_{\text{theo}}}\times 100")
        st.markdown("A ring counts as *matched* if |Error| is within the chosen tolerance (default about 3 %).")
        st.markdown("**Confidence score (0 to 100)** used in the phase leaderboard:")
        st.latex(
            r"S = 100\left[\,0.50\,\frac{n_{\text{matched}}}{n_{\text{rings}}} "
            r"+ 0.25\left(1-\frac{\overline{\text{err}}}{5\,\%}\right) + 0.25\,f_{\text{strong}}\right]"
        )
        st.markdown(
            "n_matched / n_rings is the fraction of measured rings explained, the mean error term rewards close agreement, "
            "and f_strong is the fraction of the phase's three strongest lines that were found. "
            "It is a ranking aid, not a probability."
        )

    # ------------------------------------------------------------------ 11
    with st.expander("11. Mixed phases: unique, shared and unexplained rings  ✅"):
        st.markdown(
            "When several phases are present, each ring is compared with every candidate:  \n"
            "- **Unique**: only one phase group has a reflection within the tolerance, so the ring is evidence for it.  \n"
            "- **Shared**: two or more groups match, so the ring cannot choose between them.  \n"
            "- **Unexplained**: no candidate matches. This suggests a missing phase, a superlattice, or a calibration error."
        )
        st.markdown(
            "**Indistinguishable groups.** Two phases are merged into one group when at least 80 % of the "
            "reflections of each lie within 1 % of a reflection of the other (e.g. magnetite, maghemite and chromite, "
            "which differ by only about 0.5 % in lattice parameter)."
        )
        st.markdown(
            "A group is called **supported** only with at least two unique rings and at most one missing strong line. "
            "Ring brightness is *not* used to estimate phase fractions: it depends on grain orientation, thickness "
            "and multiple scattering."
        )

    # ------------------------------------------------------------------ 12
    with st.expander("12. Crystallite size and strain from ring widths  📘 (not computed)"):
        st.markdown(
            "Broad rings can mean small crystallites, microstrain, or both. In reciprocal-space units "
            "(q = 1/d) the Scherrer relation is"
        )
        st.latex(r"D \approx \frac{K}{\Delta q}\qquad(K\approx 0.9)")
        st.markdown(
            "where Δq is the ring FWHM in nm⁻¹ after subtracting the instrument width (use a gold or aluminium "
            "standard recorded at the same settings). With strain ε the Williamson–Hall form separates the two:"
        )
        st.latex(r"\Delta q = \frac{K}{D} + 2\,\varepsilon\, q")
        st.markdown(
            "Example: Δq = 0.5 nm⁻¹ gives D ≈ 0.9 / 0.5 = 1.8 nm.  \n"
            "Use these only as rough estimates in SAED. Multiple scattering, thickness, beam convergence "
            "and overlapping rings all broaden peaks too."
        )

    # ------------------------------------------------------------------ 13
    with st.expander("13. Limits of SAED analysis  📘"):
        st.markdown(
            "- Accuracy is typically **1 to 3 %** in d, set mostly by calibration and centre errors.  \n"
            "- Ring patterns have few rings, so **several phases can fit**. Combine with EDS, XRD or synthesis information.  \n"
            "- Phases with similar structures cannot be separated (bcc Fe versus Fe-Cr martensite; magnetite versus maghemite).  \n"
            "- Intensities are affected by texture and multiple scattering, so they are not quantitative.  \n"
            "- Small lattice shifts (strain, vacancies) are only meaningful if they exceed the combined uncertainty in section 6."
        )
