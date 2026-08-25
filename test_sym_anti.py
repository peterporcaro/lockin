"""
Checks symmetric_antisymmetric_profile() against a synthetic field with a
KNOWN step (the genuine zone-to-zone power-density difference) and a KNOWN
symmetric bump (the candidate leakage signal) straddling it -- the exact
scenario CHANGE 1 is built around: the step must decompose into `anti` and
NOT leak into `sym`, and the bump must decompose into `sym` and NOT leak
into `anti`, even when the given line position is deliberately off by a
few pixels (recentre=True should correct that).

Uses a genuinely TILTED line (not axis-aligned) with a matching rotated
field, since an exactly axis-aligned line is precisely the degenerate case
check_line_angle() warns about -- see test_slanted_edge.py for that check
and for a direct demonstration of the sub-pixel resolution gain itself.
"""
import numpy as np
from scipy.special import erf
from lockin_thermography import symmetric_antisymmetric_profile, analyse_deletion_line

h, w = 120, 300
theta = np.radians(20.0)          # well clear of 0/45/90
unit = np.array([np.cos(theta), np.sin(theta)])
normal = np.array([-np.sin(theta), np.cos(theta)])
centre_true = np.array([150.0, 60.0])       # true line passes through here
half_len = 40.0
p0_true = tuple(centre_true - half_len * unit)
p1_true = tuple(centre_true + half_len * unit)

L, R = 1.0, 1.6          # the two zones' genuine power-density levels
step_mag = R - L
bump_amp, bump_sigma = 0.15, 6.0
step_sigma = 2.0         # the physical step's own diffusion smoothing

yy, xx = np.mgrid[0:h, 0:w]
n = (xx - centre_true[0]) * normal[0] + (yy - centre_true[1]) * normal[1]
step = L + (R - L) * 0.5 * (1 + erf(n / (np.sqrt(2) * step_sigma)))
bump = bump_amp * np.exp(-(n ** 2) / (2 * bump_sigma ** 2))
img_with_bump = step + bump
img_step_only = step

mu_px = 6.0
mm_per_px = 1.0
half_width_mm = 30.0
ok = []


def offset_line(p0, p1, off_px):
    """Shift a line by off_px along its own normal -- a stand-in for a few
    pixels of click/detection error."""
    p0a, p1a = np.asarray(p0, float), np.asarray(p1, float)
    u = (p1a - p0a) / np.hypot(*(p1a - p0a))
    n_ = np.array([-u[1], u[0]])
    return tuple(p0a + off_px * n_), tuple(p1a + off_px * n_)


# --- 1. Pure step (no leakage): sym should stay near zero everywhere -------
res = symmetric_antisymmetric_profile(img_step_only, p0_true, p1_true,
                                      mu_px, mm_per_px, half_width_mm=half_width_mm,
                                      recentre=True)
print(f"[1] pure step -> sym peak {res['peak']:.4g} (expect ~0), "
      f"anti step {2 * res['anti_step']:+.3f} (expect {-step_mag:+.3f})")
ok.append(abs(res["peak"]) < 0.01)
ok.append(abs(abs(2 * res["anti_step"]) - step_mag) < 0.05)
print(f"    sym~0: {'PASS' if ok[-2] else 'FAIL'}, "
      f"anti matches step: {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 2. Step + bump, line given exactly at the true centre -----------------
res = symmetric_antisymmetric_profile(img_with_bump, p0_true, p1_true,
                                      mu_px, mm_per_px, half_width_mm=half_width_mm,
                                      recentre=True)
print(f"[2] step+bump, exact centre -> sym peak {res['peak']:.4g} "
      f"(expect ~{bump_amp:.3f}), anti step {2 * res['anti_step']:+.3f} "
      f"(expect {-step_mag:+.3f})")
ok.append(abs(res["peak"] - bump_amp) < 0.03)
ok.append(abs(abs(2 * res["anti_step"]) - step_mag) < 0.05)
print(f"    sym matches bump: {'PASS' if ok[-2] else 'FAIL'}, "
      f"anti matches step: {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 3. Same, but the given line is off by 3 px -- recentre should fix it --
p0_off, p1_off = offset_line(p0_true, p1_true, 3.0)
res = symmetric_antisymmetric_profile(img_with_bump, p0_off, p1_off,
                                      mu_px, mm_per_px, half_width_mm=half_width_mm,
                                      recentre=True)
print(f"[3] step+bump, line off by 3px -> centre_shift {res['centre_shift_px']:+.2f} "
      f"(expect ~-3), sym peak {res['peak']:.4g}")
ok.append(abs(abs(res["centre_shift_px"]) - 3.0) < 1.5)
ok.append(abs(res["peak"] - bump_amp) < 0.03)
print(f"    recentring found the true line: {'PASS' if ok[-2] else 'FAIL'}, "
      f"sym still matches bump after correction: {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 4. Without recentring, accuracy against the KNOWN bump amplitude ------
# Mis-centring leaks step into sym, but not necessarily by inflating the
# peak in a fixed direction -- it can shift or distort the profile instead.
# The defensible, direction-agnostic claim is that recentring gets closer
# to the known ground truth than not recentring does.
res_recentred = symmetric_antisymmetric_profile(img_with_bump, p0_off, p1_off,
                                                 mu_px, mm_per_px,
                                                 half_width_mm=half_width_mm, recentre=True)
res_no_recentre = symmetric_antisymmetric_profile(img_with_bump, p0_off, p1_off,
                                                   mu_px, mm_per_px,
                                                   half_width_mm=half_width_mm, recentre=False)
err_recentred = abs(res_recentred["peak"] - bump_amp)
err_no_recentre = abs(res_no_recentre["peak"] - bump_amp)
print(f"[4] step+bump, line off by 3px -> sym peak error: recentred "
      f"{err_recentred:.4g}, not recentred {err_no_recentre:.4g} "
      f"(recentring should be more accurate)")
ok.append(err_recentred < err_no_recentre)
print(f"    {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 5. analyse_deletion_line() shares one recentred line across channels --
phase_img = step * 0.01    # a smaller "phase" step, no bump
combo = analyse_deletion_line(img_with_bump, phase_img, p0_off, p1_off,
                              mu_mm=mu_px, mm_per_px=mm_per_px,
                              half_width_mm=half_width_mm, ply_thickness_mm=0.0)
same_line = (combo["amp"]["p0"] == combo["phase"]["p0"] and
            combo["amp"]["p1"] == combo["phase"]["p1"])
print(f"[5] analyse_deletion_line: amp/phase share the same recentred line, "
      f"angle {combo['angle_deg']:.1f} deg (expect ~{np.degrees(theta):.1f}): {same_line}")
ok.append(same_line)
ok.append(combo["phase"]["centre_shift_px"] == 0.0)   # phase call used recentre=False
ok.append(abs(combo["angle_deg"] - np.degrees(theta)) < 0.5)
print(f"    {'PASS' if ok[-3] and ok[-2] and ok[-1] else 'FAIL'}\n")

print(f"{sum(ok)}/{len(ok)} passed")
