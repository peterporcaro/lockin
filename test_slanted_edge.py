"""
Demonstrates the actual deliverable of CHANGE 1: slanted_edge_profile(),
pooling a genuinely tilted line's crossings via continuous bilinear
interpolation, recovers a transition narrower than one native pixel far
more accurately than the old nearest-neighbour cross_line_profile() does.
Also checks check_line_angle()'s near-critical-angle warnings.
"""
import numpy as np
from scipy.special import erf, erfinv
from lockin_thermography import slanted_edge_profile, cross_line_profile, check_line_angle

ok = []

# --- 1. Sub-pixel gain on a genuinely sub-pixel-narrow transition ----------
h, w = 150, 300
true_sigma = 0.3          # narrower than 1 native pixel
theta = np.radians(15.0)  # well clear of 0/45/90
unit = np.array([np.cos(theta), np.sin(theta)])
normal = np.array([-np.sin(theta), np.cos(theta)])
centre = np.array([150.0, 75.0])
half_len = 60.0
p0 = tuple(centre - half_len * unit)
p1 = tuple(centre + half_len * unit)

yy, xx = np.mgrid[0:h, 0:w]
n = (xx - centre[0]) * normal[0] + (yy - centre[1]) * normal[1]
img = 0.5 * (1 + erf(n / (np.sqrt(2) * true_sigma)))   # step from 0 to 1

off_fine, prof_fine = slanted_edge_profile(img, p0, p1, half_width_px=5.0,
                                           bin_px=0.05, n_along=1500)
off_nn, prof_nn = cross_line_profile(img, p0, p1, half_width_px=5, n_samples=1500)


def rise_1090(off, prof):
    """10-90% rise width via linearly-interpolated level crossings."""
    lo, hi = prof.min(), prof.max()
    t10, t90 = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    order = np.argsort(off)
    o, p = off[order], prof[order]

    def cross(level):
        idx = np.nonzero(np.diff(np.sign(p - level)))[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        frac = (level - p[i]) / (p[i + 1] - p[i])
        return o[i] + frac * (o[i + 1] - o[i])

    x10, x90 = cross(t10), cross(t90)
    return None if x10 is None or x90 is None else abs(x90 - x10)


true_rise = true_sigma * np.sqrt(2) * 2 * erfinv(0.8)
rise_fine = rise_1090(off_fine, prof_fine)
rise_nn = rise_1090(off_nn, prof_nn)
err_fine = abs(rise_fine - true_rise)
err_nn = abs(rise_nn - true_rise)
print(f"[1] true 10-90% rise = {true_rise:.3f} px")
print(f"    slanted_edge_profile (bin_px=0.05): recovered {rise_fine:.3f} px, "
      f"error {err_fine:.3f}")
print(f"    cross_line_profile (nearest-neighbour): recovered {rise_nn:.3f} px, "
      f"error {err_nn:.3f}")
ok.append(err_fine < err_nn)
# Bilinear interpolation between native pixels has its own accuracy limit
# at a scale this far sub-pixel (0.3 px true sigma) -- it can't perfectly
# reconstruct structure finer than the native sampling, only mitigate the
# quantisation/staircasing naive nearest-neighbour sampling adds on top.
# The honest claim here is "meaningfully closer to true", not "exact".
ok.append(err_fine < 0.7 * err_nn)
print(f"    slanted-edge is more accurate: {'PASS' if ok[-2] else 'FAIL'}, "
      f"meaningfully so (>=30% closer to truth): {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 2. check_line_angle(): warns near 0/45/90, quiet for a good tilt ------
cases = [
    (0.0, True), (1.5, True), (89.0, True), (90.0, True),
    (44.0, True), (45.0, True), (46.5, True),
    (15.0, False), (20.0, False), (30.0, False), (60.0, False), (70.0, False),
]
for angle, expect_warn in cases:
    rad = np.radians(angle)
    p0c, p1c = (0.0, 0.0), (100 * np.cos(rad), 100 * np.sin(rad))
    _, warned = check_line_angle(p0c, p1c, bin_px=0.2)
    good = warned == expect_warn
    ok.append(good)
    print(f"[2] angle={angle:5.1f} deg -> warned={warned} "
          f"(expected {expect_warn}): {'PASS' if good else 'FAIL'}")

print(f"\n{sum(ok)}/{len(ok)} passed")
