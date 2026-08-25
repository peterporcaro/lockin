"""Checks that auto_fiducial_roi() lands in the quiet zone between/away from lines."""
import numpy as np
from lockin_thermography import auto_fiducial_roi, find_deletion_lines


def canvas(h=300, w=400):
    return np.zeros((h, w)), np.ones((h, w), dtype=bool)


def draw(img, p0, p1, amp, width=1.5, n=2000):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    for s in np.linspace(0, 1, n):
        x, y = p0 + s * (p1 - p0)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                yy, xx = int(round(y)) + dy, int(round(x)) + dx
                if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                    r2 = (yy - y) ** 2 + (xx - x) ** 2
                    img[yy, xx] = max(img[yy, xx],
                                      amp * np.exp(-r2 / (2 * width ** 2)))


ok = []

# --- 1. Two vertical lines -- ROI should land in the gap between them ------
img, part = canvas()
draw(img, (100, 20), (100, 280), 1.0)
draw(img, (300, 20), (300, 280), 0.8)
img += np.random.default_rng(0).normal(0, 0.02, img.shape)
lines = find_deletion_lines(img, part)
y0, y1, x0, x1 = auto_fiducial_roi(part, lines, exclusion_px=15.0)
cx = (x0 + x1) / 2
print(f"[1] two lines at x=100,300 -> ROI x-center {cx:.0f} "
      f"(y[{y0}:{y1}] x[{x0}:{x1}])")
ok.append(100 + 15 < cx < 300 - 15)
print(f"    {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 2. One line -- ROI should clear it by at least exclusion_px -----------
img, part = canvas()
draw(img, (150, 20), (150, 280), 1.0)
img += np.random.default_rng(1).normal(0, 0.02, img.shape)
lines = find_deletion_lines(img, part)
exclusion = 15.0
y0, y1, x0, x1 = auto_fiducial_roi(part, lines, exclusion_px=exclusion)
dist = min(abs(x0 - 150), abs(x1 - 150))
print(f"[2] one line at x=150 -> ROI x[{x0}:{x1}]  min clearance {dist:.0f} px")
ok.append(dist >= exclusion - 1)   # -1 for rounding
print(f"    {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 3. No lines -- falls back to the part's most interior point -----------
part = np.zeros((200, 300), dtype=bool)
part[30:170, 40:260] = True
y0, y1, x0, x1 = auto_fiducial_roi(part, [])
cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
print(f"[3] no lines -> ROI center ({cx:.0f}, {cy:.0f}), part center (150, 100)")
ok.append(abs(cx - 150) < 20 and abs(cy - 100) < 20)
print(f"    {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 4. Empty part mask raises --------------------------------------------
try:
    auto_fiducial_roi(np.zeros((50, 50), dtype=bool), [])
    ok.append(False)
    print("[4] empty part -> FAIL (no exception raised)")
except ValueError:
    ok.append(True)
    print("[4] empty part -> PASS (raised ValueError)")

# --- 5. Box scales with exclusion_px (diffusion length), not a flat cap ----
# On a big, wide-open gap the old flat max_half_px=40 always saturated the
# box to 80x80 regardless of scale.  It should now track ~2x exclusion_px.
img, part = canvas(h=600, w=800)
draw(img, (150, 20), (150, 580), 1.0)
draw(img, (650, 20), (650, 580), 1.0)
img += np.random.default_rng(2).normal(0, 0.02, img.shape)
lines = find_deletion_lines(img, part)
for exclusion in (10.0, 30.0):
    y0, y1, x0, x1 = auto_fiducial_roi(part, lines, exclusion_px=exclusion)
    half = (x1 - x0) / 2
    expected = 2 * exclusion
    print(f"[5] exclusion_px={exclusion:.0f} -> box half-width {half:.0f} px "
          f"(expected ~{expected:.0f})")
    ok.append(abs(half - expected) <= 1)
    print(f"    {'PASS' if ok[-1] else 'FAIL'}")
print()

# --- 6. Tiny exclusion_px (e.g. a fine mm_per_px shrinks the diffusion
# length to ~1px) must NOT collapse edge clearance to ~1px too -- this is
# the actual reported bug: with sigma_px this small, the old
# edge_clear_px=exclusion_px default gave almost no real edge margin, and
# the ROI landed right against a busbar/scalloped edge.
part = np.zeros((300, 400), dtype=bool)
part[10:290, 10:390] = True                    # true edge 10 px from array edge
img = np.zeros((300, 400))
draw(img, (200, 20), (200, 280), 1.0)           # single line, well clear of edges
img += np.random.default_rng(3).normal(0, 0.02, img.shape)
lines = find_deletion_lines(img, part)
y0, y1, x0, x1 = auto_fiducial_roi(part, lines, exclusion_px=1.0)
edge_clearance = min(y0 - 0, 300 - y1, x0 - 0, 400 - x1)
print(f"[6] tiny exclusion_px=1.0 -> ROI y[{y0}:{y1}] x[{x0}:{x1}], "
      f"clearance from true part edge {edge_clearance} px")
ok.append(edge_clearance >= 15)    # should be ~8% of 300 = 24, not ~1
print(f"    {'PASS' if ok[-1] else 'FAIL'}\n")

print(f"\n{sum(ok)}/{len(ok)} passed")
