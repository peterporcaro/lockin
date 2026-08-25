"""Synthetic checks that find_deletion_line() picks exactly one real line."""
import numpy as np
from lockin_thermography import find_deletion_line, find_deletion_lines


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


def check(name, got, want, tol=6.0):
    (gx0, gy0), (gx1, gy1) = got
    (wx0, wy0), (wx1, wy1) = want
    fwd = max(abs(gx0 - wx0), abs(gy0 - wy0), abs(gx1 - wx1), abs(gy1 - wy1))
    rev = max(abs(gx0 - wx1), abs(gy0 - wy1), abs(gx1 - wx0), abs(gy1 - wy0))
    err = min(fwd, rev)
    print(f"{name}: {'PASS' if err <= tol else 'FAIL'}  (max endpoint error "
          f"{err:.1f} px)\n")
    return err <= tol


rng = np.random.default_rng(0)
ok = []

# --- 1. Three parallel lines, the middle one brightest -----------------------
img, part = canvas()
draw(img, (60, 20), (60, 280), 0.4)
draw(img, (200, 20), (200, 280), 1.0)          # the one we want
draw(img, (330, 20), (330, 280), 0.6)
img += rng.normal(0, 0.02, img.shape)
print("[1] three parallel lines, middle brightest")
ok.append(check("  ", find_deletion_line(img, part), ((200, 20), (200, 280))))

# --- 2. Crossing lines at different angles -----------------------------------
img, part = canvas()
draw(img, (20, 20), (380, 280), 0.5)
draw(img, (20, 280), (380, 20), 1.0)           # the one we want
img += rng.normal(0, 0.02, img.shape)
print("[2] crossing lines, one brighter")
ok.append(check("  ", find_deletion_line(img, part), ((20, 280), (380, 20))))

# --- 3. Bright line broken by a gap, vs a solid dimmer line ------------------
img, part = canvas()
draw(img, (120, 20), (120, 130), 1.0)          # same line, split by a 30 px gap
draw(img, (120, 160), (120, 280), 1.0)
draw(img, (300, 20), (300, 280), 0.45)
img += rng.normal(0, 0.02, img.shape)
print("[3] bright line with a 30 px gap vs dim solid line")
ok.append(check("  ", find_deletion_line(img, part), ((120, 20), (120, 280))))

# --- 4. Near-vertical and near-horizontal together ---------------------------
img, part = canvas()
draw(img, (20, 150), (380, 148), 1.0)          # the one we want
draw(img, (250, 20), (252, 280), 0.7)
img += rng.normal(0, 0.02, img.shape)
print("[4] near-horizontal (bright) vs near-vertical")
ok.append(check("  ", find_deletion_line(img, part), ((20, 150), (380, 148))))

# --- 5. A line plus a bright compact blob (blob must not win) ----------------
img, part = canvas()
draw(img, (150, 20), (150, 280), 0.8)
img[100:118, 300:318] = 1.5                    # hot spot, high amplitude
img += rng.normal(0, 0.02, img.shape)
print("[5] line vs bright compact blob")
ok.append(check("  ", find_deletion_line(img, part), ((150, 20), (150, 280))))

# --- 6. Old failure mode: the naive single-fit answer ------------------------
img, part = canvas()
draw(img, (60, 20), (60, 280), 1.0)
draw(img, (340, 20), (340, 280), 1.0)
img += rng.normal(0, 0.02, img.shape)
lines = find_deletion_lines(img, part)
xs = sorted(round(s["p0"][0]) for s in lines)
print(f"[6] two equal lines -> found x = {xs} "
      f"(a single global fit would land near x=200, on neither line)")
ok.append(len(lines) == 2 and abs(xs[0] - 60) < 6 and abs(xs[1] - 340) < 6)
print(f"    {'PASS' if ok[-1] else 'FAIL'}\n")

print(f"{sum(ok)}/{len(ok)} passed")
