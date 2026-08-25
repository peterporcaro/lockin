"""
Proves the mechanism the report described: a static, high-contrast ridge that
moves coherently with the excitation produces a dipole under lock-in (two
opposite-sign flanks straddling a null at the ridge centre -- the derivative
of the static pattern), not the symmetric bump a real heat source gives, and
that register_frames() removes it.
"""
import numpy as np
from scipy.ndimage import shift as ndshift, gaussian_filter
from lockin_thermography import lockin, register_frames, _coherent_amplitude

rng = np.random.default_rng(0)
h, w, fps, f = 60, 160, 5.0, 0.1
n = int(300 * fps)
t = np.arange(n) / fps

# A static, high-contrast ridge at x=80 -- a stand-in for a deletion line's
# own optical/emissivity edge (or the part boundary), NOT a heat source:
# nothing in this pattern is time-varying except its position.  A bump
# rather than an infinite step so it decays to ~0 well before either image
# edge -- an infinite step wraps discontinuously in the FFT phase
# correlation uses internally and fools it into chasing that seam instead
# of the real feature.
i_edge, sigma = 80, 3.0
x = np.arange(w)
edge = np.exp(-((x - i_edge) ** 2) / (2 * sigma ** 2))

# Real IR frames have broadband texture (surface finish, mild real gradients)
# everywhere, not just at the ridge.  Phase correlation is phase-only -- it
# normalises away magnitude -- so an otherwise near-blank image lets noise
# dominate that normalisation and gives meaningless offsets.  If the whole
# part is rigidly translating (the physical scenario here), that texture
# translates right along with the ridge, so it belongs baked into the same
# static pattern that gets shifted every frame.
texture = gaussian_filter(rng.normal(0, 1, (h, w)), sigma=1.5)
texture *= 0.3 / texture.std()
pattern = np.tile(edge, (h, 1)) + texture

dx_true = 0.35 * np.sin(2 * np.pi * f * t)             # coherent "breathing"
frames = np.empty((n, h, w), dtype=np.float64)
for i in range(n):
    frames[i] = ndshift(pattern, (0, dx_true[i]), order=3, mode="nearest")
frames += rng.normal(0, 0.01, frames.shape)
frames = frames.astype(np.float32)

# --- 1. Before registration: lock-in of pure coherent motion is a dipole ---
# The lock-in of a shifting static pattern is proportional to -dx(t) times
# the pattern's own spatial derivative -- for a bump, that derivative is
# zero at the peak and largest a bit to either side, with opposite sign.
# So the signature is a NULL at the ridge centre flanked by two comparable
# peaks in antiphase, not one peak centred on the ridge.
# Average over rows: the ridge (and its motion) is identical in every row,
# but the random texture differs per row, so averaging cancels the texture's
# own local-gradient contribution while reinforcing the row-invariant
# dipole -- a cleaner read on the mechanism than any single row.
amp_before, phase_before = lockin(frames, t, f)
row = amp_before.mean(axis=0)
flank = int(round(sigma))
at_edge = row[i_edge - 1:i_edge + 2].mean()
left = row[i_edge - 2 * flank:i_edge - flank].max()
right = row[i_edge + flank:i_edge + 2 * flank].max()
signed_before = phase_before[h // 2, i_edge - flank] - phase_before[h // 2, i_edge + flank]
print(f"[1] before registration -- amplitude {at_edge:.4f} at centre (null "
      f"expected), {left:.4f} left flank, {right:.4f} right flank (peaks "
      f"expected), phase flip across centre {np.degrees(signed_before):.0f} deg")
ok1 = (left > 1.4 * at_edge and right > 1.4 * at_edge
       and abs(abs(np.degrees(signed_before)) - 180) < 45)
print(f"    dipole confirmed (null at centre, antiphase flanks): "
      f"{'PASS' if ok1 else 'FAIL'}\n")

# --- 2. register_frames() recovers the injected motion -----------------
frames_reg = frames.copy()
offsets = register_frames(frames_reg, t, f, upsample_factor=20)
dx_amp_true = _coherent_amplitude(dx_true, t, f)
dx_amp_recovered = _coherent_amplitude(offsets[:, 1], t, f)
print(f"[2] injected dx coherent amplitude {dx_amp_true:.3f} px, "
      f"recovered {dx_amp_recovered:.3f} px")
ok2 = abs(dx_amp_recovered - dx_amp_true) < 0.12
print(f"    {'PASS' if ok2 else 'FAIL'}\n")

# --- 3. After registration: the dipole collapses ------------------------
amp_after, _ = lockin(frames_reg, t, f)
row_after = amp_after.mean(axis=0)
left_after = row_after[i_edge - 2 * flank:i_edge - flank].max()
right_after = row_after[i_edge + flank:i_edge + 2 * flank].max()
print(f"[3] after registration -- flank amplitude {left_after:.4f} / "
      f"{right_after:.4f} (was {left:.4f} / {right:.4f})")
ok3 = left_after < 0.3 * left and right_after < 0.3 * right
print(f"    dipole flanks collapsed by >70%: {'PASS' if ok3 else 'FAIL'}\n")

oks = [ok1, ok2, ok3]
print(f"{sum(oks)}/{len(oks)} passed")
