"""
Reproduces the reported "pasted patch" artefact: a single frame with a
localized corrupted region (a NUC-event stand-in), invisible in the smooth
raw amplitude but showing up sharply in the lock-in amplitude at that spot
if left in.  reject_outlier_frames() should detect and drop exactly that
frame, and doing so should collapse the artefact.
"""
import numpy as np
from lockin_thermography import reject_outlier_frames, lockin, spatial_highpass

rng = np.random.default_rng(5)
h, w, fps, f = 80, 120, 5.0, 0.1
n = int(300 * fps)
t = np.arange(n) / fps

part = np.zeros((h, w))
part[10:70, 15:105] = 1.0
bulk = 0.5 * part          # coherent bulk heating, same reasoning as other tests

frames = (part[None] * 0.6 + bulk[None] * np.sin(2 * np.pi * f * t)[:, None, None]
          + rng.normal(0, 0.02, (n, h, w))).astype(np.float32)

# Corrupt ONE frame with a localized patch of garbage content -- a stand-in
# for a NUC event that doesn't just shift the frame's overall level (which
# remove_global_offsets would fix) but replaces a chunk of pixels outright.
bad_idx = 137
patch = (slice(30, 45), slice(50, 65))
frames_corrupted = frames.copy()
frames_corrupted[bad_idx][patch] = 8.0     # way outside the normal ~0.6-1.1 range

ok = []

# --- 1. Clean data: no false positives -------------------------------------
_, t_clean = reject_outlier_frames(frames.copy(), t.copy())
ok.append(len(t_clean) == n)
print(f"[1] clean data -> {n - len(t_clean)} frames dropped "
      f"(expected 0): {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 2. Corrupted data: detects exactly the bad frame -----------------------
frames_clean, t_after = reject_outlier_frames(frames_corrupted.copy(), t.copy())
ok.append(len(t_after) == n - 1 and bad_idx not in
          {i for i, tv in enumerate(t) if tv in t_after})
dropped_time_present = t[bad_idx] not in t_after
ok[-1] = len(t_after) == n - 1 and dropped_time_present
print(f"[2] corrupted data -> dropped {n - len(t_after)} frame(s), "
      f"bad frame's timestamp removed: {dropped_time_present}: "
      f"{'PASS' if ok[-1] else 'FAIL'}\n")

# --- 3. The artefact collapses once the frame is dropped --------------------
# The reported symptom is specifically in the SPATIAL HIGH-PASS panel: the
# bulk heating signal is spatially smooth and gets removed by
# spatial_highpass, which is exactly what a single bad frame's small but
# sharply-localized contribution survives -- on raw (non-highpassed)
# amplitude, that same contribution is negligible next to the coherent bulk
# signal every frame contributes, so it wouldn't show up there at all.
amp_with, _ = lockin(frames_corrupted, t, f)
amp_without, _ = lockin(frames_clean, t_after, f)
hp_with = spatial_highpass(amp_with, sigma_px=5.0)
hp_without = spatial_highpass(amp_without, sigma_px=5.0)
patch_hp_with = np.abs(hp_with[patch]).mean()
patch_hp_without = np.abs(hp_without[patch]).mean()
print(f"[3] high-pass amplitude at corrupted patch: with={patch_hp_with:.4f}, "
      f"without={patch_hp_without:.4f}")
ok.append(patch_hp_without < 0.3 * patch_hp_with)
print(f"    artefact collapses by >70%: {'PASS' if ok[-1] else 'FAIL'}\n")

print(f"{sum(ok)}/{len(ok)} passed")
