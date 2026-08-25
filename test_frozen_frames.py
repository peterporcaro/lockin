"""
Checks detect_frozen_frames() -- the reported symptom (registration offset
pinned at exactly zero for a long stretch) points at duplicated/frozen
frames, a failure mode reject_outlier_frames() is explicitly blind to
(it only catches a frame that jumps far from its neighbours, not a run
that stays suspiciously close).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from lockin_thermography import detect_frozen_frames, diagnose_pixel, reject_outlier_frames

rng = np.random.default_rng(6)
h, w, fps, f = 60, 90, 5.0, 0.1
n = int(300 * fps)
t = np.arange(n) / fps

part = np.zeros((h, w))
part[10:50, 15:75] = 1.0
bulk = 0.5 * part
frames = (part[None] * 0.6 + bulk[None] * np.sin(2 * np.pi * f * t)[:, None, None]
          + rng.normal(0, 0.02, (n, h, w))).astype(np.float32)

ok = []

# --- 1. Clean data: no false positives --------------------------------------
runs = detect_frozen_frames(frames.copy(), t)
ok.append(len(runs) == 0)
print(f"[1] clean data -> {len(runs)} frozen run(s) (expected 0): "
      f"{'PASS' if ok[-1] else 'FAIL'}\n")

# --- 2. A genuinely frozen run is detected -----------------------------------
frozen_start, frozen_len = 100, 40    # 8 seconds of duplicated frames
frames_frozen = frames.copy()
for i in range(frozen_start, frozen_start + frozen_len):
    frames_frozen[i] = frames_frozen[frozen_start]     # exact duplicate, like a stuck grabber
runs = detect_frozen_frames(frames_frozen, t, min_run=3)
print(f"[2] injected frozen run [{frozen_start}, {frozen_start + frozen_len - 1}] "
      f"-> detected runs: {runs}")
ok.append(any(a <= frozen_start + 5 and b >= frozen_start + frozen_len - 5
              for a, b in runs))
print(f"    {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 3. reject_outlier_frames() does NOT catch this (different failure mode) -
_, t_after = reject_outlier_frames(frames_frozen.copy(), t.copy())
print(f"[3] reject_outlier_frames() on the same data dropped "
      f"{n - len(t_after)} frame(s) (expected ~0 -- a frozen run has no "
      f"sharp jump for it to catch)")
ok.append(n - len(t_after) <= 1)
print(f"    {'PASS' if ok[-1] else 'FAIL'}\n")

# --- 4. diagnose_pixel() runs and produces a plot ---------------------------
import os
path = "_test_pixel_diag.png"
diagnose_pixel(frames, t, 30, 45, f, path=path)
ok.append(os.path.exists(path) and os.path.getsize(path) > 0)
print(f"[4] diagnose_pixel() saved a non-empty file: {'PASS' if ok[-1] else 'FAIL'}")
if os.path.exists(path):
    os.remove(path)

print(f"\n{sum(ok)}/{len(ok)} passed")
