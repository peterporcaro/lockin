"""
Smoke test for auto_geometry=True: analyse() should detect both deletion
lines and place the fiducial ROI itself, headlessly, with no roi_config and
no clicking -- and lockin_line_candidates.png should be produced showing
every candidate found.

Reuses the same synthetic two-zone-step-plus-one-leak scene as
test_pipeline_smoke.py, but skips building a roi_config by hand.
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
from lockin_thermography import analyse

rng = np.random.default_rng(1)
h, w, fps, f = 200, 260, 5.0, 0.1
n = int(400 * fps)
t = np.arange(n) / fps

part = np.zeros((h, w))
part[40:160, 50:210] = 1.0

bulk = np.zeros((h, w))
bulk[40:160, 50:90] = 0.4
bulk[40:160, 90:170] = 0.7
bulk[40:160, 170:210] = 0.55

xx = np.arange(w)
leak = 0.12 * np.exp(-((xx - 170) ** 2) / (2 * 1.5 ** 2))
leak_field = np.zeros((h, w))
leak_field[40:160, :] = leak[None, :]

src = bulk + leak_field

frames = (part[None] * 0.6
          + src[None] * np.sin(2 * np.pi * f * t)[:, None, None]
          + 0.05 * t[:, None, None]
          + rng.normal(0, 0.03, (n, h, w))).astype(np.float32)
np.save("_auto_smoke.npy", frames)

amp, phase, nf, results = analyse(
    "_auto_smoke.npy", fps=fps, f_excite=f, mm_per_px=1.0, n_lines=2,
    use_saved_config=False, auto_geometry=True,
    roi_config_path="_auto_roi_config.json",
    register=False, reject_outliers=True, use_lockin_cache=False,
)

assert len(results) == 2, f"expected 2 auto-detected lines, got {len(results)}"
xs = sorted(r["p0"][0] if "p0" in r else None for r in results) if False else None
print(f"\nauto_geometry found {len(results)} line(s) -- OK")

for path in ("lockin_images.png", "lockin_line_profiles.png",
             "lockin_line_candidates.png", "_auto_roi_config.json"):
    assert os.path.exists(path), f"expected {path} to be created"
print("all expected output files (including lockin_line_candidates.png) present -- OK")

for f_ in ("_auto_smoke.npy", "_auto_roi_config.json"):
    if os.path.exists(f_):
        os.remove(f_)

print("\nAUTO GEOMETRY SMOKE TEST OK")
