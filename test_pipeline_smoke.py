"""
End-to-end smoke test of analyse() on a synthetic sequence with TWO
deletion lines, each splitting genuinely different power-density zones
(the real, non-defect step this whole feature is built around), with a
small symmetric leakage bump added at only ONE of the two lines -- the
sym/anti analysis should clearly separate the two: high ratio at the
line with a real defect, low ratio at the clean one.

Runs fully headless via a hand-built roi_config (no interactive clicking).
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
from lockin_thermography import analyse, part_mask, save_roi_config

rng = np.random.default_rng(1)
# Part occupies a modest central fraction of the frame -- otherwise
# part_mask's inward erosion (default 10 px) eats a chunk of the true part
# edge into what looks like "background", swamping the small strip of real
# background and making the two indistinguishable for the check below.
h, w, fps, f = 200, 260, 5.0, 0.1
n = int(400 * fps)
t = np.arange(n) / fps

part = np.zeros((h, w))
part[40:160, 50:210] = 1.0

# Three zones, two deletion lines (x=90 and x=170), each line carrying a
# genuine step in power density -- distinct heater geometry per zone, same
# physical picture the report described.
bulk = np.zeros((h, w))
bulk[40:160, 50:90] = 0.4
bulk[40:160, 90:170] = 0.7
bulk[40:160, 170:210] = 0.55

# A small symmetric leakage bump straddling ONLY the line at x=170 -- the
# signal the sym/anti decomposition should recover there, and correctly
# NOT report at the clean line (x=90).  Width chosen comparable to the
# diffusion length at f_excite=0.1 Hz (~1.26 mm) so it reads as a plausible
# candidate rather than tripping the "too wide" zone-scale verdict.
xx = np.arange(w)
leak = 0.12 * np.exp(-((xx - 170) ** 2) / (2 * 1.5 ** 2))
leak_field = np.zeros((h, w))
leak_field[40:160, :] = leak[None, :]

src = bulk + leak_field

frames = (part[None] * 0.6
          + src[None] * np.sin(2 * np.pi * f * t)[:, None, None]
          + 0.05 * t[:, None, None]                       # ambient ramp
          + rng.normal(0, 0.03, (n, h, w))).astype(np.float32)
np.save("_smoke.npy", frames)

roi_config = {
    "lines": [
        {"p0": [90.0, 45.0], "p1": [90.0, 155.0], "angle_deg": 90.0,
         "is_reference": True},           # clean line, tagged as reference
        {"p0": [170.0, 45.0], "p1": [170.0, 155.0], "angle_deg": 90.0,
         "is_reference": False},          # the line with the leak bump
    ],
    "fiducial_roi": [90, 110, 120, 140],   # clean interior of the middle zone
    "mm_per_px": 1.0,
}
save_roi_config(roi_config, "_roi_config_smoke.json")

amp, phase, nf, results = analyse(
    "_smoke.npy", fps=fps, f_excite=f,
    roi_config_path="_roi_config_smoke.json", use_saved_config=True,
    register=False, reject_outliers=True,
)
assert not np.isnan(phase).any(), "phase image is all/partially NaN"
assert len(results) == 2, f"expected 2 line results, got {len(results)}"
print(f"\nreturned amp {amp.shape}, phase {phase.shape}, noise floor {nf:.4g}, "
      f"{len(results)} line result(s) -- OK")

clean_line, defect_line = results[0]["amp"], results[1]["amp"]
print(f"line 0 (clean):  peak={clean_line['peak']:.4g}  "
      f"ratio={clean_line['ratio']:.2f}  fwhm={results[0]['fwhm_mm']:.2f}mm  "
      f"verdict={results[0]['verdict']}")
print(f"line 1 (defect): peak={defect_line['peak']:.4g}  "
      f"ratio={defect_line['ratio']:.2f}  fwhm={results[1]['fwhm_mm']:.2f}mm  "
      f"verdict={results[1]['verdict']}")
# The peak itself is the clean signal here (register=False avoids the
# already-documented spurious-registration-spike this synthetic sequence
# can trigger, which would otherwise confound this check independent of
# the sym/anti math being tested) -- registration correctness has its own
# dedicated tests (test_registration.py, test_registration_drift.py).
assert defect_line["peak"] > 100 * clean_line["peak"], (
    "expected the defect line's sym peak to be far above the clean line's"
)
assert defect_line["ratio"] > clean_line["ratio"], (
    "expected the defect line's ratio to exceed the clean line's"
)
assert "plausible candidate" in results[1]["verdict"], (
    f"expected the defect line to read as a plausible candidate, "
    f"got: {results[1]['verdict']!r}"
)

for path in ("lockin_images.png", "lockin_line_profiles.png"):
    assert os.path.exists(path), f"expected {path} to be created"
print("all expected output files present -- OK")

# Empty fiducial_roi is still rejected with a clear error.
bad_config = dict(roi_config, fiducial_roi=[110, 90, 140, 120])   # inverted
save_roi_config(bad_config, "_roi_config_bad.json")
try:
    analyse("_smoke.npy", fps=fps, f_excite=f,
            roi_config_path="_roi_config_bad.json", use_saved_config=True)
    raise AssertionError("expected ValueError for an empty fiducial_roi")
except ValueError as e:
    print(f"empty fiducial_roi correctly rejected: {e}")

# Background pixels carry no coherent lock-in signal, so their phase is
# essentially random -- confirm the part's own phase spread (what the
# display should scale to) is much tighter than the full-frame spread
# (what it would scale to if the background weren't masked out).
part_est = part_mask(amp)
deg = np.degrees(phase)
full_span = np.percentile(deg, 98) - np.percentile(deg, 2)
part_span = np.percentile(deg[part_est], 98) - np.percentile(deg[part_est], 2)
print(f"phase spread (2-98pct): part {part_span:.0f} deg, full frame {full_span:.0f} deg")
assert part_span < 0.5 * full_span, (
    "masking isn't helping -- part phase spread isn't much tighter than "
    "the unmasked full-frame spread"
)

for f_ in ("_smoke.npy", "_roi_config_smoke.json", "_roi_config_bad.json"):
    if os.path.exists(f_):
        os.remove(f_)

print("\nSMOKE TEST OK")
