"""Checks the save_roi_config()/load_roi_config() JSON roundtrip used to make
setup repeatable across runs of the same recording without re-clicking."""
import os
from lockin_thermography import save_roi_config, load_roi_config

path = "_test_roi_config.json"
config = {
    "lines": [
        {"p0": [130.0, 45.0], "p1": [130.0, 155.0], "angle_deg": 90.0,
         "is_reference": True},
        {"p0": [210.0, 45.0], "p1": [212.0, 155.0], "angle_deg": 88.9,
         "is_reference": False},
    ],
    "fiducial_roi": [90, 110, 60, 80],
    "mm_per_px": 2.734,
    "processing_params": {
        "mm_per_px": 2.734, "half_width_mm": 10.0, "bin_px": 0.2,
        "n_along": 800, "f_excite": 0.1, "ply_thickness_mm": 3.0,
    },
}

ok = []

save_roi_config(config, path)
ok.append(os.path.exists(path))
print(f"[1] save_roi_config() wrote the file: {'PASS' if ok[-1] else 'FAIL'}")

loaded = load_roi_config(path)
ok.append(loaded == config)
print(f"[2] load_roi_config() roundtrips exactly: {'PASS' if ok[-1] else 'FAIL'}")

ok.append(len(loaded["lines"]) == 2)
print(f"[3] multi-line config preserved ({len(loaded['lines'])} lines): "
      f"{'PASS' if ok[-1] else 'FAIL'}")

ok.append(loaded["lines"][0]["is_reference"] is True
          and loaded["lines"][1]["is_reference"] is False)
print(f"[4] per-line reference tag and angle preserved: "
      f"{'PASS' if ok[-1] else 'FAIL'}")

ok.append(loaded["processing_params"]["half_width_mm"] == 10.0)
print(f"[5] processing_params (half_width_mm, bin_px, ...) preserved: "
      f"{'PASS' if ok[-1] else 'FAIL'}")

os.remove(path)

print(f"\n{sum(ok)}/{len(ok)} passed")
