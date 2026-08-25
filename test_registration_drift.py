"""
Reproduces the reported symptom: with a real monotonic drift on top of the
oscillatory motion, registering to frame 0 anchors the whole corrected
sequence to one end of that drift, making every output image look shifted
in one consistent direction.  register_frames(reference="middle") should
centre the correction instead, roughly halving the apparent net shift.
"""
import numpy as np
from scipy.ndimage import shift as ndshift, gaussian_filter
from lockin_thermography import register_frames

rng = np.random.default_rng(4)
h, w, fps, f = 60, 160, 5.0, 0.1
n = int(300 * fps)
t = np.arange(n) / fps

i_edge, sigma = 80, 3.0
x = np.arange(w)
edge = np.exp(-((x - i_edge) ** 2) / (2 * sigma ** 2))
texture = gaussian_filter(rng.normal(0, 1, (h, w)), sigma=1.5)
texture *= 0.3 / texture.std()
pattern = np.tile(edge, (h, 1)) + texture

# A real rig-settling drift (monotonic, ~20 px total) plus the usual
# excitation-locked breathing (~0.35 px) -- the drift dwarfs the breathing,
# same as "tens of pixels" in the real recording.
drift_total = 20.0
dx_true = (drift_total * (t / t[-1])) + 0.35 * np.sin(2 * np.pi * f * t)

frames_base = np.empty((n, h, w), dtype=np.float64)
for i in range(n):
    frames_base[i] = ndshift(pattern, (0, dx_true[i]), order=3, mode="nearest")
frames_base += rng.normal(0, 0.01, frames_base.shape)
frames_base = frames_base.astype(np.float32)

ok = []
for reference in ("first", "middle"):
    frames = frames_base.copy()
    print(f"--- reference={reference} ---")
    register_frames(frames, t, f, upsample_factor=20, reference=reference)

    # "Shifted in one consistent direction" = the corrected sequence's mean
    # frame sits far from where the RAW sequence's mean frame was -- measure
    # that displacement via the ridge's fitted centre position pre/post.
    raw_mean_peak = np.argmax(gaussian_filter(frames_base.mean(axis=0)[h // 2], 1))
    reg_mean_peak = np.argmax(gaussian_filter(frames.mean(axis=0)[h // 2], 1))
    net_shift = abs(reg_mean_peak - raw_mean_peak)
    print(f"  net shift of the corrected sequence's average vs raw average: "
          f"{net_shift} px")
    ok.append((reference, net_shift))
    print()

first_shift = dict(ok)["first"]
middle_shift = dict(ok)["middle"]
print(f"first: {first_shift} px, middle: {middle_shift} px")
result = middle_shift < 0.6 * first_shift
print(f"middle-frame reference roughly halves the net shift: "
      f"{'PASS' if result else 'FAIL'}")
