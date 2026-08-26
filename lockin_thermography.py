#!/usr/bin/env python3
"""
Lock-in thermography analysis for laser deletion line inspection.

Takes a radiometric IR image sequence recorded while the part is powered with a
square-wave-modulated supply, and extracts the component of the thermal response
that is coherent with the excitation frequency.

Inputs:
    A 3D array of frames, shape (n_frames, height, width), in either raw counts
    or degrees C.  Both work -- lock-in is a linear operation, so an affine
    radiometric conversion only rescales the amplitude image and leaves phase
    untouched.

Outputs:
    amplitude image, phase image, spatially high-passed amplitude, an
    off-frequency noise floor, and profiles along / across a deletion line.

Dependencies: numpy, scipy, matplotlib, scikit-image (for register_frames())
Optional: flirpy (pip install flirpy) for direct .csq / .seq import -- see
load_csq() below.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import (gaussian_filter, binary_erosion, binary_dilation,
                           binary_closing, binary_fill_holes,
                           distance_transform_edt, map_coordinates, label)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


def _fmt_dt(seconds):
    if not np.isfinite(seconds):
        return "?"
    m, s = divmod(max(seconds, 0), 60)
    return f"{int(m)}m{s:04.1f}s" if m else f"{s:.1f}s"


class _stage:
    """
    Context manager that prints a "name..." header immediately and a
    "done (elapsed)" footer on exit, so a slow step shows up as a status
    line rather than as silence.
    """
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        print(f"{self.msg}...", flush=True)
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f"  done ({_fmt_dt(time.time() - self.t0)})")
        return False


def _progress(i, n, t0, last, prefix="  frame"):
    """
    Throttled progress line for a tight per-item loop: fraction done,
    processing rate, and ETA.  Returns the new `last` report time so the
    caller can pass it back in on the next iteration.

    Redraws in place (\\r) on an interactive terminal; on redirected output
    (a log file, CI) that would just leave thousands of \\r-separated
    fragments, so it falls back to one line every few seconds instead.
    """
    now = time.time()
    last_item = i == n - 1
    every_s = 1.0 if sys.stdout.isatty() else 5.0
    if now - last < every_s and not last_item:
        return last
    rate = (i + 1) / max(now - t0, 1e-9)
    eta = (n - i - 1) / rate if rate > 0 else float("nan")
    line = (f"{prefix} {i + 1}/{n} ({100 * (i + 1) / n:.0f}%)  "
            f"{rate:.1f}/s  ETA {_fmt_dt(eta)}")
    if sys.stdout.isatty():
        print(line + " " * max(0, 70 - len(line)),
              end="\n" if last_item else "\r", flush=True)
    else:
        print(line, flush=True)
    return now


# ----------------------------------------------------------------------------
# 1. LOADING
# ----------------------------------------------------------------------------

def load_sequence(path):
    """
    Load frames as (n_frames, h, w) float array from a saved export (.npy,
    CSV/TIFF sequence, etc).  For a FLIR .csq / .seq recorded straight off
    the camera, use load_csq() instead -- analyse() dispatches to it
    automatically based on the file extension.

    Replace the body of this function with whatever matches your export.
    The rest of the pipeline does not care.
    """
    # Example for a saved numpy array:
    size_mb = os.path.getsize(path) / 1e6
    print(f"  reading {path} ({size_mb:.0f} MB)...", flush=True)
    return np.load(path).astype(np.float32)


def _decode_raw_record(fff):
    """
    Return a frame's raw 16-bit counts as a (h, w) uint16 array.

    The FFF record's "subtype" byte is not a reliable guide to how the pixel
    data is actually packed -- it varies by camera/firmware and doesn't match
    what ExifTool's own FLIR.pm (the most battle-tested FLIR parser there is)
    keys off.  ExifTool instead sniffs the data itself, so we do the same:
    PNG signature -> PNG-compressed (common on ResearchIR/CSQ exports);
    JPEG SOI marker -> JPEG-LS-compressed (also seen on CSQ, decoded here
    with pylibjpeg since PNG/JPEG-LS both wrap 16-bit raw counts, not visual
    JPEG); otherwise, if the byte count matches h*w*2 exactly, it's flat
    uncompressed 16-bit data.
    """
    record = next(r for r in fff.records if r.record_type == 1)
    data = fff.data[
        record.record_offset + 0x20 : record.record_offset + record.record_length
    ]
    h, w = fff.height, fff.width

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        import io
        from PIL import Image
        img = np.array(Image.open(io.BytesIO(data)))
        return img.reshape(h, w).astype("uint16")

    if data[:3] == b"\xff\xd8\xff":
        from libjpeg import decode as jpegls_decode
        img = jpegls_decode(data)
        return np.asarray(img).reshape(h, w).astype("uint16")

    if len(data) == h * w * 2:
        return np.frombuffer(data, dtype="uint16").reshape(h, w)

    raise ValueError(
        f"can't decode raw frame: {len(data)} bytes for a {w}x{h} frame "
        f"(expected {h * w * 2} uncompressed, or a PNG/JPEG signature). "
        f"First bytes: {data[:16].hex()}"
    )


def load_csq(path):
    """
    Load a FLIR .csq / .seq radiometric recording directly, using flirpy
    (pure-Python FFF container parser) for the file structure and
    calibration metadata, plus PNG/JPEG-LS decoding (Pillow, pylibjpeg) for
    whichever way this particular camera/firmware happens to have packed
    the pixel data -- see _decode_raw_record() above.

    pip install flirpy pillow pylibjpeg pylibjpeg-libjpeg

    Returns (frames [n, h, w] float32, degrees C, using the camera's own
    Planck-law calibration constants; t [n] float64 seconds).
    """
    from flirpy.io.seq import Seq
    from flirpy.util.raw import raw2temp

    seq = Seq(path)
    n = len(seq)
    if n == 0:
        raise ValueError(f"no frames found in {path}")

    first = seq[0]
    h, w = first.height, first.width
    frames = np.empty((n, h, w), dtype=np.float32)
    raw_ts = np.full(n, np.nan)

    print(f"  decoding {n} frames ({w}x{h}) from {path}...", flush=True)
    t0 = last = time.time()
    for i, fff in enumerate(seq):
        raw = _decode_raw_record(fff)
        frames[i] = raw2temp(raw.astype(np.float64), fff.meta)
        if "Timestamp" in fff.meta:
            raw_ts[i] = fff.meta["Timestamp"]
        last = _progress(i, n, t0, last, prefix="  decoded")

    # Camera timestamps here are whole seconds. That's plenty of resolution
    # for the slow excitation frequencies this pipeline targets and catches
    # dropped frames over a long record, but on a fast recording many frames
    # will share the same integer second -- detect that and fall back to a
    # constant frame rate instead of a lumpy, wrong time axis.
    if not np.any(np.isnan(raw_ts)) and np.all(np.diff(raw_ts) > 0):
        t = raw_ts - raw_ts[0]
    else:
        fps = first.meta.get("FrameRate", 30.0)
        print(f"  no usable per-frame timestamps -- assuming constant {fps:.3g} fps")
        t = np.arange(n, dtype=np.float64) / fps

    print(f"  loaded {n} frames ({w}x{h}) in {_fmt_dt(time.time() - t0)}")
    return frames, t


def build_time_vector(n_frames, fps):
    """
    Frame timestamps in seconds.

    If your export carries per-frame timestamps, USE THOSE instead of assuming
    a constant frame rate.  Dropped frames will smear the lock-in result, and
    handheld cameras drop frames more often than their spec implies.
    """
    return np.arange(n_frames) / fps


def decimate(frames, t, target_fps):
    """
    Block-average groups of frames down to target_fps.

    This matters more than it sounds.  A 30 minute record at 30 Hz and full
    T540 resolution is roughly 35 GB in float32 -- you will not hold that in
    memory.  At 0.05 Hz excitation you only need about 8 samples per cycle, so
    decimating 30 Hz to 1-2 Hz costs nothing and shrinks the problem by 15-30x.

    Block averaging (not sub-sampling) also acts as an anti-alias filter and
    lowers the noise floor by sqrt(block) before the lock-in even starts.
    """
    block = max(1, int(round(len(t) / (t[-1] - t[0]) / target_fps)))
    if block == 1:
        return frames, t
    n = (len(t) // block) * block
    frames = frames[:n].reshape(n // block, block, *frames.shape[1:]).mean(axis=1)
    t = t[:n].reshape(n // block, block).mean(axis=1)
    print(f"  decimated by {block}x -> {len(t)} frames at {1/np.mean(np.diff(t)):.2f} Hz")
    return frames, t


# ----------------------------------------------------------------------------
# 2. FRAME QUALITY & MOTION REGISTRATION
# ----------------------------------------------------------------------------

def reject_outlier_frames(frames, t, mad_threshold=8.0, max_reject_frac=0.05):
    """
    Detect and drop frames corrupted by a camera glitch -- most commonly a
    NUC (non-uniformity correction) shutter recalibration event -- that
    remove_global_offsets() can't fix on its own, because that only
    corrects a spatially UNIFORM per-frame offset.  A NUC event that also
    corrupts pixel content locally (a documented FLIR quirk) leaves a
    small, high-contrast, spatially localized anomaly that survives
    straight into the lock-in average: invisible in the raw amplitude image
    (dominated by the smooth bulk heating pattern) but sharp after spatial
    high-pass strips everything else away -- it looks like a patch of a
    different image pasted into the part.

    Scores each frame by the 99.5th percentile of |frame[i] - frame[i-1]|
    -- how much of it changed abruptly from its neighbour.  A percentile
    rather than a mean is what catches a small corrupted patch: it doesn't
    need to move the WHOLE frame's average, only enough pixels within it.
    A frame is flagged only if BOTH its neighbouring jumps are anomalous (a
    MAD-based robust z-score against the whole recording) -- i.e. it
    differs abnormally from what comes before AND after it.  A jump on only
    one side is more likely a real, persistent step (drift, a genuine fast
    thermal transient) than a one-frame glitch, so those are left alone
    rather than guessed at.

    Frames are dropped outright, not interpolated: lockin() is just a
    weighted dot product against whatever time vector it's given, so
    removing a frame and its timestamp together keeps the math and the
    (2/n) normalization consistent, with no gap to fill.

    Run this before register_frames(): a corrupted frame chosen as the
    registration reference would otherwise misalign every other frame
    against it.

    Raises ValueError if more than max_reject_frac of frames would be
    dropped -- at that point something systematic is more likely than a run
    of isolated glitches, and the detector itself should be treated with
    suspicion rather than trusted to silently discard that much of the
    recording.
    """
    n = len(frames)
    if n < 3:
        return frames, t

    score = np.empty(n - 1)
    t0 = last = time.time()
    for i in range(n - 1):
        score[i] = np.percentile(np.abs(frames[i + 1] - frames[i]), 99.5)
        last = _progress(i, n - 1, t0, last, prefix="  scanned")

    typical = np.median(score)
    mad = np.median(np.abs(score - typical)) + 1e-9
    z = (score - typical) / (1.4826 * mad)     # 1.4826 makes MAD ~= std for Gaussian data
    spike = z > mad_threshold

    bad = np.zeros(n, dtype=bool)
    bad[0], bad[-1] = spike[0], spike[-1]
    bad[1:-1] = spike[:-1] & spike[1:]

    n_bad = int(bad.sum())
    if n_bad > max_reject_frac * n:
        raise ValueError(
            f"{n_bad}/{n} frames ({100 * n_bad / n:.1f}%) flagged as corrupted -- "
            f"more than max_reject_frac={max_reject_frac:.0%} allowed. That many "
            "drops suggests something systematic (a real fast transient, or "
            "mad_threshold set too low) rather than isolated glitches -- "
            "inspect before trusting this many frames to be dropped."
        )
    if n_bad == 0:
        print("  no corrupted frames detected")
        return frames, t

    idx = np.nonzero(bad)[0]
    print(f"  dropped {n_bad} corrupted frame(s) at index {idx.tolist()} "
          f"(t={np.round(t[idx], 2).tolist()}s)")
    keep = ~bad
    return frames[keep], t[keep]


def detect_frozen_frames(frames, t, similarity_z=-3.0, min_run=3):
    """
    Flag contiguous runs of frames suspiciously similar to their neighbour.

    A real thermal sequence always carries some sensor noise frame to
    frame, so a run where that difference collapses toward zero for many
    consecutive frames means the camera or the loader is repeating stale
    frame data, not that the scene actually froze.  This is a DIFFERENT
    failure mode from reject_outlier_frames(): that catches a single frame
    that jumped abnormally FAR from its neighbours; this catches a run of
    frames that stayed abnormally CLOSE to one -- sometimes for long
    stretches -- a frozen block, not an isolated glitch, and one that
    survives reject_outlier_frames() undetected because nothing about it
    looks like a sudden jump.

    A frozen run is also the natural explanation if register_frames()'s
    printed offset range shows a long, flat stretch pinned at exactly
    (0, 0): frames that are genuinely identical register with zero offset
    by construction, and if the registration reference frame happens to
    fall inside that stretch, every OTHER frame outside it can appear to
    need one large, physically implausible correction to align with it --
    that's a strong sign of exactly this, not real part motion.

    Uses the MEDIAN (not a high percentile) of |frame[i] - frame[i-1]| -- a
    global similarity measure, unlike reject_outlier_frames()'s 99.5th
    percentile, which is deliberately sensitive to a small LOCALIZED spike
    instead.

    Returns a list of (start_idx, end_idx) frame-index ranges (inclusive),
    each covering min_run or more suspiciously-similar consecutive frames.
    This only reports -- it does not modify frames or drop anything, since
    how much of a long run to discard (and whether it's actually a data
    fault rather than a genuinely quiet period, e.g. after the excitation
    was switched off) is a judgement call for you to make.
    """
    n = len(frames)
    if n < min_run + 1:
        return []

    score = np.empty(n - 1)
    for i in range(n - 1):
        score[i] = np.median(np.abs(frames[i + 1].astype(np.float32)
                                    - frames[i].astype(np.float32)))

    typical = np.median(score)
    mad = np.median(np.abs(score - typical)) + 1e-9
    z = (score - typical) / (1.4826 * mad)
    frozen = z < similarity_z          # a LOW outlier: too similar, not too different

    runs = []
    i = 0
    while i < len(frozen):
        if frozen[i]:
            j = i
            while j < len(frozen) and frozen[j]:
                j += 1
            if j - i + 1 >= min_run:            # a run of k True diffs spans k+1 frames
                runs.append((i, j))
            i = j
        else:
            i += 1

    if runs:
        print(f"  WARNING: {len(runs)} run(s) of suspiciously frozen/duplicate "
              "frames detected:")
        for a, b in runs:
            print(f"    frames [{a}, {b}]  (t=[{t[a]:.1f}, {t[b]:.1f}]s, "
                  f"{b - a + 1} frames spanning {t[b] - t[a]:.1f}s) -- check "
                  "whether these are genuinely near-identical (a camera or "
                  "loader freeze) or a real quiet/settled period")
    else:
        print("  no frozen frame runs detected")
    return runs


def diagnose_pixel(frames, t, y, x, f_excite, path="lockin_pixel_diag.png",
                   title=None):
    """
    Plot one pixel's raw time series with its lock-in fit overlaid, and
    save to `path`.

    The single fastest way to tell a real, physically reasonable signal
    (a smooth thermal response, or genuine coherent motion) apart from a
    data artefact (a step, a single-frame spike, a frozen run) at a
    specific location -- an amplitude number alone only says "how much
    energy is at this frequency", not "does it look reasonable getting
    there".  A real leakage-line pixel should look like a clean sinusoid
    riding on a slow trend; a spike, a step, or a patch of near-constant
    value that then jumps is a data problem, not a thermal one.
    """
    series = frames[:, y, x].astype(np.float64)
    n = len(t)
    s, c = np.sin(2 * np.pi * f_excite * t), np.cos(2 * np.pi * f_excite * t)
    S = (2.0 / n) * np.dot(s, series)
    C = (2.0 / n) * np.dot(c, series)
    fit = series.mean() + S * s + C * c

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, series, lw=0.7, alpha=0.8, label="raw pixel value")
    ax.plot(t, fit, lw=1.6,
            label=f"lock-in fit @ {f_excite:.3g} Hz (amplitude {np.hypot(S, C):.3g})")
    ax.set_xlabel("time [s]"); ax.set_ylabel("pixel value")
    ax.set_title(title or f"pixel (y={y}, x={x})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}  (pixel y={y}, x={x})")


def _coherent_amplitude(signal, t, f):
    """Amplitude of `signal` at frequency f -- the same single-bin DFT lockin() does, for a 1D series."""
    n = len(t)
    s, c = np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)
    S = (2.0 / n) * np.dot(s, signal)
    C = (2.0 / n) * np.dot(c, signal)
    return np.hypot(S, C)


def register_frames(frames, t, f_excite, upsample_factor=20, normalization=None,
                    reference="middle"):
    """
    Sub-pixel-register every frame to a common reference frame, in place,
    via phase correlation.

    reference picks which frame every other frame is aligned to: "middle"
    (default) uses the temporally central frame; "first" uses frame 0, as
    in the naive version of this check.  This matters whenever the part has
    a real monotonic drift on top of the oscillatory "breathing" this
    function is chiefly aimed at -- rig settling, a cold-start transient,
    anything that doesn't reverse over the recording.  Anchoring to frame 0
    pins the whole corrected sequence to whichever end of that drift frame
    0 happened to sit at, so once it's removed, every image can look
    shifted by the drift's full range relative to what the uncorrected
    footage showed.  Anchoring to the middle frame instead centres the
    correction in that range, which is the more representative choice and
    the fix if you're seeing every image shifted in one consistent
    direction rather than just cleaned up.  A large offset range on its own
    isn't evidence of a bug -- see the printed drift-vs-residual breakdown
    below to tell a real monotonic drift (expected, correctly removed)
    apart from the excitation-locked oscillation (the dipole cause) apart
    from plain frame-to-frame jitter.

    normalization=None (plain cross-correlation) rather than skimage's own
    default of "phase" (phase-only correlation): phase normalisation
    equalises every frequency's magnitude to 1 before correlating, which is
    fine for genuinely broadband, richly-textured images but on smooth
    thermal imagery -- most of the frame's real content sits at low spatial
    frequency -- that same normalisation hands high-frequency sensor noise
    equal weight to the real signal, and badly underestimates exactly the
    sub-pixel shifts (a few tenths of a pixel) this function exists to
    catch.  Plain cross-correlation instead weights each frequency by its
    actual power, which is what recovers small shifts reliably here.

    Deletion lines are optical/emissivity edges as well as thermal ones. If
    the part physically moves in lockstep with the excitation -- e.g. a
    freely-expanding ply breathing in and out as it heats and cools on each
    half-cycle -- the lock-in amplifies that coherent edge motion exactly as
    enthusiastically as it amplifies real heat flow.  A derivative of a step
    (the image sliding across a sharp edge) is a dipole: positive on one
    side, negative on the other, crossing zero at the edge -- which is
    exactly what a real leakage line does NOT look like (that's a symmetric
    bump, positive on both sides, peaking at the line). If your amplitude or
    phase images show that red/blue or +/- split straddling every line,
    this is almost always why, and it's the one noise source lock-in can't
    reject, because it's coherent with f by construction.

    Registering here, before decimation, matters: block-averaging first
    would blur any sub-block motion into the frames themselves before this
    function ever sees it.

    Returns (dy, dx) offsets applied, one pair per frame (zero at the
    reference frame), for the caller to inspect or plot -- e.g. via
    track_part_centroid() for an independent check that doesn't depend on
    phase correlation at all.
    """
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        raise ImportError(
            "register_frames() needs scikit-image (pip install scikit-image) "
            "-- or pass register=False to analyse() to skip motion "
            "registration, at the cost of dipole artifacts from any part "
            "motion coherent with the excitation"
        ) from None
    from scipy.ndimage import shift as ndshift

    n = len(frames)
    ref_idx = {"first": 0, "middle": n // 2}[reference]
    ref = frames[ref_idx].copy()
    offsets = np.zeros((n, 2))
    order = [i for i in range(n) if i != ref_idx]
    t0 = last = time.time()
    for k, i in enumerate(order):
        dy_dx, _, _ = phase_cross_correlation(ref, frames[i],
                                              upsample_factor=upsample_factor,
                                              normalization=normalization)
        offsets[i] = dy_dx
        frames[i] = ndshift(frames[i], dy_dx, order=1, mode="nearest")
        last = _progress(k, len(order), t0, last, prefix="  registered")

    # Decompose into a linear drift trend (real, monotonic, expected to be
    # fully removed) and the residual around it (oscillation + jitter) --
    # printed range alone conflates "the part genuinely drifted a lot" with
    # "registration is misbehaving", and only the excitation-coherent slice
    # of the residual is the dipole-causing artifact this function targets.
    tc = t - t.mean()
    denom = np.dot(tc, tc)
    drift = np.empty(2)
    for axis in (0, 1):
        slope = np.dot(tc, offsets[:, axis]) / denom
        drift[axis] = slope * (t[-1] - t[0])
    residual = offsets - np.outer(tc, [np.dot(tc, offsets[:, 0]) / denom,
                                       np.dot(tc, offsets[:, 1]) / denom])

    dy_amp = _coherent_amplitude(residual[:, 0], t, f_excite)
    dx_amp = _coherent_amplitude(residual[:, 1], t, f_excite)
    print(f"  offset range: dy [{offsets[:, 0].min():.2f}, {offsets[:, 0].max():.2f}] px, "
          f"dx [{offsets[:, 1].min():.2f}, {offsets[:, 1].max():.2f}]  "
          f"(registered to frame {ref_idx} of {n})")
    print(f"  linear drift over the recording: dy {drift[0]:+.2f} px, dx {drift[1]:+.2f} px")
    print(f"  residual motion coherent with excitation: dy {dy_amp:.3f} px, dx {dx_amp:.3f} px")
    if max(dy_amp, dx_amp) > 0.1:
        print("  -- motion locked to the excitation frequency at a "
              "few tenths of a pixel or more: expect dipole artifacts at "
              "every sharp edge (deletion lines, part boundary) if this "
              "hadn't been corrected")
    if max(abs(drift[0]), abs(drift[1])) > 5:
        print(f"  -- large linear drift ({np.hypot(*drift):.1f} px over the "
              "recording): likely rig settling or a slow real motion, not "
              "excitation-driven -- registration removes it either way, "
              "but it's a separate finding from the dipole check above")
    return offsets


def track_part_centroid(frames, threshold_frac=0.4, percentile=95):
    """
    Per-frame intensity centroid of the hot region -- a check on part motion
    that is independent of register_frames() (no phase correlation
    involved), useful to run both before and after registration: plot the
    centroid position against time in both cases and see whether the
    oscillation at the excitation frequency collapses.

    Thresholds each frame independently (the same fraction-of-peak rule as
    part_mask()) rather than reusing one fixed mask, so it tracks the part
    even as it moves under a fixed window.

    Returns (cy, cx) arrays, one value per frame (NaN for a frame with no
    pixels above threshold).
    """
    n = len(frames)
    cy, cx = np.full(n, np.nan), np.full(n, np.nan)
    for i, frame in enumerate(frames):
        mask = frame > threshold_frac * np.percentile(frame, percentile)
        if not mask.any():
            continue
        yy, xx = np.nonzero(mask)
        w = np.clip(frame[mask] - frame[mask].min(), 0, None)
        wsum = w.sum()
        if wsum > 0:
            cy[i], cx[i] = np.average(yy, weights=w), np.average(xx, weights=w)
        else:
            cy[i], cx[i] = yy.mean(), xx.mean()
    return cy, cx


# ----------------------------------------------------------------------------
# 3. PRE-CONDITIONING
# ----------------------------------------------------------------------------

def remove_global_offsets(frames):
    """
    Subtract each frame's spatial median from that frame.

    This is what kills the offset steps produced by the camera's internal NUC
    shutter, plus any global ambient drift.  It works because those artifacts
    shift the WHOLE image at once, while the defect signal is confined to a
    thin line.

    Caveat: if your defect covered most of the field of view, this would remove
    real signal along with the artifact.  For line-shaped features occupying a
    few percent of the pixels, the loss is negligible.
    """
    med = np.median(frames.reshape(frames.shape[0], -1), axis=1)
    return frames - med[:, None, None]


def detrend_per_pixel(frames, t):
    """
    Remove a per-pixel linear trend in time.

    The part is still slowly equilibrating with the room during the run.  That
    is a ramp, not a periodic signal, but a ramp has energy at every frequency
    including yours, so removing it lowers the noise floor.
    """
    tc = (t - t.mean()).astype(frames.dtype)
    denom = np.dot(tc, tc)
    slope = np.tensordot(tc, frames, axes=(0, 0)) / denom     # (h, w)
    mean = frames.mean(axis=0)                                 # (h, w)
    frames -= mean[None, :, :]
    frames -= slope[None, :, :] * tc[:, None, None]
    return frames


def trim_to_whole_cycles(frames, t, f, skip_cycles=2):
    """
    Discard the initial thermal transient and truncate to an integer number of
    excitation periods.

    Both matter.  The first cycles are contaminated by the part warming from
    ambient to its cyclic steady state.  A non-integer number of cycles causes
    spectral leakage -- energy from the strong DC and low-frequency components
    bleeds into your measurement bin and raises the noise floor.
    """
    period = 1.0 / f
    t0 = t[0] + skip_cycles * period
    usable = t[t >= t0]
    n_cycles = int(np.floor((usable[-1] - usable[0]) / period))
    if n_cycles < 3:
        raise ValueError("Fewer than 3 usable cycles after trimming.")
    t_end = usable[0] + n_cycles * period
    mask = (t >= t0) & (t < t_end)
    print(f"  using {mask.sum()} frames = {n_cycles} full cycles")
    return frames[mask], t[mask]


# ----------------------------------------------------------------------------
# 4. THE LOCK-IN ITSELF
# ----------------------------------------------------------------------------

def lockin(frames, t, f):
    """
    Project every pixel's time series onto sin(2*pi*f*t) and cos(2*pi*f*t).

    This is a single-bin discrete Fourier transform.  Equivalent view: it is a
    bandpass filter centred at f whose bandwidth is roughly 1 / (total record
    length).  Twenty cycles at 0.05 Hz gives a bandwidth near 0.0025 Hz, so
    essentially all broadband noise is rejected.

    Returns (amplitude, phase_radians).
    """
    n = len(t)
    s = np.sin(2 * np.pi * f * t)
    c = np.cos(2 * np.pi * f * t)

    S = (2.0 / n) * np.tensordot(s, frames, axes=(0, 0))
    C = (2.0 / n) * np.tensordot(c, frames, axes=(0, 0))

    return np.hypot(S, C), np.arctan2(S, C)


def noise_floor(frames, t, f, factor=1.37):
    """
    Amplitude at a frequency that is deliberately NOT commensurate with the
    excitation or its harmonics.

    There is no real signal there, so whatever comes back is the measurement
    noise floor in the same units as your result.  This is how you decide
    whether a faint line in the amplitude image is a defect or wishful
    thinking.  Require at least 3x, preferably 5x, over this number.
    """
    amp, _ = lockin(frames, t, f * factor)
    return np.median(amp)


# ----------------------------------------------------------------------------
# 5. SPATIAL PROCESSING
# ----------------------------------------------------------------------------

def spatial_highpass(amp, sigma_px):
    """
    Subtract a blurred copy of the amplitude image.

    The bulk zone heating is spatially smooth -- it varies over the scale of
    the whole part.  A leakage line is sharp.  Choosing sigma at roughly two to
    three times the thermal diffusion length (in pixels) suppresses the former
    while preserving the latter.

    Diffusion length: mu = sqrt(alpha / (pi * f)), with alpha about 5e-7 m^2/s
    for glass.  At 0.05 Hz that is about 1.8 mm; convert to pixels using your
    mm-per-pixel scale.

    NOT part of analyse()'s default pipeline: this assumes the bulk field is
    smooth everywhere except at the defect.  On a part where adjacent zones
    have a genuine, physical STEP in power density (different heater
    geometry, a graded coating thickness -- a real difference, not a
    defect), subtracting a blurred copy turns that step into a spurious
    ANTISYMMETRIC dipole: positive on one side, negative on the other,
    crossing zero at the line.  That's because a Gaussian blur of a step is
    itself a smoothed step, and subtracting a smoothed step from a sharp one
    leaves their difference -- which is approximately the step's own
    derivative, an odd function about the step's centre.  A genuine leakage
    signal is a SYMMETRIC bump (heat spreads both directions alike), so the
    step-derived dipole doesn't cancel out of the difference; on a part
    with real zone-to-zone steps it can swamp the much smaller symmetric
    signal entirely.  symmetric_antisymmetric_profile() handles this
    correctly instead, by decomposing the step and the candidate signal
    directly rather than filtering first.  Still useful standalone (or for
    a part that has no real zone-to-zone steps, only isolated line
    features) -- see find_deletion_lines()/auto_fiducial_roi(), which still
    use it.
    """
    return amp - gaussian_filter(amp, sigma_px)


def _largest_component(mask):
    """Keep only the largest connected True region of a binary mask."""
    lbl, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0        # background label
    return lbl == np.argmax(sizes)


def _point_in_mask(mask, x, y):
    yi, xi = int(round(y)), int(round(x))
    return 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and bool(mask[yi, xi])


def _component_size_at(mask, y, x):
    """
    Size (px) of the connected component of `mask` containing point (x, y),
    or 0 if that point is False / out of bounds.  Used only for diagnostics
    -- to tell "your point is in the wrong class entirely" apart from "your
    point's class is right, but got fragmented and lost to
    _largest_component()", which look identical from the outside (both end
    up "not in the mask") but need a different fix (mask_polarity vs.
    close_px / mask_roi).
    """
    yi, xi = int(round(y)), int(round(x))
    if not (0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]) or not mask[yi, xi]:
        return 0
    lbl, _ = label(mask)
    return int((lbl == lbl[yi, xi]).sum())


def _region_in_mask(mask, y0, y1, x0, x1, min_frac=0.5):
    region = mask[y0:y1, x0:x1]
    return region.size > 0 and region.mean() >= min_frac


def _polygon_mask(shape, polygon_xy):
    """Boolean mask, True inside the (x, y) polygon `polygon_xy`."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    from matplotlib.path import Path
    inside = Path(polygon_xy).contains_points(pts)
    return inside.reshape(h, w)


def _choose_polarity(amp, high, low, fiducial_roi, variance_sigma_px):
    """
    Decide which of the two Otsu classes (`high` = brighter, `low` =
    darker) is the part.  See part_mask()'s docstring for the criteria.
    Returns (chosen: "bright"|"dark", chosen_mask).
    """
    high_cc = _largest_component(high)
    low_cc = _largest_component(low)

    if fiducial_roi is not None:
        y0, y1, x0, x1 = fiducial_roi
        roi_high = high_cc[y0:y1, x0:x1]
        roi_low = low_cc[y0:y1, x0:x1]
        if roi_high.size:
            high_frac, low_frac = float(roi_high.mean()), float(roi_low.mean())
            if max(high_frac, low_frac) > 0.5:
                choice = "bright" if high_frac >= low_frac else "dark"
                print(f"  auto polarity: fiducial ROI is "
                      f"{100 * max(high_frac, low_frac):.0f}% inside the "
                      f"'{choice}' class's region -> selecting '{choice}'")
                return choice, (high_cc if choice == "bright" else low_cc)

    # No decisive fiducial ROI signal -- fall back to contiguity (does the
    # largest component account for most of the class?) and large-scale
    # variance (the part's coating is usually more spatially uniform than a
    # cluttered background, at the scale of variance_sigma_px).
    smooth = gaussian_filter(amp, sigma=variance_sigma_px)
    high_contig = high_cc.sum() / max(high.sum(), 1)
    low_contig = low_cc.sum() / max(low.sum(), 1)
    high_var = float(smooth[high_cc].var()) if high_cc.any() else float("inf")
    low_var = float(smooth[low_cc].var()) if low_cc.any() else float("inf")
    high_score = high_contig / (high_var + 1e-9)
    low_score = low_contig / (low_var + 1e-9)
    choice = "bright" if high_score >= low_score else "dark"
    print(f"  auto polarity (no decisive fiducial ROI): bright class "
          f"contiguity={high_contig:.2f} var={high_var:.3g}, dark class "
          f"contiguity={low_contig:.2f} var={low_var:.3g} -> selecting "
          f"'{choice}'")
    return choice, (high_cc if choice == "bright" else low_cc)


def part_mask(amp, mask_polarity="auto", erosion_px=10, close_px=3,
             fiducial_roi=None, line_endpoints=None, mask_roi=None,
             min_area_frac=0.05, max_area_frac=0.95, variance_sigma_px=15):
    """
    Binary mask of the part footprint within the frame.

    The part can read either BRIGHTER or DARKER than its surroundings
    depending on framing -- a part filling most of a close-framed shot
    commonly reads darker than the sliver of warm background around it, the
    opposite of a part imaged from farther back with cool background all
    around it.  Assuming one polarity is what silently hands every
    downstream step (phase reference, profiles, statistics) the background
    instead of the part.

    mask_polarity controls how the part is picked out of a two-class Otsu
    split of `amp`:
      'auto' (default) -- decide automatically.  If fiducial_roi is given
               and falls decisively (>50% coverage) inside one class's
               largest connected component, that class wins outright -- the
               user's own chosen reference site is the strongest available
               signal for which class is the part.  Otherwise falls back to
               preferring the class that is more spatially CONTIGUOUS (its
               largest connected component captures most of the class) and
               has lower large-scale variance (see _choose_polarity()) --
               printed either way so a bad call is visible in the log, not
               just downstream.
      'bright' / 'dark' -- skip auto-detection and force that class, for
               when auto gets it wrong.
    mask_roi bypasses thresholding entirely: a polygon -- a list of (x, y)
               vertices, e.g. from interactive_setup(mask_roi=True) or
               hand-built and stored in roi_config.json -- that IS the part
               outline.  Use when the part doesn't separate from the
               background by brightness at all.

    Whichever way the part class is chosen, close_px first bridges any thin
    gap in that class -- a deletion line, a scratch, a specular streak --
    via binary closing (dilate then erode by close_px), so a high-contrast
    feature crossing the part doesn't split it into two disconnected
    islands.  Without this, _largest_component() below would silently keep
    only ONE side of such a divider and drop the other -- which looks
    identical to a polarity failure from outside (whatever's on the
    dropped side reads as "not in the mask") but needs a different fix:
    raising close_px (or, if the divider is very wide/high-contrast,
    mask_roi) rather than switching mask_polarity.  Sizes typical of a
    deletion line's own width are covered by the default (3px); a wider
    divider needs a larger close_px.  Then the largest single connected
    component is kept (drops stray bright/dark specks elsewhere in frame),
    holes are filled (closes over whatever of the deletion lines close_px
    didn't already bridge), and the boundary is eroded inward by
    erosion_px so the true part edge -- which the spatial high-pass
    otherwise rings against -- doesn't contaminate the interior.

    Sanity checks, since a masking failure otherwise propagates silently
    into every downstream number:
      - fiducial_roi (y0, y1, x0, x1) and every point in line_endpoints (a
        list of (p0, p1) pairs, each an (x, y) tuple) are checked against
        the mask BEFORE erosion.  A deletion line legitimately runs busbar
        to busbar right up to the part's true edge, so checking against
        the eroded interior would flag correctly-placed endpoints near the
        edge as errors.  Either falling outside the pre-erosion footprint
        raises -- naming inverted polarity as the likely cause, but also
        reporting (when thresholding was used, i.e. mask_roi wasn't
        passed) the point's own raw Otsu class and that class's raw
        connected-component size at that point vs. the size of the
        component actually selected, so a fragmentation failure (fix:
        close_px or mask_roi) isn't mistaken for a polarity failure (fix:
        mask_polarity) or vice versa.
      - the final mask's area is printed as a percentage of the frame on
        every run, and raises if it's under min_area_frac or over
        max_area_frac -- an (almost) all-or-nothing mask is symptomatic of
        exactly this kind of polarity failure, not a real part footprint.

    Returns the final (filled, largest-component, eroded) boolean mask.
    """
    thresh = high = low = None
    if mask_roi is not None:
        raw = _largest_component(binary_fill_holes(_polygon_mask(amp.shape, mask_roi)))
        source = f"manual ROI polygon ({len(mask_roi)} vertices)"
    else:
        try:
            from skimage.filters import threshold_otsu
        except ImportError:
            raise ImportError(
                "part_mask() needs scikit-image for automatic thresholding "
                "(pip install scikit-image) -- or pass mask_roi to bypass "
                "thresholding with a manually clicked polygon"
            ) from None

        thresh = threshold_otsu(amp)
        high, low = amp > thresh, amp <= thresh
        if close_px > 0:
            high = binary_closing(high, iterations=close_px)
            low = binary_closing(low, iterations=close_px)

        if mask_polarity == "bright":
            raw = _largest_component(high)
            source = f"Otsu threshold {thresh:.4g}, polarity forced 'bright'"
        elif mask_polarity == "dark":
            raw = _largest_component(low)
            source = f"Otsu threshold {thresh:.4g}, polarity forced 'dark'"
        elif mask_polarity == "auto":
            choice, raw = _choose_polarity(amp, high, low, fiducial_roi,
                                           variance_sigma_px)
            source = f"Otsu threshold {thresh:.4g}, auto-selected polarity '{choice}'"
        else:
            raise ValueError(
                f"mask_polarity must be 'auto', 'bright', or 'dark', got "
                f"{mask_polarity!r}"
            )
        raw = binary_fill_holes(raw)

    fail_hint = ("masking likely failed (inverted polarity is the usual "
                "cause on a close-framed image) -- try mask_polarity="
                "'bright'/'dark' explicitly, raise close_px if a line/streak "
                "may be splitting the part in two, or pass mask_roi to click "
                "the part outline by hand")

    def _point_diag(x, y):
        if thresh is None:
            return ""
        yi, xi = int(round(y)), int(round(x))
        if not (0 <= yi < amp.shape[0] and 0 <= xi < amp.shape[1]):
            return "  (point is outside the frame entirely)"
        val = float(amp[yi, xi])
        cls = "bright" if val > thresh else "dark"
        raw_size = _component_size_at(high if cls == "bright" else low, y, x)
        sel_size = int(raw.sum())
        return (f"  (this point has amp={val:.4g}, on the '{cls}' side of the "
                f"Otsu threshold {thresh:.4g}; its own raw connected "
                f"component there is {raw_size}px vs. {sel_size}px for the "
                f"component actually selected -- if raw_size is large but "
                f"still not selected, it's a polarity mismatch; if it's "
                f"small, the part is likely fragmented by something thin "
                f"crossing it, try a larger close_px)")

    if fiducial_roi is not None:
        y0, y1, x0, x1 = fiducial_roi
        if not _region_in_mask(raw, y0, y1, x0, x1):
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            raise ValueError(
                f"fiducial ROI y[{y0}:{y1}] x[{x0}:{x1}] is not inside the "
                f"part mask ({source}) -- {fail_hint}{_point_diag(cx, cy)}"
            )
    for i, (p0, p1) in enumerate(line_endpoints or []):
        for tag, (x, y) in (("p0", p0), ("p1", p1)):
            if not _point_in_mask(raw, x, y):
                raise ValueError(
                    f"line {i} endpoint {tag} at ({x:.0f}, {y:.0f}) is not "
                    f"inside the part mask ({source}) -- "
                    f"{fail_hint}{_point_diag(x, y)}"
                )

    part = binary_erosion(raw, iterations=erosion_px)
    area_frac = float(part.mean())
    print(f"  part mask: {source}, area {100 * area_frac:.1f}% of frame "
          f"(after eroding {erosion_px}px inward)")
    if area_frac < min_area_frac or area_frac > max_area_frac:
        raise ValueError(
            f"part mask covers {100 * area_frac:.1f}% of the frame -- "
            f"outside the expected [{100 * min_area_frac:.0f}%, "
            f"{100 * max_area_frac:.0f}%] range -- {fail_hint}"
        )

    return part


def reference_phase(phase, part, fiducial_roi=None, method="circular_mean",
                    wrap_warn_deg=20.0, wrap_warn_frac=0.03):
    """
    Rotate `phase` (radians, on lockin()'s native arctan2 range (-pi, pi])
    so its branch cut lands away from the bulk of the masked part, and
    return (phase_referenced, ref_rad, wrap_frac).

    method:
      'circular_mean' (default) -- reference to the circular mean of every
               masked (`part`) pixel's phase:
                   ref = angle(mean(exp(1j * phase[part])))
                   phase = angle(exp(1j * (phase - ref)))
               lockin()'s phase is wrapped to (-180, +180] with an
               essentially arbitrary branch-cut location -- wherever
               arctan2 happens to flip sign.  Referencing to a SINGLE site
               (the previous default: the fiducial ROI's median) has no way
               to know whether that site's own phase happens to sit near
               that cut; if it does, subtracting it rotates the cut
               straight into the middle of the part instead of away from
               it, and everything on one side of some arbitrary line reads
               close to +180 while everything just across it reads close to
               -180 -- a real part reading as two, discontinuous halves,
               even though nothing about the underlying phase is
               physically unstable there. The circular mean of the WHOLE
               masked population is the phasor sum's own direction, i.e.
               by construction it points at wherever the population is
               most concentrated -- referencing to it puts the bulk of the
               distribution at 0 and pushes the branch cut out to
               wherever the population is thinnest, which is the best any
               single global rotation can do.
      'fiducial_roi' -- the previous behaviour: reference to the median
               phase within `fiducial_roi` (y0, y1, x0, x1).  Still useful
               when the fiducial site is deliberately meant to be the
               profiles' physical zero, but carries the branch-cut risk
               above; interactive_setup() checks a candidate ROI against
               this same risk before accepting the click (see its
               docstring) specifically because of it.

    After referencing, reports the fraction of masked pixels within
    wrap_warn_deg of +/-180 and warns if it exceeds wrap_warn_frac (a few
    percent, default 3%): if referencing genuinely centred the bulk of the
    distribution, that fraction should be small regardless of which method
    was used.  A large residual means either the part's phase genuinely
    spans more than a full turn (real thermal-wave phase lag CAN do this
    far enough from the excitation source -- see unwrap_phase_field()), or
    `part` includes pixels whose phase is noise-dominated (near-random, so
    some land near the cut no matter where it's rotated to).  Either way,
    phase images/profiles should not be trusted without investigating
    further when this fires.

    Returns phase_referenced (radians, still wrapped to (-pi, pi]),
    ref_rad (the rotation applied), and wrap_frac (the diagnostic above).
    """
    if method == "circular_mean":
        if not part.any():
            raise ValueError("reference_phase(): part mask is empty")
        ref = float(np.angle(np.mean(np.exp(1j * phase[part]))))
    elif method == "fiducial_roi":
        if fiducial_roi is None:
            raise ValueError("reference_phase(): method='fiducial_roi' needs fiducial_roi")
        y0, y1, x0, x1 = fiducial_roi
        roi = phase[y0:y1, x0:x1]
        if roi.size == 0:
            raise ValueError(
                f"fiducial_roi {fiducial_roi} is empty -- it must be "
                "(y0, y1, x0, x1) with y0 < y1 and x0 < x1"
            )
        ref = float(np.median(roi))
    else:
        raise ValueError(
            f"method must be 'circular_mean' or 'fiducial_roi', got {method!r}"
        )

    phase_ref = np.angle(np.exp(1j * (phase - ref)))
    print(f"  phase referenced to {method} ({np.degrees(ref):.1f} deg)")

    if part.any():
        near_cut = np.abs(np.abs(np.degrees(phase_ref[part])) - 180.0) < wrap_warn_deg
        wrap_frac = float(near_cut.mean())
    else:
        wrap_frac = 0.0
    if wrap_frac > wrap_warn_frac:
        print(f"  WARNING: {100 * wrap_frac:.1f}% of masked pixels are "
              f"within {wrap_warn_deg:.0f} deg of the +/-180 branch cut "
              "after referencing -- the phase field likely spans more than "
              "a full turn somewhere in the part; phase images/profiles "
              "should not be trusted as-is (consider unwrap_phase=True, or "
              "check that `part` isn't including noise-dominated pixels)")

    return phase_ref, ref, wrap_frac


def unwrap_phase_field(phase, part):
    """
    2D spatial phase unwrapping (skimage.restoration.unwrap_phase) over the
    masked region, for when reference_phase()'s wrap-fraction warning fires
    and a genuine multi-turn phase spread remains after referencing.

    reference_phase() only rotates WHERE the branch cut sits; it cannot
    remove a real wrap, because a real wrap is the phase actually crossing
    a full 2*pi somewhere in the field (see reference_phase()'s docstring
    on why phase lag can genuinely grow past +-180 deg far enough from the
    excitation source).  This instead integrates the phase gradient across
    the masked region and adds back whichever multiple of 2*pi makes the
    result spatially continuous, the standard 2D unwrapping approach.

    Only `part` pixels are unwrapped -- passed to skimage as a masked
    array so background noise phase (already established, in part_mask()
    and reference_phase()'s own docstrings, to carry no reliable
    coherent-frequency signal) can't corrupt the unwrapping of the part
    itself.  Pixels outside `part` are returned unchanged (still wrapped,
    and not meant to be read).

    Returns an array the same shape as `phase`, in radians -- values
    inside `part` are NOT wrapped to (-pi, pi]; that's the point, an
    unwrapped field is exactly the one place a value outside that range is
    meaningful rather than a bug. Both the phase display panel and the
    per-line profile extraction downstream can use this array directly in
    place of the still-wrapped `phase`.
    """
    try:
        from skimage.restoration import unwrap_phase
    except ImportError:
        raise ImportError(
            "unwrap_phase_field() needs scikit-image "
            "(pip install scikit-image)"
        ) from None

    masked = np.ma.masked_array(phase, mask=~part)
    unwrapped = unwrap_phase(masked)
    out = phase.copy()
    out[part] = np.ma.getdata(unwrapped)[part]
    return out


def _tls_axis(pts, w):
    """
    Weighted total-least-squares axis through a point cloud.

    Returns (centroid, unit direction).  Total least squares -- the major
    axis of the weighted covariance -- treats x and y symmetrically, so it
    doesn't blow up on a near-vertical line the way an ordinary y-on-x
    regression would.
    """
    centroid = np.average(pts, axis=0, weights=w)
    centered = pts - centroid
    cov = (centered * w[:, None]).T @ centered / w.sum()
    eigvals, eigvecs = np.linalg.eigh(cov)
    return centroid, eigvecs[:, np.argmax(eigvals)]


def _hough_peak(pts, w, n_angles, rho_bin_px):
    """
    Strongest straight line through a weighted point cloud, as (theta, rho)
    with rho measured along the normal (cos theta, sin theta) from the origin
    of `pts`.

    This is the step that makes multi-line parts work.  Fitting a single axis
    to all the ridge pixels at once averages the lines together and returns an
    axis that lies on none of them; a Hough accumulator instead lets each line
    vote into its own (theta, rho) bin, so they stay separate and the tallest
    bin is unambiguously ONE line.  Votes are weighted by lock-in amplitude,
    so the winner is the line carrying the most signal -- the most apparent
    one -- rather than merely the one with the most pixels above threshold.
    """
    thetas = np.linspace(0, np.pi, n_angles, endpoint=False)
    rho_max = np.abs(pts).sum(axis=1).max() + rho_bin_px
    n_rho = int(np.ceil(2 * rho_max / rho_bin_px)) + 1

    acc = np.zeros((n_angles, n_rho))
    for j, th in enumerate(thetas):
        rho = pts[:, 0] * np.cos(th) + pts[:, 1] * np.sin(th)
        idx = np.clip(((rho + rho_max) / rho_bin_px).astype(int), 0, n_rho - 1)
        acc[j] = np.bincount(idx, weights=w, minlength=n_rho)

    # Blur along rho only (never across theta -- that axis wraps): a real line
    # is a couple of pixels wide and never perfectly straight, so its votes
    # land in two or three adjacent rho bins and would otherwise be split.
    acc = gaussian_filter(acc, sigma=(0, 1.0))

    j, k = np.unravel_index(np.argmax(acc), acc.shape)
    return thetas[j], k * rho_bin_px - rho_max


def _extract_segment(pts, w, theta, rho, inlier_tol_px, max_gap_px,
                     bin_px=4.0, tail_frac=0.25):
    """
    Turn one Hough peak into a concrete segment: refine the fit on its own
    inliers, then decide where the line starts and stops.

    Endpoints come from binning the inliers along the axis and keeping the
    bins that carry real amplitude.  Because the bins are amplitude-weighted,
    a bin holding a piece of the line outweighs a bin holding a couple of
    stray noise pixels by more than an order of magnitude, so thresholding at
    a fraction of the typical bin trims the noise tails that would otherwise
    stretch the endpoints past the ends of the actual line.

    Bins separated by less than max_gap_px are joined.  Deletion lines run
    busbar to busbar and a stretch of one may legitimately carry no signal --
    a line whose only defect is a single discrete bridge is quiet everywhere
    else -- so a gap is not evidence of two separate lines.  Two distinct
    deletion lines are two distinct cuts and won't share an axis anyway.

    Returns a dict describing the segment, or None if the peak didn't
    resolve into one.
    """
    normal = np.array([np.cos(theta), np.sin(theta)])
    sel = np.abs(pts @ normal - rho) <= inlier_tol_px

    # The accumulator is quantised in theta and rho; two passes of
    # refit-then-reselect pull the line onto its own pixels at full precision.
    for _ in range(2):
        if sel.sum() < 2:
            return None
        centroid, direction = _tls_axis(pts[sel], w[sel])
        normal = np.array([-direction[1], direction[0]])
        rho = float(centroid @ normal)
        sel = np.abs(pts @ normal - rho) <= inlier_tol_px
    if sel.sum() < 2:
        return None

    axis_idx = np.nonzero(sel)[0]                  # everything on this axis
    proj = (pts[axis_idx] - centroid) @ direction
    order = np.argsort(proj)
    axis_idx, proj = axis_idx[order], proj[order]

    n_bins = max(1, int(np.ceil((proj[-1] - proj[0]) / bin_px)))
    b = np.clip(((proj - proj[0]) / bin_px).astype(int), 0, n_bins - 1)
    wb = np.bincount(b, weights=w[axis_idx], minlength=n_bins)
    occupied = wb > 0
    if not occupied.any():
        return None

    strong = np.nonzero(wb >= tail_frac * np.median(wb[occupied]))[0]
    if len(strong) == 0:
        return None
    gap_bins = max(1, int(np.ceil(max_gap_px / bin_px)))
    runs = np.split(strong, np.nonzero(np.diff(strong) > gap_bins)[0] + 1)
    run = max(runs, key=lambda r: wb[r].sum())
    b0, b1 = run[0], run[-1]

    seg_idx = axis_idx[(b >= b0) & (b <= b1)]
    if len(seg_idx) < 2:
        return None
    centroid, direction = _tls_axis(pts[seg_idx], w[seg_idx])
    normal = np.array([-direction[1], direction[0]])
    rho = float(centroid @ normal)
    proj = (pts[seg_idx] - centroid) @ direction
    span = float(proj.max() - proj.min())
    if span <= 0:
        return None

    return {
        "p0": tuple(centroid + direction * proj.min()),
        "p1": tuple(centroid + direction * proj.max()),
        "centroid": centroid,
        "direction": direction,
        "normal": normal,
        "rho": rho,
        "length_px": span,
        "weight": float(w[seg_idx].sum()),
        # Coverage: does the line carry signal all the way along, or is the
        # fit bridging a couple of clumps?  A solid line scores near 1.
        "coverage": len(run) / (b1 - b0 + 1),
        "rms_px": float(np.std(pts[seg_idx] @ normal - rho)),
        "n_px": len(seg_idx),
    }


def find_deletion_lines(amp_hp, part, ridge_percentile=97, min_ridge_px=10,
                        n_angles=720, rho_bin_px=2.0, inlier_tol_px=3.0,
                        max_gap_px=None, min_length_px=30.0,
                        min_coverage=0.35, max_lines=4, min_score_frac=0.02):
    """
    Find every line-like feature in the part, ranked strongest first.

    Takes the brightest pixels of the (masked, high-passed) amplitude image
    -- lines stand out because they're spatially coherent while noise isn't
    -- and repeatedly pulls out the strongest straight line through them:
    Hough peak, refit on its own inliers, trim to where the signal actually
    is, then erase that axis and look again.  Erasing before the next search
    is what keeps the lines separate; without it every round would return
    the same feature.

    Each candidate is a dict with endpoints ``p0`` / ``p1``, ``length_px``,
    ``weight`` (summed lock-in amplitude along it), ``coverage``,
    ``rms_px`` (straightness), ``n_px``, and ``score``.

    Ranking is by score = weight * coverage: integrated amplitude is what
    "most apparent" means physically, and scaling by coverage stops a bright
    compact blob from outranking a long real line just because the fit found
    an axis through it.

    max_gap_px defaults to a fifth of the part's linear size, capped at 60 px
    -- gaps are measured against the part, not against a flat pixel count
    that means different things at different standoffs, but the proportional
    formula alone still grows without bound on a large part.  Uncapped, that
    let a big part with several closely-spaced, near-parallel deletion lines
    bridge clean across the gap to an unrelated neighbouring line whenever
    two lines happened to fall in nearby Hough bins -- producing endpoints
    that zigzag across many lines instead of following one.  On a part like
    that, pass max_gap_px explicitly (tighter) or use pick_line_endpoints()
    to click the two ends of one line directly.  Candidates shorter than
    min_length_px, patchier than min_coverage, or below min_score_frac of
    the winner are dropped: those are clumps and noise, not lines.
    """
    ridge = part & (amp_hp > np.percentile(amp_hp[part], ridge_percentile))
    ys, xs = np.nonzero(ridge)
    if len(xs) < min_ridge_px:
        raise ValueError(
            f"only {len(xs)} ridge pixels found (need >= {min_ridge_px}) -- "
            "lower ridge_percentile, check the part mask, or pick the line "
            "by hand with pick_line_endpoints()"
        )
    if max_gap_px is None:
        max_gap_px = min(60.0, max(20.0, 0.2 * np.sqrt(part.sum())))

    pts = np.stack([xs, ys], axis=1).astype(float)
    origin = pts.mean(axis=0)          # keeps rho small and symmetric
    pts -= origin

    w = np.clip(amp_hp[ys, xs], 0, None).astype(float)
    if w.sum() <= 0:
        w = np.ones(len(pts))

    live = np.ones(len(pts), dtype=bool)
    lines = []
    for _ in range(max_lines):
        if live.sum() < min_ridge_px:
            break
        pool = np.nonzero(live)[0]
        theta, rho = _hough_peak(pts[pool], w[pool], n_angles, rho_bin_px)
        seg = _extract_segment(pts[pool], w[pool], theta, rho,
                               inlier_tol_px, max_gap_px)
        if seg is None:
            break

        # Claim the whole axis, not just the segment: anything left lying on
        # it is part of the same cut, and leaving it in the pool would only
        # let the next round rediscover this line in pieces.  A parallel
        # neighbour sits at a different rho and survives untouched.
        live[pool[np.abs(pts[pool] @ seg["normal"] - seg["rho"])
                  <= 2 * inlier_tol_px]] = False

        if seg["length_px"] < min_length_px or seg["coverage"] < min_coverage:
            continue
        seg["score"] = seg["weight"] * seg["coverage"]
        seg["p0"] = tuple(np.asarray(seg["p0"]) + origin)
        seg["p1"] = tuple(np.asarray(seg["p1"]) + origin)
        lines.append(seg)

    lines.sort(key=lambda s: s["score"], reverse=True)
    if lines:
        floor = min_score_frac * lines[0]["score"]
        lines = [s for s in lines if s["score"] >= floor]
    return lines


def find_deletion_line(amp_hp, part, verbose=True, return_all=False, **kwargs):
    """
    Auto-detect the ONE most apparent deletion line inside the part and
    return its endpoints as ((x0, y0), (x1, y1)).

    Standalone utility, not used by analyse()'s default pipeline (which
    gets its line geometry from interactive_setup() or a saved
    roi_config.json instead -- see analyse()'s docstring).  Still useful on
    its own, or to seed a roi_config by hand: build
    {"lines": [[list(p0), list(p1)]], ...} from the endpoints this returns.

    Ranks every line in the part with find_deletion_lines() and returns the
    strongest.  Parts with several deletion lines are handled: the others
    are found and reported, not averaged into the answer.

    The margin over the runner-up is printed.  If it's slim, two lines are
    comparably bright and "most apparent" is a coin toss -- pick manually
    with pick_line_endpoints() instead.

    With return_all=True, also returns the full ranked list (each entry as
    returned by find_deletion_lines()) -- e.g. for auto_fiducial_roi(), which
    needs every line's location, not just the winner's.
    """
    lines = find_deletion_lines(amp_hp, part, **kwargs)
    if not lines:
        raise ValueError(
            "no line-like feature found -- lower ridge_percentile, relax "
            "min_length_px / min_coverage, check the part mask, or pick the "
            "line by hand with pick_line_endpoints()"
        )

    best = lines[0]
    if verbose:
        print(f"  found {len(lines)} line-like feature(s), "
              "ranked by integrated amplitude x coverage:")
        for i, s in enumerate(lines):
            (x0, y0), (x1, y1) = s["p0"], s["p1"]
            mark = "<- selected" if i == 0 else ""
            print(f"    [{i}] ({x0:4.0f},{y0:4.0f}) -> ({x1:4.0f},{y1:4.0f})  "
                  f"len {s['length_px']:5.0f} px  score {s['score']:9.3g}  "
                  f"coverage {s['coverage']:.2f}  rms {s['rms_px']:.1f} px  {mark}")
        if len(lines) > 1:
            ratio = best["score"] / lines[1]["score"]
            print(f"    selected line is {ratio:.2f}x the runner-up"
                  + ("   -- AMBIGUOUS, pick the line by hand with "
                     "pick_line_endpoints() instead"
                     if ratio < 1.3 else ""))

    result = (best["p0"], best["p1"])
    return (result, lines) if return_all else result


def _rasterize_segment(mask, p0, p1):
    """Set mask=True along the pixels of segment p0->p1 (both (x, y))."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.hypot(*(p1 - p0)))) + 1)
    xs = np.round(np.linspace(p0[0], p1[0], n)).astype(int)
    ys = np.round(np.linspace(p0[1], p1[1], n)).astype(int)
    h, w = mask.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    mask[ys[valid], xs[valid]] = True


def auto_fiducial_roi(part, lines, exclusion_px=15.0, edge_clear_px=None,
                      box_frac=0.3, max_half_px=None):
    """
    Automatically place the fiducial ROI in the quiet zone farthest from
    every detected deletion line AND from the part's own boundary: the gap
    between two lines when there are two or more, or simply the part's most
    interior point when there's only one (or find_deletion_lines found
    none).

    `exclusion_px` should be at least the thermal diffusion length (the same
    sigma used for spatial_highpass) -- the area right around a line is
    already contaminated by its own thermal spread, so the site must clear
    that halo, not just the bare line pixels.

    `edge_clear_px` applies the same idea to the part's own boundary.  A
    scalloped edge or a busbar is exactly where motion artefacts and
    emissivity variation are worst, so "farthest from every line" alone
    isn't enough if the winning point happens to sit right at the rim -- the
    site has to clear both.  This does NOT default to exclusion_px: a
    busbar or scalloped region is a physically different, typically much
    bigger feature than the thermal diffusion length exclusion_px is sized
    to, and at a coarse mm_per_px that length can be a couple of pixels --
    nowhere near enough edge clearance.  It defaults instead to 8% of the
    part's own smaller dimension, so it stays meaningful regardless of how
    small exclusion_px happens to be.

    The ROI is a small sample of that quiet zone, not the whole zone: on a
    large part the farthest-from-every-line point can be hundreds of pixels
    out, and growing the box to match would pull in unrelated structure.
    max_half_px defaults to twice exclusion_px (floored the same way, at 2%
    of the part's smaller dimension), so the box scales with a physical
    length rather than a fixed pixel count that means different things at
    different standoffs/resolutions, and doesn't collapse to a sliver
    alongside a tiny exclusion_px either.

    Returns (y0, y1, x0, x1).  Raises ValueError if the part mask is empty
    or no site inside it clears every line and the part edge -- in either
    case, pass fiducial_roi explicitly instead.
    """
    if not part.any():
        raise ValueError("part mask is empty -- cannot auto-place a fiducial ROI")
    h, w = part.shape
    if max_half_px is None:
        max_half_px = max(6, int(round(2 * exclusion_px)), int(round(0.02 * min(h, w))))
    if edge_clear_px is None:
        edge_clear_px = max(exclusion_px, int(round(0.08 * min(h, w))))

    # Shrink the searchable area inward by edge_clear_px, the same way a
    # line's own exclusion zone is built below -- skip if this `part` array
    # has no real background pixel at all (a synthetic all-True mask, or a
    # tight crop with no edge in frame): there's no boundary to clear.
    safe_part = part
    edge_active = edge_clear_px > 0 and (~part).any()
    if edge_active:
        eroded = binary_erosion(part, iterations=int(round(edge_clear_px)))
        if eroded.any():
            safe_part = eroded
        else:
            edge_active = False        # part too small to clear it at all

    search = safe_part
    if lines:
        line_mask = np.zeros(part.shape, dtype=bool)
        for ln in lines:
            _rasterize_segment(line_mask, ln["p0"], ln["p1"])
        if exclusion_px > 0:
            line_mask = binary_dilation(line_mask, iterations=int(round(exclusion_px)))
        dist = distance_transform_edt(~line_mask)

        # A single line has zero extent along its own axis, so its endpoint
        # bbox is a sliver -- "between lines" only means something with two
        # or more, each contributing a genuinely different position. There,
        # restrict the search to the box the lines actually flank: otherwise
        # a corner beyond every line's tip can out-score the real gap just
        # by being far from both endpoints, without lying between them.
        if len(lines) >= 2:
            xs = np.concatenate([[ln["p0"][0], ln["p1"][0]] for ln in lines])
            ys = np.concatenate([[ln["p0"][1], ln["p1"][1]] for ln in lines])
            x0b, x1b = max(0, int(xs.min())), min(w, int(np.ceil(xs.max())) + 1)
            y0b, y1b = max(0, int(ys.min())), min(h, int(np.ceil(ys.max())) + 1)
            bbox = np.zeros(part.shape, dtype=bool)
            bbox[y0b:y1b, x0b:x1b] = True
            if (search & bbox).any():
                search = search & bbox
    else:
        dist = distance_transform_edt(safe_part)      # most interior point

    dist = np.where(search, dist, -1.0)
    best_dist = dist.max()
    if best_dist <= 0:
        raise ValueError(
            "no site inside the part clears the deletion line(s) and the "
            "part edge -- pass fiducial_roi explicitly"
        )

    # The best score is often a plateau (e.g. anywhere between two parallel
    # lines is equally far from both) -- argmax alone would tie-break to
    # whichever pixel comes first in scan order, pinning the ROI to one edge
    # of the gap.  Average the whole near-max plateau instead so it lands
    # centred.
    ys_p, xs_p = np.nonzero(dist >= best_dist - 1.0)
    y, x = int(round(ys_p.mean())), int(round(xs_p.mean()))

    # When there's a real edge to clear, the centre point is guaranteed at
    # least edge_clear_px from it (it came from safe_part), but nothing
    # above bounds the box size -- cap the half-width at edge_clear_px too,
    # or a large best_dist could still grow the box back out across that
    # margin toward the edge.
    caps = [max_half_px, int(round(box_frac * best_dist))]
    if edge_active:
        caps.append(edge_clear_px)
    half = max(1, min(caps))
    return max(0, y - half), min(h, y + half), max(0, x - half), min(w, x + half)


def pick_line_endpoints(amp):
    """
    Interactively click both ends of ONE deletion line on the raw amplitude
    image.  Returns ((x0, y0), (x1, y1)), ready to pass as line_endpoints to
    analyse().

    find_deletion_line()'s Hough fit assumes the part's deletion lines are
    each a single straight, well-separated ridge; on a skewed part with
    several lines running close together at an angle, it can bridge across
    the gap to a neighbouring line instead of following the one you meant --
    the along-line profile then shows periodic spikes from crossing every
    line on the part rather than one clean profile.  A human eye picking the
    two endpoints directly sidesteps that entirely.  Requires an interactive
    matplotlib backend (this will not work with Agg / headless).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(amp)
    ax.set_title("click both ends of ONE deletion line")
    pts = fig.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise ValueError(f"need exactly two clicks to define a line, got {len(pts)}")
    p0, p1 = pts
    print(f"  picked line: ({p0[0]:.0f}, {p0[1]:.0f}) -> ({p1[0]:.0f}, {p1[1]:.0f})")
    return p0, p1


def cross_line_profile(image, p0, p1, half_width_px=6, n_samples=400):
    """
    Average the image along a deletion line and return the profile
    perpendicular to it.

    Averaging along the line is where most of your sensitivity comes from: the
    defect is coherent along hundreds of pixels while noise is not, so this
    buys another factor of sqrt(N_along) in SNR on top of the temporal lock-in.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    along = p1 - p0
    length = np.hypot(*along)
    unit = along / length
    normal = np.array([-unit[1], unit[0]])

    offsets = np.linspace(-half_width_px, half_width_px, 2 * half_width_px + 1)
    steps = np.linspace(0, length, n_samples)

    profile = np.zeros_like(offsets)
    for i, off in enumerate(offsets):
        pts = p0[None, :] + steps[:, None] * unit[None, :] + off * normal[None, :]
        yy = np.clip(np.round(pts[:, 1]).astype(int), 0, image.shape[0] - 1)
        xx = np.clip(np.round(pts[:, 0]).astype(int), 0, image.shape[1] - 1)
        profile[i] = np.mean(image[yy, xx])
    return offsets, profile


def along_line_profile(image, p0, p1, band_px=4, n_samples=300):
    """
    Integrate the image across a narrow band centred on the line, as a function
    of position ALONG the line.

    This is the diagnostic profile.  Its SHAPE tells you which failure mode you
    have -- see the notes at the bottom of this file.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    along = p1 - p0
    length = np.hypot(*along)
    unit = along / length
    normal = np.array([-unit[1], unit[0]])

    steps = np.linspace(0, length, n_samples)
    offsets = np.arange(-band_px, band_px + 1)

    values = np.zeros_like(steps)
    for i, s in enumerate(steps):
        pts = p0[None, :] + s * unit[None, :] + offsets[:, None] * normal[None, :]
        yy = np.clip(np.round(pts[:, 1]).astype(int), 0, image.shape[0] - 1)
        xx = np.clip(np.round(pts[:, 0]).astype(int), 0, image.shape[1] - 1)
        values[i] = np.sum(image[yy, xx])
    return steps, values


def _line_angle_deg(p0, p1):
    """Line angle from horizontal, in degrees, folded into [0, 180)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    dx, dy = p1 - p0
    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)


def check_line_angle(p0, p1, bin_px, critical_tol_deg=3.0):
    """
    Warn if a line's angle is too close to axis-aligned (0 or 90 deg) or to
    45 deg for the sub-pixel gain in slanted_edge_profile() to do much.

    The gain comes from phase diversity: a tilted line crosses the pixel
    grid at a different sub-pixel offset at every position along it, so
    pooling hundreds of those crossings reconstructs the cross-line profile
    finer than one pixel -- the same principle a slanted-edge MTF
    measurement uses.  At exactly 0 or 90 degrees that diversity vanishes:
    moving along an exactly horizontal line never changes which row you're
    sampling, so every along-line position reads the identical sub-pixel
    row phase and pooling only reduces noise, not resolution.  Near 45
    degrees the along-line steps advance both axes in lockstep, which is a
    known degenerate case for the same kind of phase-diversity argument in
    slant-edge imaging practice (poor, aliased phase coverage rather than
    the rich, evenly-spread coverage a shallow tilt gives).

    This never raises -- it prints a warning and an estimate of the bin
    size actually achievable (a geometric estimate, not a precise bound:
    roughly the reciprocal of however many distinct row/column crossings
    the line's own pixel extent provides) and lets the caller continue with
    whatever bin_px was requested.

    Returns (angle_deg, warned: bool).
    """
    angle_deg = _line_angle_deg(p0, p1)
    reduced = angle_deg % 90.0
    dist_to_critical = min(reduced, 90.0 - reduced, abs(reduced - 45.0))
    if dist_to_critical > critical_tol_deg:
        return angle_deg, False

    dx, dy = np.asarray(p1, float) - np.asarray(p0, float)
    n_phases = max(min(abs(dx), abs(dy)), 1e-6)
    effective_bin_px = max(bin_px, 1.0 / n_phases)
    near = ("axis-aligned" if min(reduced, 90.0 - reduced) <= critical_tol_deg
           else "45 degrees")
    print(f"  WARNING: line angle {angle_deg:.1f} deg is within "
          f"{critical_tol_deg:.0f} deg of {near} -- sub-pixel oversampling "
          f"will be largely ineffective for this line.  Requested bin_px="
          f"{bin_px:.2g}, estimated achievable bin size ~{effective_bin_px:.2g} px "
          "-- continuing anyway.")
    return angle_deg, True


def slanted_edge_profile(image, p0, p1, half_width_px, bin_px=0.2, n_along=800):
    """
    Sub-pixel cross-line profile via the slanted-edge method.

    cross_line_profile() rounds every sample to its nearest pixel before
    averaging -- fine when the feature is many pixels wide, but the heat
    source here sits at the coating plane and is imaged through the full
    glass ply, so it's blurred by at least the ply thickness (millimetres)
    before it ever reaches the surface.  At typical mm/px sampling that
    diffusion-blurred feature can be sub-pixel, and rounding every sample to
    its nearest pixel turns that into single-pixel spiking artefacts that
    look like signal but are really just quantisation noise.

    Because the line is tilted relative to the pixel grid, each position
    along it crosses the perpendicular direction at a slightly different
    sub-pixel offset.  This samples a dense (n_along x n_bins) grid --
    n_along positions along the line, perpendicular offsets from
    -half_width_px to +half_width_px in bin_px steps -- with CONTINUOUS
    bilinear interpolation (scipy.ndimage.map_coordinates, order=1, no
    rounding anywhere in the sampling path), then averages along the line
    at each offset.  Pooling that many along-line crossings recovers
    genuinely finer effective sampling across the line than the camera's
    own pixel pitch -- see check_line_angle() for the one case (a line too
    close to axis-aligned or 45 degrees) where this doesn't work.

    Returns (offsets, profile): offsets is the symmetric, bin_px-spaced
    grid of perpendicular distances from the line (px), profile is the
    along-line average at each.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    along = p1 - p0
    length = np.hypot(*along)
    unit = along / length
    normal = np.array([-unit[1], unit[0]])

    n_half = max(1, int(round(half_width_px / bin_px)))
    offsets = np.arange(-n_half, n_half + 1) * bin_px
    steps = np.linspace(0, length, n_along)

    # (n_bins, n_along) grid of continuous (x, y) image coordinates.
    base = p0[None, :] + steps[:, None] * unit[None, :]           # (n_along, 2)
    pts = base[None, :, :] + offsets[:, None, None] * normal[None, None, :]
    xx = pts[:, :, 0]
    yy = pts[:, :, 1]

    sampled = map_coordinates(image, [yy, xx], order=1, mode="nearest")
    profile = sampled.mean(axis=1)
    return offsets, profile


# ----------------------------------------------------------------------------
# 6. GEOMETRY SELECTION (interactive + persisted JSON config)
# ----------------------------------------------------------------------------

def interactive_setup(image, phase=None, n_lines=1, calibrate=False,
                      mask_roi=False, wrap_reject_deg=20.0):
    """
    One click-through setup session on a displayed frame -- typically the
    raw lock-in amplitude image, or a single representative raw frame if
    amplitude isn't computed yet -- that defines every piece of geometry
    this pipeline needs by hand.  Used by analyse() when auto_geometry=False
    (the default -- see analyse()'s docstring for why manual is preferred
    here): the alternative to auto_setup() for a part where the true signal
    is a subtle symmetric bump riding on top of a much bigger, genuine
    zone-to-zone step, and auto_setup()'s detector -- built around "big and
    sharp" -- has locked onto the step instead (check
    lockin_line_candidates.png to tell).

    Walks through, in order:
      1. n_lines deletion lines, each as two endpoint clicks.  Each line's
         angle from horizontal is computed and printed immediately (needed
         because slanted_edge_profile()'s sub-pixel gain depends on it --
         see check_line_angle()), and you're asked whether it's a
         "reference" line: one believed sound, used in the summary table to
         put every other line's signal in context as a ratio rather than a
         bare number.
      2. one fiducial/phase-reference ROI, as two opposite-corner clicks.
         Put this in clean, mid-zone coating -- well away from every line
         AND from the part edge.  Both are bad reference sites: a line
         carries the zone step this whole feature exists to separate out,
         and an edge is where motion artefacts and emissivity variation are
         worst (see auto_fiducial_roi()'s docstring -- analyse() places the
         ROI this way automatically when auto_geometry=True; click it by
         hand here otherwise).  If `phase` was passed, the pick is also
         checked against the raw (unreferenced) phase in that ROI: a
         candidate whose median phase sits within wrap_reject_deg of the
         +/-180 branch cut is REJECTED with an explanation printed, and you
         are prompted to click again -- referencing to (or near) a site
         sitting on the branch cut is exactly what previously put the cut
         through the middle of the part (see reference_phase()'s
         docstring). This check is about the ROI SITE, independent of
         which reference method analyse() ends up using.
      3. if calibrate=True, two points spanning a KNOWN physical distance,
         prompted for at the console, to compute mm_per_px directly instead
         of hard-coding it.
      4. if mask_roi=True, a polygon (three or more clicks, Enter to
         finish, right-click to undo the last point) traced around the
         part outline by hand -- the manual fallback for part_mask() when
         its automatic Otsu polarity detection picks the wrong region.
         Stored in the returned config and reused headlessly by
         part_mask() on every later run of the same saved config, with no
         need to re-click.

    `phase` (radians, same shape as `image`) is optional -- when given, it's
    shown in its own panel alongside `image` (percentile-scaled, same
    "twilight" cyclic colormap analyse() uses for its own phase panel) so a
    wrap boundary is visible before you click, not just discovered after
    the fact from the rejection message.  Every click still targets the
    SAME pixel coordinates regardless of which panel it lands in -- both
    panels show the same (h, w) array, just two different views of it.

    Each selection is confirmed visually (crosshairs for a line or the
    calibration pair, a rectangle outline for the ROI, a closed polygon for
    the mask ROI) before moving to the next, on the same persistent figure.
    Requires an interactive matplotlib backend -- this will not work with
    Agg / headless.  Returns a plain dict; see save_roi_config() /
    load_roi_config() to persist it and skip clicking on the next run of
    the same recording.  Each line is stored as {"p0": [x, y], "p1":
    [x, y], "angle_deg": ..., "is_reference": bool} -- analyse_deletion_line()
    recomputes angle_deg itself from p0/p1 at analysis time too, so the
    stored value is for the record, not load-bearing.
    """
    if phase is not None:
        fig, (ax, axp) = plt.subplots(1, 2, figsize=(18, 9))
        lo_p, hi_p = np.nanpercentile(np.degrees(phase), [2, 98])
        axp.imshow(np.degrees(phase), cmap="twilight", vmin=lo_p, vmax=hi_p)
        axp.set_title("phase [deg] (raw, unreferenced) -- for reference only, "
                      "click on the LEFT image")
    else:
        fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(image)

    lines = []
    for i in range(n_lines):
        ax.set_title(f"line {i + 1}/{n_lines}: click both endpoints")
        fig.canvas.draw()
        pts = fig.ginput(2, timeout=0)
        if len(pts) != 2:
            plt.close(fig)
            raise ValueError(f"line {i + 1}: need exactly two clicks, got {len(pts)}")
        p0, p1 = pts
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "r-", marker="+", ms=12, mew=2)
        fig.canvas.draw()
        angle_deg = _line_angle_deg(p0, p1)
        print(f"  line {i + 1}: ({p0[0]:.0f}, {p0[1]:.0f}) -> ({p1[0]:.0f}, {p1[1]:.0f})"
              f"  ({angle_deg:.1f} deg from horizontal)")
        is_reference = input(f"  line {i + 1}: reference line (believed sound)? "
                             "[y/N]: ").strip().lower().startswith("y")
        lines.append({"p0": list(p0), "p1": list(p1), "angle_deg": angle_deg,
                     "is_reference": is_reference})

    base_title = ("fiducial ROI: click two opposite corners\n"
                 "(clean mid-zone coating -- away from every line and the edge)")
    ax.set_title(base_title)
    while True:
        fig.canvas.draw()
        pts = fig.ginput(2, timeout=0)
        if len(pts) != 2:
            plt.close(fig)
            raise ValueError(f"fiducial ROI: need exactly two clicks, got {len(pts)}")
        (rx0, ry0), (rx1, ry1) = pts
        x0, x1 = sorted((rx0, rx1))
        y0, y1 = sorted((ry0, ry1))
        fiducial_roi = [int(round(y0)), int(round(y1)), int(round(x0)), int(round(x1))]

        if phase is not None:
            roi_deg = np.degrees(phase[fiducial_roi[0]:fiducial_roi[1],
                                       fiducial_roi[2]:fiducial_roi[3]])
            if roi_deg.size:
                median_deg = float(np.median(roi_deg))
                dist_to_cut = 180.0 - abs(median_deg)   # distance to the
                                                         # nearer of +/-180
                if dist_to_cut < wrap_reject_deg:
                    print(f"  REJECTED: this ROI's median raw phase "
                          f"({median_deg:.1f} deg) is within "
                          f"{wrap_reject_deg:.0f} deg of the +/-180 wrap "
                          "boundary -- referencing to (or near) this site "
                          "would put the branch cut through the middle of "
                          "the part (see reference_phase()'s docstring). "
                          "Click a different spot.")
                    continue

        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="lime", lw=1.5))
        fig.canvas.draw()
        break
    print(f"  fiducial ROI: y[{fiducial_roi[0]}:{fiducial_roi[1]}] "
          f"x[{fiducial_roi[2]}:{fiducial_roi[3]}]")

    mm_per_px = None
    if calibrate:
        ax.set_title("calibration: click two points spanning a KNOWN distance")
        fig.canvas.draw()
        pts = fig.ginput(2, timeout=0)
        if len(pts) != 2:
            plt.close(fig)
            raise ValueError(f"calibration: need exactly two clicks, got {len(pts)}")
        c0, c1 = pts
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], "c-", marker="+", ms=12, mew=2)
        fig.canvas.draw()
        px_dist = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))
        mm_dist = float(input(f"  distance between those two calibration points, in mm: "))
        mm_per_px = mm_dist / px_dist
        print(f"  calibration: {px_dist:.1f} px = {mm_dist:.2f} mm "
              f"-> {mm_per_px:.4f} mm/px")

    mask_roi_pts = None
    if mask_roi:
        ax.set_title("part mask ROI: click a polygon around the part outline\n"
                     "(3+ points, Enter to finish, right-click to undo)")
        fig.canvas.draw()
        pts = fig.ginput(-1, timeout=0)
        if len(pts) < 3:
            plt.close(fig)
            raise ValueError(f"mask ROI: need at least 3 clicks for a "
                            f"polygon, got {len(pts)}")
        ax.add_patch(Polygon(pts, closed=True, fill=False,
                             edgecolor="yellow", lw=1.5))
        fig.canvas.draw()
        mask_roi_pts = [list(p) for p in pts]
        print(f"  mask ROI polygon: {len(pts)} vertices")

    plt.close(fig)
    return {"lines": lines, "fiducial_roi": fiducial_roi, "mm_per_px": mm_per_px,
            "mask_roi": mask_roi_pts}


def auto_setup(amp, f_excite, mm_per_px, n_lines=1, mask_polarity="auto",
               mask_close_px=3, highpass_mu_factor=2.5, **find_kwargs):
    """
    Automatic alternative to interactive_setup(): detects deletion lines and
    places the fiducial ROI without clicking, so a full run (including
    geometry) can go headless end to end.  This is what analyse() uses by
    default (auto_geometry=True).

    find_deletion_lines() is built around "line is bright, sharp, and
    spatially coherent" -- exactly the trait a genuine zone-to-zone
    power-density step also has, so on a part where two or more candidates
    score close together the automatic pick can land on a step-adjacent
    artefact rather than the real deletion line a human eye would pick
    immediately (see find_deletion_lines()'s own docstring on this).
    ALWAYS check lockin_line_candidates.png after a run (see analyse(),
    auto_geometry=True), which shows every candidate considered, not just
    the winner(s) -- if the printed margin over the runner-up is slim, or
    the plot shows the wrong pick, re-run with auto_geometry=False and a
    manually clicked config (interactive_setup()) instead.

    Builds a preliminary part mask via part_mask() -- without the
    fiducial_roi / line_endpoints hints, since those come FROM this
    function and can't be used to inform it -- spatially high-passes the
    amplitude image at highpass_mu_factor times the thermal diffusion
    length at f_excite (see spatial_highpass()'s docstring for why that
    factor range isolates line-like features), and runs
    find_deletion_lines() / auto_fiducial_roi() on the result.  The final,
    hint-informed part mask actually used for analysis is recomputed later
    in analyse() once this geometry is known, so a rough mask here is fine.

    mm_per_px is required -- there's no calibration click in this path, and
    it's needed anyway to convert the diffusion length to pixels.

    **find_kwargs are passed through to find_deletion_lines() (e.g.
    ridge_percentile, min_length_px, min_coverage) to tune detection on an
    unusual part.

    Returns (config, candidates, n_selected): config is shaped like
    interactive_setup()'s return (every line untagged as a reference --
    there's no human judgement call here to make that call; edit
    roi_config.json by hand afterwards if you want one tagged).
    candidates is the FULL ranked list find_deletion_lines() produced,
    strongest first; the selected lines are candidates[:n_selected]. Both
    are what lockin_line_candidates.png visualises.
    """
    part = part_mask(amp, mask_polarity=mask_polarity, close_px=mask_close_px)

    alpha = 5e-7                                        # m^2/s, glass
    mu_mm = np.sqrt(alpha / (np.pi * f_excite)) * 1000
    sigma_px = highpass_mu_factor * mu_mm / mm_per_px
    amp_hp = spatial_highpass(amp, sigma_px)

    find_kwargs.setdefault("max_lines", max(n_lines, 4))
    candidates = find_deletion_lines(amp_hp, part, **find_kwargs)
    if not candidates:
        raise ValueError(
            "auto_setup(): no line-like feature found -- fall back to "
            "interactive_setup() / pick_line_endpoints()"
        )

    chosen = candidates[:n_lines]
    if len(chosen) < n_lines:
        print(f"  WARNING: auto_setup() only found {len(chosen)} line(s), "
              f"fewer than the requested n_lines={n_lines}")

    lines = []
    for seg in chosen:
        p0, p1 = seg["p0"], seg["p1"]
        angle_deg = _line_angle_deg(p0, p1)
        lines.append({"p0": list(p0), "p1": list(p1), "angle_deg": angle_deg,
                     "is_reference": False})
        print(f"  auto-detected line: ({p0[0]:.0f}, {p0[1]:.0f}) -> "
              f"({p1[0]:.0f}, {p1[1]:.0f})  ({angle_deg:.1f} deg)  "
              f"score {seg['score']:.3g}  coverage {seg['coverage']:.2f}")

    fiducial_roi = list(auto_fiducial_roi(part, chosen))
    print(f"  auto-placed fiducial ROI: y[{fiducial_roi[0]}:{fiducial_roi[1]}] "
          f"x[{fiducial_roi[2]}:{fiducial_roi[3]}]")

    config = {"lines": lines, "fiducial_roi": fiducial_roi,
             "mm_per_px": mm_per_px, "mask_roi": None}
    return config, candidates, len(chosen)


def plot_line_candidates(amp, part, candidates, n_selected, fiducial_roi,
                         path="lockin_line_candidates.png"):
    """
    Diagnostic for auto_geometry=True: every line-like feature
    find_deletion_lines() found, ranked strongest first, drawn as a mostly
    transparent overlay on the masked amplitude image so a human can
    confirm at a glance that the automatic pick(s) -- solid, bright green --
    landed on the intended line(s) rather than a close runner-up.  Faint,
    unlabelled runners-up are the point: a wrong pick with a comparable
    score should stay visible here, not disappear.

    Kept as its own separate figure rather than added to lockin_images.png,
    which deliberately omits line overlays even for a single manually
    clicked line (see analyse()'s "rendering diagnostic images" stage) --
    that decision was about decluttering the routine per-run output, not
    about candidate review, which only exists on the auto-detection path
    and is exactly the moment an overlay earns its clutter.
    """
    disp = np.where(part, amp, np.nan)
    lo, hi = np.nanpercentile(disp, [2, 98])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(disp, vmin=lo, vmax=hi)

    for i, seg in enumerate(candidates):
        (x0, y0), (x1, y1) = seg["p0"], seg["p1"]
        selected = i < n_selected
        ax.plot([x0, x1], [y0, y1],
               color="lime" if selected else "red",
               lw=2.5 if selected else 1.5,
               alpha=0.9 if selected else 0.25,
               solid_capstyle="round")
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.annotate(f"{i}" + (" selected" if selected else ""),
                   (mx, my), color="white", fontsize=8,
                   ha="center", va="center",
                   alpha=1.0 if selected else 0.5)

    fy0, fy1, fx0, fx1 = fiducial_roi
    ax.add_patch(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0,
                           fill=False, edgecolor="cyan", lw=1.5))

    ax.set_title(f"auto-detected line candidates: {len(candidates)} found, "
                f"{n_selected} selected (green, solid)")
    ax.set_xlim(0, part.shape[1]); ax.set_ylim(part.shape[0], 0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def save_roi_config(config, path="roi_config.json"):
    """
    Persist a geometry config (from interactive_setup(), or hand-built with
    the same shape: {"lines": [{"p0": [x, y], "p1": [x, y], "angle_deg":
    ..., "is_reference": bool}, ...], "fiducial_roi": [y0, y1, x0, x1],
    "mm_per_px": ..., "mask_roi": [[x, y], ...] or None}, plus whatever
    processing parameters analyse() has stamped in -- see its docstring) to
    JSON, so the next run of the same recording can load it with
    load_roi_config() instead of clicking again.
    """
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {path}")


def load_roi_config(path="roi_config.json"):
    """Load a geometry config previously written by save_roi_config()."""
    with open(path) as f:
        config = json.load(f)
    print(f"  loaded {path}: {len(config.get('lines', []))} line(s), "
          f"fiducial_roi={config.get('fiducial_roi')}, "
          f"mm_per_px={config.get('mm_per_px')}, "
          f"mask_roi={'set (' + str(len(config['mask_roi'])) + ' vertices)' if config.get('mask_roi') else None}")
    return config


# ----------------------------------------------------------------------------
# 7. SYMMETRIC / ANTISYMMETRIC LINE ANALYSIS
# ----------------------------------------------------------------------------

def symmetric_antisymmetric_profile(image, p0, p1, mu_px, mm_per_px,
                                    half_width_mm=10.0, bin_px=0.2,
                                    n_along=800, wing_factor=3.0,
                                    baseline_frac=0.6, recentre=True,
                                    max_shift_mm=1.5, detrend_profile=True):
    """
    Decompose the sub-pixel cross-line profile of `image` about a deletion
    line into symmetric and antisymmetric parts.

    On this part, adjacent zones legitimately run at different power
    density -- different heater geometry, a graded coating thickness -- so
    the true lock-in amplitude has a genuine STEP at every deletion line.
    A resistive leakage defect is a much smaller, symmetric heat bump on
    top of that step: a real heat source dumps heat into both neighbouring
    zones alike, since heat doesn't know which zone it started in. Taking
        sym  = (right + mirrored_left) / 2   (even about the centre)
        anti = (right - mirrored_left) / 2   (odd about the centre)
    separates the two directly, without ever spatially high-pass filtering
    the step -- which is what turns it into a spurious antisymmetric dipole
    (see spatial_highpass()'s docstring) that would swamp the much smaller
    symmetric signal this function is built to isolate.

    The profile itself comes from slanted_edge_profile(), not
    cross_line_profile(): the heat source sits at the coating plane and is
    imaged through the full glass ply, so it's blurred by at least the ply
    thickness before it reaches the surface, and can be sub-pixel wide at
    typical mm/px sampling.  Nearest-neighbour sampling (cross_line_profile)
    rounds every sample to a whole pixel, which turns that into one-sample
    spiking artefacts; slanted_edge_profile()'s continuous bilinear
    sampling, pooled across many along-line crossings at different sub-pixel
    phases, recovers genuine sub-pixel resolution across the line instead.

    Locating the centre precisely matters: decomposing about the wrong
    point leaks step into sym and vice versa.  With recentre=True
    (default), the line's clicked/given position is treated as only an
    initial estimate, refined by finding where the profile is STEEPEST
    near that estimate (the peak of |d profile / d offset|) and treating
    that as the step's true position.  This is deliberately not simply
    "where the raw profile crosses the midpoint between its two wing
    levels": that reads back the step's OWN antisymmetric structure fine
    in isolation, but a real leakage bump sitting at the same location adds
    a symmetric offset to the raw profile there too, which shifts a level
    crossing away from the true centre.  A symmetric bump's slope is
    exactly zero at its own centre by construction (that's what
    "symmetric about the centre" means), so it contributes nothing to
    where the profile is steepest -- the steepest point stays pinned to the
    step regardless of how big a co-located symmetric signal sits on top of
    it, which is exactly the robustness this needs.  Bilinear interpolation
    is piecewise LINEAR between native pixel columns, so the steepest point
    is usually a short PLATEAU of tied gradient magnitude (one native-pixel
    interval), not a single isolated peak -- taking argmax alone would pick
    whichever end of that plateau happens to come first in scan order,
    biasing the estimate toward one edge instead of its true centre.
    Averaging the whole near-max plateau instead gives the unbiased centre
    (the same fix used for the same reason in auto_fiducial_roi()'s
    tie-breaking).  The line is shifted along its own normal by that
    amount, and the profile is resampled ONCE at the corrected position --
    not iterated to convergence, per the physical picture that a single
    clean step has one well-defined location.

    max_shift_mm (default 1.5) clamps how far that steepest-point search is
    even allowed to look, converted to px via mm_per_px.  A real deletion
    line is 25-50 um wide, so a genuine step's steepest point sits right at
    the clicked/given position; a crossing found several mm away is not the
    line, it's the zone-to-zone background gradient's own antisymmetric
    zero crossing -- a real feature, but a much bigger and differently
    located one that the sym/anti split downstream is not built to handle
    correctly if recentring locks onto it instead of the line.  The search
    window (previously max(3.0, half_width_px / 3) px, unclamped) is
    capped at max_shift_px; if the steepest point found is at or beyond
    that cap, this is a strong signal the search escaped onto the zone
    gradient rather than finding the line, so recentring is abandoned for
    this call -- centre_shift_px/_mm come back 0 and p0/p1 are the
    given/clicked position unchanged, with a warning printed rather than
    silently returning a clamped-but-still-wrong shift.

    detrend_profile (default True) fits a straight line (slope + intercept,
    least squares, in mm) to the two-sided sampled profile using ONLY the
    outer/wing samples on both sides (|offset| > baseline_frac *
    half_width_px, the same fraction used for sym's own baseline below) and
    subtracts it from the whole profile before the sym/anti split.  This
    matters because the background here varies by more than an order of
    magnitude across the part over tens of mm while the feature of interest
    is a couple of mm wide at most: over a +/-half_width_mm window that
    background dominates the raw profile, and even though a perfectly
    linear ramp is pure antisymmetric about its own centre (so in theory it
    can't leak into sym), the window is wide enough that the real
    background isn't perfectly linear either, and imperfect recentring adds
    a small offset between the ramp's true centre and the one used for the
    split -- both let a good chunk of that dominant trend leak into sym as
    a spurious symmetric bump. Removing the wing-fitted straight line first
    (which cannot remove a genuine symmetric feature -- see above) shrinks
    that leakage to whatever the background's actual local curvature is,
    instead of its full linear magnitude.  The fitted slope is returned
    (amplitude units per mm) precisely because a large one is itself a
    warning that half_width_mm is too wide for how fast the local
    background moves.

    mu_px is the thermal diffusion length in pixels (mu_mm / mm_per_px) --
    the physical scale that sets the wing/core boundary (wing_factor *
    mu_px): a genuine buried heat source has a footprint comparable to one
    diffusion length, so anything that far out is background, not signal.
    half_width_mm sets how far the profile extends in each direction (in
    mm, converted here via mm_per_px, deliberately not hard-coded in
    pixels): wide enough for a real wing baseline, narrow enough that
    zone-scale power-density variation elsewhere on the part doesn't
    dominate the fit -- the default (10 mm) is roughly 8 diffusion lengths
    at a typical excitation frequency here, comfortably inside where a
    single zone's own variation should still be small.

    baseline_frac sets where sym's own zero level is measured: the region
    |d| > baseline_frac * half_width_px (default 0.6, i.e. the outer 40% of
    the sampled window).  This is deliberately a fraction of the SAMPLING
    WINDOW (half_width_px), not of wing_factor * mu_px (the diffusion-
    length-based wing used for peak/wing_rms/ratio/anti_step below): those
    two can disagree by a lot -- a large wing_factor or a small half_width_mm
    can shrink wing_factor * mu_px's outer region down to a handful of
    samples, or past the edge of the window entirely, which previously left
    the baseline pinned to one single noisy edge sample and sym visibly not
    settling to zero out in the wings.  Tying the baseline to the window
    itself instead guarantees a robust multi-sample region regardless of
    how wing_factor and mu_px happen to relate to half_width_mm.

    Returns a dict:
      d           -- offsets from the (possibly recentred) line, >= 0, px
      sym, anti   -- the two decomposed profiles, same length as d.  sym has
                     the mean of its own baseline region (see baseline_frac
                     above) subtracted, so it approaches zero out in the
                     wings and the reported peak is a genuine excess over a
                     true baseline, not just relative to the window mean.
      p0, p1      -- the (possibly recentred) line endpoints actually used
      centre_shift_px, centre_shift_mm -- how far recentring moved the line
                     (0 if disabled, too few samples were available near
                     the line, or the search hit the max_shift_mm clamp and
                     fell back to the given position)
      trend_slope -- the fitted background slope removed by detrend_profile
                     (0.0 if disabled or too few outer samples to fit), in
                     `image` units per mm
      baseline_frac, baseline_px, n_baseline_samples -- the baseline region
                     actually used (baseline_px is the (lo, hi) px range
                     |d| was required to exceed) and how many samples fell
                     in it, for reporting.
      peak        -- max of sym within the core region (|d| <= wing_factor
                     * mu_px) -- the candidate leakage signal
      wing_rms    -- RMS of sym in the wings (|d| > wing_factor * mu_px) --
                     the noise floor for that peak
      ratio       -- peak / wing_rms
      anti_step   -- median of anti in the wings -- HALF the total
                     zone-to-zone difference (anti is itself the halved
                     odd part); multiply by 2 for the physical step size.
      fwhm_mm     -- full width at half max of the sym peak (NaN if peak
                     isn't positive), the width check this whole change is
                     built around -- see analyse_deletion_line()'s docstring
                     for how it's used.
    """
    half_width_px = half_width_mm / mm_per_px

    def _sample(a, b):
        return slanted_edge_profile(image, a, b, half_width_px,
                                    bin_px=bin_px, n_along=n_along)

    offs, prof = _sample(p0, p1)
    centre_shift_px = 0.0

    if recentre:
        # Search near the assumed centre only, so a real near-line bump (the
        # signal we're actually after) can't be mistaken for a second step
        # somewhere else in the sampled range -- and never look past
        # max_shift_px, so the search cannot even consider a "steepest
        # point" that's actually the zone gradient's own crossing several mm
        # away (see max_shift_mm above).
        max_shift_px = max_shift_mm / mm_per_px
        search_radius_px = min(max(3.0, half_width_px / 3), max_shift_px)
        near = np.abs(offs) <= search_radius_px
        o_near, p_near = offs[near], prof[near]
        order = np.argsort(o_near)
        o_near, p_near = o_near[order], p_near[order]

        crossing = None
        if len(o_near) >= 3:
            grad = np.gradient(p_near, o_near)
            mag = np.abs(grad)
            max_mag = mag.max()
            tied = mag >= max_mag - 1e-9 * max(1.0, max_mag)
            crossing = float(o_near[tied].mean())

        if crossing is None:
            print("  centre refinement: too few samples near the given line "
                  "-- using the given position as-is")
        elif abs(crossing) >= search_radius_px - 1e-9 and search_radius_px >= max_shift_px - 1e-9:
            print(f"  WARNING: centre refinement's steepest point sits at "
                  f"{crossing * mm_per_px:+.2f} mm, at or beyond the "
                  f"max_shift_mm={max_shift_mm:.2f} search limit -- this is "
                  "almost certainly the zone-to-zone background gradient's "
                  "own crossing, not the line (a real line is typically "
                  "0.025-0.05 mm wide).  Falling back to the given/clicked "
                  "centre with zero shift.")
        elif crossing != 0.0:
            centre_shift_px = crossing
            p0a, p1a = np.asarray(p0, float), np.asarray(p1, float)
            unit = (p1a - p0a) / np.hypot(*(p1a - p0a))
            normal = np.array([-unit[1], unit[0]])
            shift = centre_shift_px * normal
            p0, p1 = tuple(p0a + shift), tuple(p1a + shift)
            offs, prof = _sample(p0, p1)

    centre_shift_mm = centre_shift_px * mm_per_px

    trend_slope = 0.0
    if detrend_profile:
        # Fit on the outer/wing region of the FULL two-sided profile (both
        # sides of the line), in mm so the reported slope means the same
        # thing regardless of bin_px -- see this function's docstring for
        # why subtracting a wing-fitted straight line cannot remove a
        # genuine symmetric feature, only shrink the background's leakage
        # into one down to its actual local curvature.
        outer = np.abs(offs) > baseline_frac * half_width_px
        if outer.sum() >= 2:
            d_mm_all = offs * mm_per_px
            design = np.vstack([d_mm_all[outer], np.ones(int(outer.sum()))]).T
            trend_slope, trend_intercept = np.linalg.lstsq(design, prof[outer], rcond=None)[0]
            trend_slope, trend_intercept = float(trend_slope), float(trend_intercept)
            prof = prof - (trend_slope * d_mm_all + trend_intercept)
        else:
            print("  detrend_profile: too few outer/wing samples to fit a "
                  "background trend -- skipping")

    mid = len(offs) // 2                # offs is the symmetric bin_px grid -H..H
    d = offs[mid:]
    right = prof[mid:]
    mirrored_left = prof[mid::-1]       # prof at offsets 0, -bin_px, -2*bin_px, ...
    sym = (right + mirrored_left) / 2.0
    anti = (right - mirrored_left) / 2.0

    baseline_lo = baseline_frac * half_width_px
    baseline_region = d > baseline_lo
    if not baseline_region.any():
        baseline_region = d >= d.max()     # fall back to the outermost sample(s)
    baseline = float(sym[baseline_region].mean())
    sym = sym - baseline
    n_baseline_samples = int(baseline_region.sum())

    wing = d > wing_factor * mu_px

    # core is d <= threshold and d is sorted ascending from 0, so core is
    # always a PREFIX of the array -- an index into sym[core] is therefore
    # already a valid index into the full sym/d arrays too, which the FWHM
    # search below relies on.
    core = d <= wing_factor * mu_px
    i_peak = int(np.argmax(sym[core])) if core.any() else int(np.argmax(sym))
    peak = float(sym[i_peak])
    wing_rms = float(np.sqrt(np.mean(sym[wing] ** 2))) if wing.any() else float("nan")
    ratio = peak / wing_rms if wing_rms > 0 else float("inf")
    anti_step = float(np.median(anti[wing])) if wing.any() else float(anti[-1])

    # FWHM of the (one-sided) sym peak: the offset, moving outward from the
    # peak, where sym first drops to half its peak value.  The physical,
    # two-sided bump's FWHM is twice that, since sym(d) already represents
    # the shared value at +-d.
    fwhm_mm = float("nan")
    if peak > 0:
        half = peak / 2.0
        below = np.nonzero(sym[i_peak:] <= half)[0]
        if len(below) > 0:
            j = i_peak + below[0]
            if j > 0 and sym[j - 1] != sym[j]:
                frac = (sym[j - 1] - half) / (sym[j - 1] - sym[j])
                d_half = d[j - 1] + frac * (d[j] - d[j - 1])
            else:
                d_half = d[j]
            fwhm_mm = float(2 * d_half * mm_per_px)

    return {
        "d": d, "sym": sym, "anti": anti, "p0": p0, "p1": p1,
        "centre_shift_px": centre_shift_px, "centre_shift_mm": centre_shift_mm,
        "trend_slope": trend_slope,
        "baseline_frac": baseline_frac, "baseline_px": (baseline_lo, float(d.max())),
        "n_baseline_samples": n_baseline_samples,
        "peak": peak, "wing_rms": wing_rms, "ratio": ratio, "anti_step": anti_step,
        "fwhm_mm": fwhm_mm,
    }


def analyse_deletion_line(amp, phase, p0, p1, mu_mm, mm_per_px,
                          ply_thickness_mm=3.0, half_width_mm=10.0,
                          bin_px=0.2, n_along=800, wing_factor=3.0,
                          baseline_frac=0.6, max_shift_mm=1.5,
                          detrend_profile=True):
    """
    Full symmetric/antisymmetric analysis of ONE deletion line.

    Computes and checks the line's angle first (see check_line_angle()):
    the sub-pixel gain depends on it, and it's worth knowing about before
    trusting a fine profile off a near-axis-aligned or near-45-degree line.

    Recentres once on the AMPLITUDE channel (higher SNR on the step than
    phase usually gives) via symmetric_antisymmetric_profile(), then reuses
    that corrected line -- unchanged -- for the phase channel, so the two
    are decomposed about the same physical point and stay directly
    comparable rather than each finding a slightly different centre from
    its own noise.

    Physical plausibility check: the true source sits at the coating plane
    and is imaged through the full glass ply, so any real signature is
    blurred by AT LEAST the ply thickness before it reaches the surface --
    it physically cannot be narrower than that.  Comparing the amplitude
    sym peak's FWHM (see symmetric_antisymmetric_profile()) against
    ply_thickness_mm and the diffusion length mu_mm gives a verdict:
      - narrower than ply_thickness_mm: "implausibly narrow -- likely a
        surface artefact or sampling residue", not a real buried source.
      - much wider than a few diffusion lengths (> 5 * mu_mm here): "likely
        zone-scale structure, not a line defect" -- probably a symmetric
        component of imperfect recentring or a broader thermal gradient,
        not a compact leakage bump.
      - in between: "plausible candidate".

    baseline_frac is passed straight through to symmetric_antisymmetric_profile()
    for both channels -- see its docstring for what it controls.
    max_shift_mm and detrend_profile are also passed straight through --
    the recentring clamp and the wing-fitted background removal -- see
    symmetric_antisymmetric_profile()'s docstring for both.

    Returns {"p0", "p1" (corrected), "mu_px", "angle_deg", "fwhm_mm",
    "verdict", "short_verdict", "amp", "phase"}, where "amp" and "phase"
    are the two symmetric_antisymmetric_profile() result dicts.
    short_verdict is one of "PLAUSIBLE", "TOO NARROW", "TOO WIDE", "NO PEAK"
    -- the same call as `verdict` in one word, for print_line_summary()'s
    table column; `verdict` keeps the full reasoning.
    """
    angle_deg, _ = check_line_angle(p0, p1, bin_px)

    mu_px = mu_mm / mm_per_px
    amp_result = symmetric_antisymmetric_profile(
        amp, p0, p1, mu_px, mm_per_px, half_width_mm=half_width_mm,
        bin_px=bin_px, n_along=n_along, wing_factor=wing_factor,
        baseline_frac=baseline_frac, recentre=True,
        max_shift_mm=max_shift_mm, detrend_profile=detrend_profile)
    phase_result = symmetric_antisymmetric_profile(
        phase, amp_result["p0"], amp_result["p1"], mu_px, mm_per_px,
        half_width_mm=half_width_mm, bin_px=bin_px, n_along=n_along,
        wing_factor=wing_factor, baseline_frac=baseline_frac, recentre=False,
        max_shift_mm=max_shift_mm, detrend_profile=detrend_profile)

    fwhm_mm = amp_result["fwhm_mm"]
    if np.isnan(fwhm_mm):
        short_verdict = "NO PEAK"
        verdict = "no positive peak -- nothing to assess"
    elif fwhm_mm < ply_thickness_mm:
        short_verdict = "TOO NARROW"
        verdict = (f"implausibly narrow ({fwhm_mm:.2f} mm < ply thickness "
                  f"{ply_thickness_mm:.2f} mm) -- likely a surface artefact "
                  "or sampling residue")
    elif fwhm_mm > 5 * mu_mm:
        short_verdict = "TOO WIDE"
        verdict = (f"too wide ({fwhm_mm:.2f} mm > 5x diffusion length "
                  f"{mu_mm:.2f} mm) -- likely zone-scale structure, not a "
                  "line defect")
    else:
        short_verdict = "PLAUSIBLE"
        verdict = f"plausible candidate ({fwhm_mm:.2f} mm wide)"

    return {
        "p0": amp_result["p0"], "p1": amp_result["p1"], "mu_px": mu_px,
        "angle_deg": angle_deg, "fwhm_mm": fwhm_mm, "verdict": verdict,
        "short_verdict": short_verdict,
        "amp": amp_result, "phase": phase_result,
    }


def print_line_summary(results, is_reference=None):
    """
    Print the per-line summary table: sym peak, wing RMS, and their ratio
    for the amplitude channel (the candidate leakage signal), the
    zone-to-zone step size (2x the antisymmetric wing level, since anti is
    itself half the difference between the two zones), the sym peak's width,
    the line's own angle, and -- if is_reference marks any lines as
    reference (believed sound) -- each line's amplitude peak as a ratio to
    the mean of the reference lines' peaks, so an elevated line stands out
    as a number rather than something you have to eyeball off a plot.
    Phase's three stats are printed alongside amplitude's too -- phase is
    generally the more reliable channel for detection (see the PHASE note
    near the bottom of this file), so worth reading side by side rather
    than as an afterthought.

    The physical-plausibility verdict (see analyse_deletion_line()'s
    docstring) gets its OWN column -- `verdict` -- placed right after
    fwhm_mm as a single word (PLAUSIBLE / TOO NARROW / TOO WIDE / NO PEAK),
    specifically so a table of several lines shows at a glance whether ANY
    of them are plausible candidates without reading every row's full text.
    The full-sentence reasoning (e.g. "too wide (12.3mm > 5x diffusion
    length 1.3mm)...") is still printed too, in `detail` at the end of the
    row, for when the one-word column raises a question about why.
    """
    has_ref = is_reference is not None and any(is_reference)
    ref_mean = (np.mean([r["amp"]["peak"] for r, is_ref
                        in zip(results, is_reference) if is_ref])
               if has_ref else None)

    header = (f"{'line':>4}  {'angle':>6}  {'amp_peak':>10}  "
              f"{'amp_wing_rms':>13}  {'amp_ratio':>9}  {'amp_step':>9}  "
              f"{'fwhm_mm':>8}  {'verdict':>11}")
    if has_ref:
        header += f"  {'vs_ref':>7}"
    header += (f"  |  {'ph_peak_deg':>11}  {'ph_wing_rms_deg':>16}  "
              f"{'ph_ratio':>8}  {'ph_step_deg':>11}  detail")
    print(header)
    print("-" * len(header))
    for i, res in enumerate(results):
        a, p = res["amp"], res["phase"]
        a_step = 2 * a["anti_step"]
        p_peak, p_wing, p_step = (np.degrees(p["peak"]), np.degrees(p["wing_rms"]),
                                  np.degrees(2 * p["anti_step"]))
        ref_flag = " (ref)" if has_ref and is_reference[i] else ""
        line = (f"{i:4d}  {res['angle_deg']:5.1f}d  {a['peak']:10.4g}  "
                f"{a['wing_rms']:13.4g}  {a['ratio']:9.2f}  {a_step:9.4g}  "
                f"{res['fwhm_mm']:8.3g}  {res['short_verdict']:>11}")
        if has_ref:
            vs_ref = a["peak"] / ref_mean if ref_mean else float("nan")
            line += f"  {vs_ref:6.2f}x"
        line += (f"  |  {p_peak:11.3f}  {p_wing:16.3f}  {p['ratio']:8.2f}  "
                f"{p_step:11.3f}  {res['verdict']}{ref_flag}")
        print(line)


# ----------------------------------------------------------------------------
# 8. PIPELINE
# ----------------------------------------------------------------------------

def _lockin_cache_path(path, fps, f_excite, register, register_upsample,
                       register_reference, reject_outliers, outlier_mad_threshold):
    """
    Deterministic cache file path for (path, loading/registration params).
    The same inputs always map to the same path, so a repeat run with
    unchanged loading/registration settings reuses it automatically, and
    any change to those settings -- or to the source file itself (size,
    mtime) -- lands on a different path instead of silently reusing stale
    data.  Geometry/masking/analysis parameters (fiducial ROI, lines,
    mask_polarity, wing_factor, etc.) are deliberately NOT part of the key:
    none of those change amp/phase/nf, and keying on them would invalidate
    the cache on exactly the parameters you're iterating on to fix a
    masking problem.
    """
    st = os.stat(path)
    key = "|".join(str(v) for v in (
        os.path.abspath(str(path)), st.st_size, st.st_mtime,
        fps, f_excite, register, register_upsample, register_reference,
        reject_outliers, outlier_mad_threshold,
    ))
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(f"{path}.lockin_cache_{digest}.npz")


def analyse(path, fps, f_excite, mm_per_px=None, n_lines=5,
            roi_config_path="roi_config.json", use_saved_config=False,
            auto_geometry=False,
            register=False, register_upsample=20, register_reference="middle",
            reject_outliers=True, outlier_mad_threshold=8.0,
            use_lockin_cache=True, force_recompute=False,
            mask_polarity="auto", mask_close_px=3, pick_mask_roi=False,
            phase_reference="circular_mean", unwrap_phase=True,
            wing_factor=3.0, half_width_mm=4.0, bin_px=0.2, n_along=800,
            ply_thickness_mm=2.0, baseline_frac=0.6, max_shift_mm=1.5,
            detrend_profile=True):
    """
    mm_per_px : millimetres per pixel.  None (default) means take it from
               the geometry config instead (either loaded, or measured
               during interactive setup) -- pass a number here to override
               that, or if you're reusing a saved config from before
               calibration was added.
    use_lockin_cache : cache (amp, phase, nf) -- the output of loading,
               registration, pre-conditioning, and lock-in, i.e.
               everything before geometry/masking/analysis -- to a
               .lockin_cache_<hash>.npz file next to `path` on first run,
               and reuse it on every later run that passes the SAME path,
               fps, f_excite, register*, reject_outliers*, and whose
               source file hasn't changed (see _lockin_cache_path()).
               Default True.  This is what makes iterating on masking or
               line geometry fast: those parameters aren't part of the
               cache key at all, so changing mask_polarity, fiducial_roi,
               line endpoints, wing_factor, etc. and rerunning skips
               straight to "geometry setup" instead of re-decoding and
               re-processing the whole recording.  Safe to leave on --
               changing anything that DOES affect amp/phase (loading,
               registration) computes a different cache path rather than
               reusing a stale one.  Delete the .lockin_cache_*.npz file,
               or pass force_recompute=True, to force a redo (e.g. after
               fixing a bug in loading/registration itself).
    force_recompute : ignore an existing cache and recompute (and
               overwrite it) even if use_lockin_cache=True.  Default False.
    mask_polarity : 'auto' (default), 'bright', or 'dark' -- passed
               straight through to part_mask().  The part can read either
               brighter OR darker than its surroundings depending on
               framing (see part_mask()'s docstring); 'auto' detects this,
               but override it here if auto picks the wrong region.
    mask_close_px : passed straight through to part_mask() -- how much a
               thin divider (a deletion line, a scratch) crossing the part
               gets bridged before picking the largest connected
               component.  Raise this if the fiducial ROI or a line
               endpoint is rejected as "not inside the part mask" under
               BOTH mask_polarity='bright' and 'dark': that pattern means
               the part is being split into disconnected pieces by
               something thin, not that the polarity guess is wrong (the
               error message's diagnostics say which).
    pick_mask_roi : if True and use_saved_config=False, interactive_setup()
               also prompts for a manually-clicked polygon around the part
               outline, used by part_mask() instead of thresholding --
               the fallback for a part that doesn't separate from the
               background by brightness at all.  Stored in roi_config.json
               ("mask_roi") and reused automatically, headlessly, on every
               later run with use_saved_config=True -- no need to pass this
               again.
    n_lines : how many deletion lines to click during interactive setup, or
               to keep from auto_setup()'s ranked candidates when
               auto_geometry=True.  Ignored when use_saved_config=True (the
               config already says how many there are).
    auto_geometry : if True and use_saved_config=False, get line geometry
               and the fiducial ROI from auto_setup() (find_deletion_lines()
               + auto_fiducial_roi()) instead of interactive_setup() --
               fully headless, no clicking, no interactive matplotlib
               backend required.  Requires mm_per_px (no calibration click
               in this path -- pass it explicitly).  ALWAYS check
               lockin_line_candidates.png after a run with this on: every
               candidate the detector considered is drawn there (faded),
               with the selected line(s) solid, specifically because on a
               part whose real signal is a subtle symmetric bump riding on
               a much bigger genuine zone step, the detector can lock onto
               that step instead of the intended deletion line when the two
               score comparably (see auto_setup()'s docstring); the same
               scoring ambiguity has also been seen picking a fiducial ROI
               that lands the phase reference on the wrap boundary (see
               phase_reference below).  Default False -- interactive_setup()
               is the default instead, precisely because both of those auto
               picks have caused real problems here: click through it once
               per recording and reuse the saved config with
               use_saved_config=True afterwards, the same way you would
               with the results of a manual click either way.  Set this
               True only when you've confirmed lockin_line_candidates.png
               looks right and want a fully headless run.
    roi_config_path : where to save (interactive setup) or load (saved
               config) the geometry -- lines (with each one's angle and
               reference-line tag), fiducial ROI, mm_per_px, and the
               processing parameters below -- as JSON.  See
               interactive_setup(), save_roi_config(), load_roi_config().
               Re-saved at the end of geometry setup on every run (loaded
               or freshly clicked), so it always reflects the parameters
               that run actually used.
    use_saved_config : if True, load roi_config_path instead of running
               interactive_setup() -- headless, repeatable, no display
               needed.  Default False: click through setup (including a
               reference/believed-sound tag per line) and save it to
               roi_config_path for next time.  Set this True on every
               re-analysis of the same recording so you're not re-clicking
               (and not accumulating slightly different geometry each run).
               If the loaded config's recorded processing parameters
               (half_width_mm, bin_px, n_along, f_excite, mm_per_px,
               ply_thickness_mm) differ from what THIS call passed in,
               that's printed as an explicit warning naming every parameter
               that differs -- different window widths etc. between runs
               make their figures incomparable, and this is meant to catch
               that before you notice only by eye.
    register : sub-pixel-register every frame before anything else (see
               register_frames()).  Default False.  If the part is instead
               moving coherently with the excitation (e.g. free thermal
               expansion against its fixture), lock-in amplifies that
               motion exactly as it would real signal, smearing both the
               step and any real defect signature -- registration exists
               to remove that.  But it isn't free: it costs real time
               (roughly a pass over every raw frame), and phase correlation
               can occasionally lock onto a spurious large "shift" on
               quiet, low-texture data even when the true motion is zero,
               which then does more damage than the motion it was meant to
               fix.  Worth turning on if the part's material has enough
               thermal expansion to plausibly move a measurable fraction of
               a pixel against its fixture over the recording; not worth it
               for a low-CTE part (e.g. glass) where genuine motion is
               expected to be a small fraction of a pixel -- below what
               registration can usefully resolve anyway.  If you do turn it
               on, check lockin_motion.png afterwards: a spurious lock
               looks like a single large, isolated jump rather than a
               smooth trend or a clean oscillation at the excitation
               frequency.
    register_reference : which frame every other frame is registered to --
               "middle" (default) or "first".  If everything in the output
               looks shifted in one consistent direction rather than just
               cleaned up, that's "first" anchoring the whole corrected
               sequence to one end of a real drift over the recording; see
               register_frames()'s docstring.
    reject_outliers : drop frames corrupted by a camera glitch (typically a
               NUC shutter event) before anything else runs (see
               reject_outlier_frames()).  Default True.
    phase_reference : 'circular_mean' (default) or 'fiducial_roi' -- how
               `phase` gets rotated to put its bulk away from the +/-180
               branch cut before display/analysis.  See reference_phase()'s
               docstring for the full reasoning; in short, 'circular_mean'
               references to the circular mean of every masked pixel's
               phase, which is robust to any one site (including the
               fiducial ROI) happening to sit near the cut, whereas
               'fiducial_roi' (the previous, and only, behaviour) references
               to that ROI's own median and can rotate the cut into the
               middle of the part if the ROI itself is unlucky.  Either way,
               a WARNING prints if too much of the masked part still sits
               near the cut after referencing -- read it; it means the
               phase field spans more than one turn or `part` is picking up
               noise-dominated pixels, not that referencing failed to do
               its job.
    unwrap_phase : if True, spatially unwrap `phase` (skimage.restoration.
               unwrap_phase, see unwrap_phase_field()) over the masked part
               AFTER referencing, and use the unwrapped field for both the
               phase display panel and every per-line phase profile.
               Default False -- only reach for this if reference_phase()'s
               wrap-fraction warning fires AND you've confirmed (e.g. by
               eye on lockin_images.png) that the residual really is a
               spatial wrap and not noise.
    wing_factor, half_width_mm, bin_px, n_along, ply_thickness_mm,
    baseline_frac, max_shift_mm, detrend_profile : passed straight through
               to analyse_deletion_line() / symmetric_antisymmetric_profile()
               for every line -- see their docstrings.  half_width_mm
               (default 4.0 mm here -- deliberately a few diffusion lengths
               at a typical excitation frequency here, wide enough for a
               real wing baseline but narrow enough that zone-scale
               power-density variation elsewhere on the part doesn't
               dominate; a wider window needs detrend_profile more, not
               less) and bin_px (default 0.2 px, the slanted-edge
               oversampling step) are physical/geometric quantities, not
               hard-coded pixel counts, so they carry the same meaning
               across different mm_per_px.  The wing/core boundary sits at
               wing_factor (default 3) diffusion lengths from the line.
               ply_thickness_mm (default 3.0) is the glass ply thickness
               the heat source is imaged through -- sets the narrow-end
               physical-plausibility floor (see analyse_deletion_line()'s
               docstring).  baseline_frac (default 0.6) sets where sym's
               own zero level is measured, as a fraction of half_width_mm
               -- see symmetric_antisymmetric_profile()'s docstring.
               max_shift_mm (default 1.5) clamps the centre-recentring
               search so it can't lock onto the zone-to-zone background
               gradient's own crossing several mm from the actual line --
               see symmetric_antisymmetric_profile()'s docstring.
               detrend_profile (default True) removes a wing-fitted
               straight-line background from each profile before the
               sym/anti split, for the same reason -- see the same
               docstring.  Both of these are ALSO stamped into
               roi_config.json's processing_params and checked for
               cross-run mismatch just like half_width_mm etc. (see
               use_saved_config above) -- and note that whatever value YOU
               pass to any of these processing parameters always wins over
               whatever a loaded config recorded from a previous run; the
               recorded value is only ever used to print a mismatch
               warning, never to silently override this call's own
               arguments.

    fps is ignored for .csq / .seq inputs -- the camera's own per-frame
    timestamps are used instead (see load_csq).

    Returns (amp, phase, nf, results) -- results is one dict per line, in
    the shape produced by analyse_deletion_line(), in the same order as
    roi_config["lines"].
    """
    t_start = time.time()

    cache_file = (_lockin_cache_path(path, fps, f_excite, register, register_upsample,
                                     register_reference, reject_outliers,
                                     outlier_mad_threshold)
                 if use_lockin_cache else None)
    amp = phase = nf = None
    if cache_file is not None and not force_recompute and cache_file.exists():
        try:
            with np.load(cache_file) as z:
                amp, phase, nf = z["amp"], z["phase"], float(z["nf"])
            print(f"  loaded cached lock-in result from {cache_file} -- skipping "
                  "decode/registration/pre-conditioning/lock-in (delete this file, "
                  "or pass force_recompute=True, to redo it from raw frames)")
        except Exception as e:
            print(f"  WARNING: cache at {cache_file} unreadable ({e}) -- recomputing")

    if amp is None:
        with _stage("loading"):
            if str(path).lower().endswith((".csq", ".seq")):
                frames, t = load_csq(path)
            else:
                frames = load_sequence(path)
                t = build_time_vector(len(frames), fps)
            print(f"  {frames.shape[0]} frames, {frames.nbytes / 1e6:.0f} MB in memory")

        if reject_outliers:
            with _stage("checking for corrupted frames"):
                frames, t = reject_outlier_frames(frames, t,
                                                  mad_threshold=outlier_mad_threshold)
                detect_frozen_frames(frames, t)

        if register:
            with _stage("registering frames"):
                raw_cy, raw_cx = track_part_centroid(frames)
                offsets = register_frames(frames, t, f_excite,
                                          upsample_factor=register_upsample,
                                          reference=register_reference)
                reg_cy, reg_cx = track_part_centroid(frames)

                figm, axm = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
                axm[0].plot(t, offsets[:, 1], label="dx")
                axm[0].plot(t, offsets[:, 0], label="dy")
                axm[0].set_ylabel("registration offset [px]")
                axm[0].legend(); axm[0].set_title("frame-to-frame registration offset")
                # Mean-subtracted so "before" and "after" overlay on the same
                # scale regardless of the part's absolute position in frame.
                axm[1].plot(t, raw_cx - np.nanmean(raw_cx), label="cx, before")
                axm[1].plot(t, reg_cx - np.nanmean(reg_cx), label="cx, after")
                axm[1].plot(t, raw_cy - np.nanmean(raw_cy), label="cy, before", ls="--")
                axm[1].plot(t, reg_cy - np.nanmean(reg_cy), label="cy, after", ls="--")
                axm[1].set_xlabel("time [s]")
                axm[1].set_ylabel("part centroid, mean-subtracted [px]")
                axm[1].legend(fontsize=8, ncol=2)
                axm[1].set_title("part centroid vs time -- independent motion check")
                figm.tight_layout()
                figm.savefig("lockin_motion.png", dpi=150)
                plt.close(figm)

        with _stage("pre-conditioning"):
            frames, t = decimate(frames, t, target_fps=max(8 * f_excite, 1.0))
            frames = remove_global_offsets(frames)
            frames = detrend_per_pixel(frames, t)
            frames, t = trim_to_whole_cycles(frames, t, f_excite)

        with _stage("lock-in"):
            amp, phase = lockin(frames, t, f_excite)
            nf = noise_floor(frames, t, f_excite)
            print(f"  noise floor (off-frequency amplitude): {nf:.4g}")
            print(f"  peak amplitude / noise floor: {amp.max() / nf:.1f}")

            # 2f check: strong 2f with weak f means the excitation was amplitude
            # modulated rather than switched on/off, so power varied at twice the rate.
            amp2f, _ = lockin(frames, t, 2 * f_excite)
            print(f"  median amplitude at 2f / at f: {np.median(amp2f) / np.median(amp):.2f}"
                  "   (should be well under 1)")

        if cache_file is not None:
            np.savez_compressed(cache_file, amp=amp, phase=phase, nf=np.asarray(nf))
            print(f"  cached lock-in result to {cache_file} -- reruns with the same "
                  "loading/registration settings will reuse it (see use_lockin_cache)")

    candidates = n_selected = None
    with _stage("geometry setup"):
        if use_saved_config:
            roi_config = load_roi_config(roi_config_path)
        elif auto_geometry:
            if mm_per_px is None:
                raise ValueError(
                    "auto_geometry=True needs mm_per_px passed explicitly -- "
                    "there's no calibration click in automatic mode"
                )
            roi_config, candidates, n_selected = auto_setup(
                amp, f_excite, mm_per_px, n_lines=n_lines,
                mask_polarity=mask_polarity, mask_close_px=mask_close_px)
        else:
            roi_config = interactive_setup(amp, phase=phase, n_lines=n_lines,
                                           calibrate=(mm_per_px is None),
                                           mask_roi=pick_mask_roi)

        if mm_per_px is None:
            mm_per_px = roi_config.get("mm_per_px")
        if mm_per_px is None:
            raise ValueError(
                "no mm_per_px available -- pass mm_per_px explicitly, "
                "calibrate during interactive setup, or use a saved config "
                "that already has one"
            )

        fiducial_roi = tuple(roi_config["fiducial_roi"])
        lines_raw = roi_config["lines"]
        if not lines_raw:
            raise ValueError("roi_config has no lines -- nothing to analyse")
        lines_geom = [(tuple(ln["p0"]), tuple(ln["p1"])) for ln in lines_raw]
        is_reference = [bool(ln.get("is_reference", False)) for ln in lines_raw]
        print(f"  {len(lines_geom)} line(s), fiducial ROI "
              f"y[{fiducial_roi[0]}:{fiducial_roi[1]}] "
              f"x[{fiducial_roi[2]}:{fiducial_roi[3]}], "
              f"mm_per_px={mm_per_px:.4g}")

        # Catch a re-analysis that silently used different processing
        # parameters than the run that produced this config: different
        # window widths etc. make figures/results across runs
        # incomparable, and this is much cheaper to catch here than by
        # noticing two plots don't line up.
        current_params = {
            "mm_per_px": mm_per_px, "half_width_mm": half_width_mm,
            "bin_px": bin_px, "n_along": n_along, "f_excite": f_excite,
            "ply_thickness_mm": ply_thickness_mm, "baseline_frac": baseline_frac,
            "mask_polarity": mask_polarity, "max_shift_mm": max_shift_mm,
            "detrend_profile": detrend_profile, "phase_reference": phase_reference,
            "unwrap_phase": unwrap_phase,
        }
        stored_params = roi_config.get("processing_params")
        if stored_params:
            mismatched = [(k, stored_params[k], v)
                         for k, v in current_params.items()
                         if k in stored_params and stored_params[k] != v]
            if mismatched:
                print(f"  WARNING: this run's parameters differ from what's "
                      f"recorded in {roi_config_path} -- figures/results "
                      "from different runs of this config won't be "
                      "directly comparable:")
                for k, old, new in mismatched:
                    print(f"    {k}: recorded {old!r}, this run {new!r}")

        roi_config["processing_params"] = current_params
        save_roi_config(roi_config, roi_config_path)

    with _stage("spatial processing"):
        # Only the part mask survives here -- spatial_highpass() is no
        # longer part of the default pipeline.  On a piecewise-smooth field
        # with a genuine step at every line, subtracting a blurred copy of
        # the image turns each step into a spurious antisymmetric dipole
        # (see spatial_highpass()'s own docstring), which would swamp the
        # much smaller symmetric leakage signal this analysis is built to
        # isolate.  The symmetric/antisymmetric decomposition below handles
        # the step directly, on the raw amplitude and phase, instead.
        alpha = 5e-7                                    # m^2/s, glass
        mu_mm = np.sqrt(alpha / (np.pi * f_excite)) * 1000
        print(f"  diffusion length: {mu_mm:.2f} mm")
        part = part_mask(amp, mask_polarity=mask_polarity, close_px=mask_close_px,
                         fiducial_roi=fiducial_roi, line_endpoints=lines_geom,
                         mask_roi=roi_config.get("mask_roi"))

        # Stamped on every figure below -- different window widths etc.
        # across runs make figures incomparable at a glance otherwise.
        params_str = (f"half_width={half_width_mm:.1f}mm  bin_px={bin_px:.2g}  "
                     f"n_along={n_along}  f_excite={f_excite:.4g}Hz  "
                     f"mm_per_px={mm_per_px:.4g}  ply={ply_thickness_mm:.2f}mm")

    with _stage("phase referencing"):
        # Needs `part` (just computed above) -- the default 'circular_mean'
        # method references to the whole masked population, not a single
        # site, so it has to come after the part mask exists.  See
        # reference_phase()'s docstring for why this replaced referencing
        # to the fiducial ROI alone as the default: that ROI's own phase
        # can happen to sit near the +/-180 branch cut, which then rotates
        # the cut into the middle of the part instead of away from it.
        phase, phase_ref_rad, phase_wrap_frac = reference_phase(
            phase, part, fiducial_roi=fiducial_roi, method=phase_reference)

        if unwrap_phase:
            phase = unwrap_phase_field(phase, part)
            print("  phase spatially unwrapped over the masked part -- the "
                  "display panel and every per-line phase profile below "
                  "now use the unwrapped field")

    with _stage("rendering diagnostic images"):
        disp = np.where(part, amp, np.nan)
        lo, hi = np.nanpercentile(disp, [2, 98])

        # Mask and scale the phase panel the same way as the amplitude
        # panel above -- background pixels carry no coherent signal at the
        # lock-in frequency, so their phase is essentially random; left in,
        # that speckle both stretches the colour scale until the real part
        # looks uniformly flat and buries the part's outline in noise.
        # `phase` is already referenced (and possibly unwrapped) upstream --
        # see the "phase referencing" stage above -- so no additional
        # recentring is needed here; percentile clipping (same 2-98% as the
        # amplitude panel, NOT a fixed +/-180 range) keeps a handful of
        # noisy/low-SNR pixels -- or, if unwrap_phase=True, the wider
        # unwrapped range -- from blowing out the colour scale without
        # discarding the rest of the part the way a hard concentration
        # cutoff would.
        phase_deg = np.where(part, np.degrees(phase), np.nan)
        lo_p, hi_p = np.nanpercentile(phase_deg, [2, 98])

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        im0 = ax[0].imshow(amp);     ax[0].set_title("amplitude (raw)")
        # Mask outline on the raw panel so a bad mask (wrong polarity, wrong
        # region) is visible at a glance instead of inferred from downstream
        # plots -- see part_mask()'s docstring on why polarity isn't assumed.
        ax[0].contour(part.astype(float), levels=[0.5], colors="red", linewidths=1.2)
        im1 = ax[1].imshow(disp, vmin=lo, vmax=hi)
        ax[1].set_title("amplitude (masked)")
        im2 = ax[2].imshow(phase_deg, cmap="twilight", vmin=lo_p, vmax=hi_p)
        ax[2].set_title("phase [deg]")
        for a, im in zip(ax, (im0, im1, im2)):
            fig.colorbar(im, ax=a, fraction=0.046)

        # Overlay the fiducial ROI only -- lines aren't drawn here; with
        # several lines clicked, the overlay clutters the image more than
        # it confirms, and each line gets its own dedicated, clearly
        # labelled profile figure below anyway.
        fy0, fy1, fx0, fx1 = fiducial_roi
        for a in (ax[1], ax[2]):
            a.add_patch(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0,
                                  fill=False, edgecolor="lime", lw=1.2))
            a.set_xlim(0, part.shape[1]); a.set_ylim(part.shape[0], 0)

        fig.tight_layout(rect=[0, 0.02, 1, 1])
        fig.text(0.5, 0.005, params_str, ha="center", fontsize=8, color="dimgray")
        fig.savefig("lockin_images.png", dpi=150)
        plt.close(fig)

        if candidates is not None:
            plot_line_candidates(amp, part, candidates, n_selected, fiducial_roi)

    with _stage("symmetric/antisymmetric line analysis"):
        results = []
        for i, (p0, p1) in enumerate(lines_geom):
            res = analyse_deletion_line(amp, phase, p0, p1, mu_mm, mm_per_px,
                                        ply_thickness_mm=ply_thickness_mm,
                                        half_width_mm=half_width_mm,
                                        bin_px=bin_px, n_along=n_along,
                                        wing_factor=wing_factor,
                                        baseline_frac=baseline_frac,
                                        max_shift_mm=max_shift_mm,
                                        detrend_profile=detrend_profile)
            results.append(res)
            a = res["amp"]
            baseline_lo_mm, baseline_hi_mm = (v * mm_per_px for v in a["baseline_px"])
            ref_note = "  [reference]" if is_reference[i] else ""
            print(f"  line {i} ({res['angle_deg']:.1f} deg){ref_note}: "
                  f"centre shift {a['centre_shift_px']:+.2f} px "
                  f"({a['centre_shift_mm']:+.3f} mm) from clicked position, "
                  f"trend slope {a['trend_slope']:+.4g}/mm, "
                  f"baseline region |d| in [{baseline_lo_mm:.1f}, "
                  f"{baseline_hi_mm:.1f}] mm ({a['n_baseline_samples']} samples), "
                  f"sym peak {a['peak']:.4g}, wing RMS {a['wing_rms']:.4g}, "
                  f"ratio {a['ratio']:.2f}, width {res['fwhm_mm']:.2f} mm "
                  f"-- {res['verdict']}")

        n = len(results)
        fig2, ax2 = plt.subplots(n, 2, figsize=(13, 4 * n), squeeze=False)
        for i, res in enumerate(results):
            shift_mm = res["amp"]["centre_shift_mm"]
            for col, (label, r, scale) in enumerate((
                    ("amplitude", res["amp"], 1.0),
                    ("phase [deg]", res["phase"], 180.0 / np.pi))):
                a = ax2[i][col]
                # Plot the FULL -half_width..+half_width range: d/sym/anti
                # are one-sided by construction (see
                # symmetric_antisymmetric_profile()), so the other half is
                # reconstructed here via their own defined symmetry -- sym
                # mirrored as-is (even about the centre), anti mirrored with
                # a sign flip (odd about the centre).  This is intentionally
                # redundant (sym/anti are symmetric/antisymmetric by
                # construction, so the mirror is exact) -- it's the visual
                # check that the vertical "located centre" marker below
                # actually sits where the decomposition was centred, not
                # proof the decomposition itself is unbiased.
                d_mm = np.concatenate([-r["d"][1:][::-1], r["d"]]) * mm_per_px
                sym_full = np.concatenate([r["sym"][1:][::-1], r["sym"]]) * scale
                anti_full = np.concatenate([-r["anti"][1:][::-1], r["anti"]]) * scale
                a.axvspan(-mu_mm, mu_mm, alpha=0.15, color="tab:blue",
                         label=f"diffusion length ({mu_mm:.1f} mm)")
                a.axhline(0, lw=0.5, color="k")
                a.axvline(0, lw=1.2, ls="--", color="tab:red",
                         label=f"located centre ({shift_mm:+.3f} mm from click)")
                a.plot(d_mm, sym_full, label="sym (candidate defect)")
                a.plot(d_mm, anti_full, label="anti (zone step)")
                a.set_xlabel("distance from line centre [mm]")
                a.set_ylabel(label)
                ref_tag = " [reference]" if is_reference[i] else ""
                a.set_title(f"line {i} ({res['angle_deg']:.1f} deg){ref_tag}: {label}")
                a.legend(fontsize=8)
        fig2.tight_layout(rect=[0, 0.02, 1, 1])
        fig2.text(0.5, 0.005, params_str, ha="center", fontsize=8, color="dimgray")
        fig2.savefig("lockin_line_profiles.png", dpi=150)
        plt.close(fig2)

        print()
        print_line_summary(results, is_reference=is_reference)

    print(f"done -- total {_fmt_dt(time.time() - t_start)}")
    return amp, phase, nf, results


# ----------------------------------------------------------------------------
# WHAT THE OUTPUT MEANS
# ----------------------------------------------------------------------------
#
# SYM (the candidate leakage signal)
#   A real buried heat source produces a peak whose width is comparable to the
#   thermal diffusion length -- millimetres, not pixels -- and, since the
#   source sits at the coating plane and is imaged through the full glass
#   ply, it physically CANNOT be narrower than the ply thickness either:
#   that's the floor analyse_deletion_line()'s verdict checks the measured
#   FWHM against.  "implausibly narrow" means surface artefact or sampling
#   residue, not a real buried source, however sharp the raw amplitude bump
#   looks.  This width check is your best single discriminator against false
#   positives.  A peak sitting at ratio (sym peak / wing RMS) well under ~3
#   is not distinguishable from noise, whatever its raw value -- and note
#   that width is only trustworthy if the line's angle cleared
#   check_line_angle()'s warning; a near-axis-aligned or near-45-degree line
#   can't be resolved finer than about a pixel regardless of bin_px.
#
# ANTI (the zone-to-zone step)
#   This is the expected, physical difference in power density between the
#   two zones the line divides -- not itself a defect indicator.  Its
#   magnitude (2x the wing-level plateau of anti, since anti is defined as
#   HALF the difference) is worth cross-checking against your own geometry /
#   coating-thickness model for the two zones, mostly as a sanity check that
#   the decomposition centred correctly (see centre_shift_px/_mm in each
#   line's result -- a large shift means the clicked or given line position
#   was off by that much; recentring is clamped to max_shift_mm and falls
#   back to zero shift with a warning rather than trusting a multi-mm
#   "crossing", which is the zone gradient's own transition, not the line).
#
# PHASE
#   Prefer phase for detection.  Amplitude scales with emissivity, view angle,
#   and reflected radiation; phase is a ratio of two quantities that scale
#   identically, so those multiplicative nuisances cancel.  Phase depends on
#   the thermal path -- source depth and lateral distance -- not on how shiny
#   the surface is.  On a low-emissivity coated face, amplitude images are
#   dominated by reflection artifacts and phase images largely are not.  This
#   pipeline decomposes phase the same symmetric/antisymmetric way as
#   amplitude, side by side, for exactly this reason.
#
#   Before any of that, `phase` has to be REFERENCED (rotated so its bulk
#   sits away from lockin()'s +/-180 deg branch cut -- see
#   reference_phase()) and, if the WARNING it prints fires, possibly
#   UNWRAPPED (unwrap_phase_field(), analyse()'s unwrap_phase=True).  A
#   part that reads as two discontinuous halves in lockin_images.png's
#   phase panel, or profiles with unexplained +/-20 deg swings that don't
#   track any real thermal feature, is very likely still sitting on the
#   branch cut, not a genuine defect signature -- check the wrap-fraction
#   warning before trusting either.
#
# ALONG-LINE PROFILE SHAPE (along_line_profile() -- not run by default now,
# call it yourself on `amp` for a chosen line if you want this view)
#   Smooth hump peaking mid-span, falling to zero at both busbars
#       -> distributed leakage through residual film.  Follows dV(y)^2, so it
#          should match your DMM voltage survey squared.  Fix is process dose.
#
#   Localised spike somewhere along an otherwise quiet line
#       -> discrete bridge.  Redeposited conductive debris is the prime
#          suspect.  Fix is gas management or a cleaning pass, NOT more power.
#
#   Elevated but flat along the whole line
#       -> suspect a HAZ sheet-resistance band running parallel to the cut,
#          rather than conduction across it.  Check against sheet resistance
#          mapping near the trench.
#
# NULL TEST
#   Run the identical procedure with the supply off.  Anything that still
#   appears is an artifact of your setup, and its magnitude is the real
#   detection floor -- usually higher than the off-frequency estimate.
#
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # First run on a new recording: auto_geometry=False (the default) walks
    # through interactive_setup() -- click each deletion line's endpoints,
    # then the fiducial ROI (shown side by side with phase; a click too
    # close to the phase wrap boundary is rejected on the spot, see
    # interactive_setup()'s docstring).  This saves roi_config.json.  Every
    # later run of the SAME recording: set use_saved_config=True to reload
    # roi_config.json and skip clicking entirely.  auto_geometry=True
    # switches to fully headless detection (find_deletion_lines() +
    # auto_fiducial_roi()) instead -- only turn it on after confirming
    # lockin_line_candidates.png picks the right line(s) on this part; see
    # analyse()'s docstring on auto_geometry for why manual is the default.
    analyse(
        path="FLIR0029.csq",
        fps=30.0,
        f_excite=0.1,
        mm_per_px=2.23,            # required when auto_geometry=True;
                                   # otherwise measured via interactive
                                   # setup's calibration click if left None
        n_lines=1,                 # how many deletion lines to click/detect
        use_saved_config=False,    # True to reload roi_config.json instead
        roi_config_path="roi_config.json",
        # auto_geometry=False (the default) -- set True for auto_setup()
        # (find_deletion_lines() + auto_fiducial_roi(), fully headless)
        # instead of clicking through interactive_setup().
        # register=False (the default) -- turn on only if the part's
        # material has enough thermal expansion to plausibly move a
        # measurable fraction of a pixel against its fixture; see
        # analyse()'s docstring for why it's off by default here (a
        # low-CTE glass laminate isn't expected to move enough for
        # registration to help, and it can occasionally hurt).
        # use_lockin_cache=True (the default) -- decode/registration/
        # lock-in are cached to a .lockin_cache_<hash>.npz next to the
        # .csq on first run and reused on every later run with the same
        # path/fps/f_excite/register* settings, so iterating on
        # mask_polarity, fiducial_roi, line geometry etc. below is fast
        # (no re-decode).  Delete that file, or pass force_recompute=True,
        # after fixing anything upstream of lock-in.
        mask_polarity="auto",      # 'auto' / 'bright' / 'dark' -- override
                                   # here (not by editing analyse()'s
                                   # default) if auto picks the wrong region
        mask_close_px=3,           # raise this if the fiducial ROI / a line
                                   # endpoint is rejected under BOTH 'bright'
                                   # and 'dark' -- that pattern means a line
                                   # or streak is splitting the part into two
                                   # disconnected pieces, not a polarity
                                   # mismatch (see part_mask()'s docstring)
        # phase_reference="circular_mean" (the default) -- references phase
        # to the circular mean of the whole masked part rather than just
        # the fiducial ROI, so one unlucky ROI site can't rotate the wrap
        # boundary into the middle of the part; see reference_phase().
        # unwrap_phase=False (the default) -- set True only if
        # reference_phase()'s wrap-fraction warning fires AND you've
        # confirmed by eye that it's a genuine spatial wrap, not noise.
        ply_thickness_mm=2.0,      # the glass ply the source is imaged
                                   # through -- sets the narrow-end
                                   # physical-plausibility floor
        half_width_mm=4.0,         # cross-line window -- a few diffusion
                                   # lengths at a typical f_excite here;
                                   # detrend_profile matters more the wider
                                   # this is set
        bin_px=0.2,                # slanted-edge oversampling step
        n_along=800,               # along-line samples pooled per bin
        # max_shift_mm=1.5 (the default) -- clamps how far centre
        # recentring is allowed to search; a real line is 25-50 um wide, so
        # a "steepest point" found several mm away is the zone gradient's
        # own crossing, not the line -- see symmetric_antisymmetric_profile().
        # detrend_profile=True (the default) -- removes a wing-fitted
        # straight-line background from each profile before the sym/anti
        # split, since the background here can vary by an order of
        # magnitude over the same window the line's own signal sits in.
    )
