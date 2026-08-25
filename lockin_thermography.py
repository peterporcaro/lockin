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

Dependencies: numpy, scipy, matplotlib
Optional: flirpy (pip install flirpy) for direct .csq / .seq import -- see
load_csq() below.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, binary_fill_holes
import matplotlib.pyplot as plt


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

    for i, fff in enumerate(seq):
        raw = _decode_raw_record(fff)
        frames[i] = raw2temp(raw.astype(np.float64), fff.meta)
        if "Timestamp" in fff.meta:
            raw_ts[i] = fff.meta["Timestamp"]

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

    print(f"  loaded {n} frames ({w}x{h}) from {path}")
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
# 2. PRE-CONDITIONING
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
# 3. THE LOCK-IN ITSELF
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
# 4. SPATIAL PROCESSING
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
    """
    return amp - gaussian_filter(amp, sigma_px)


def part_mask(amp, threshold_frac=0.4, percentile=95, erosion_px=10):
    """
    Binary mask of the part footprint within the frame.

    Thresholding at a fraction of the bright end of the amplitude
    distribution separates the part from the cooler background; filling
    holes closes over the deletion lines themselves; eroding pulls the
    boundary in from the true part edge so that edge -- which the spatial
    high-pass otherwise rings against -- doesn't contaminate the interior.
    """
    part = amp > threshold_frac * np.percentile(amp, percentile)
    return binary_erosion(binary_fill_holes(part), iterations=erosion_px)


def find_deletion_line(amp_hp, part, ridge_percentile=97, min_ridge_px=10):
    """
    Auto-detect the single strongest deletion line inside the part and
    return its endpoints as ((x0, y0), (x1, y1)).

    Takes the brightest pixels of the (masked, high-passed) amplitude image
    -- the line stands out because it's spatially coherent while noise
    isn't -- and fits a straight line through them by total least squares
    (the major axis of their amplitude-weighted covariance), which treats x
    and y symmetrically and so doesn't break down on a near-vertical line
    the way an ordinary y-on-x regression would. The endpoints are the
    extremes of the ridge pixels' projection onto that axis, i.e. roughly
    busbar to busbar for a line that runs the full extent of the part.

    Only fits ONE line. If the part has multiple deletion lines, this will
    fit through the combined pixel cloud and return something meaningless
    -- crop amp_hp / part to isolate the line of interest first, or pass
    line_endpoints explicitly to analyse() instead of relying on this.
    """
    ridge = part & (amp_hp > np.percentile(amp_hp[part], ridge_percentile))
    ys, xs = np.nonzero(ridge)
    if len(xs) < min_ridge_px:
        raise ValueError(
            f"only {len(xs)} ridge pixels found (need >= {min_ridge_px}) -- "
            "lower ridge_percentile, check the part mask, or pass "
            "line_endpoints explicitly"
        )

    pts = np.stack([xs, ys], axis=1).astype(float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    weights = np.clip(amp_hp[ys, xs], 0, None)
    cov = (centered * weights[:, None]).T @ centered / weights.sum()
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]

    proj = centered @ direction
    p0 = centroid + direction * proj.min()
    p1 = centroid + direction * proj.max()
    return tuple(p0), tuple(p1)


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


# ----------------------------------------------------------------------------
# 5. PIPELINE
# ----------------------------------------------------------------------------

def analyse(path, fps, f_excite, fiducial_roi=None, mm_per_px=2.5,
            line_endpoints=None):
    """
    fiducial_roi : (y0, y1, x0, x1) around the reference resistor in the frame.
                   Its phase becomes the zero reference for the phase image.
    line_endpoints : ((x0, y0), (x1, y1)) pixel coordinates of one deletion
                      line, or None (default) to auto-detect it with
                      find_deletion_line().

    fps is ignored for .csq / .seq inputs -- the camera's own per-frame
    timestamps are used instead (see load_csq).
    """
    print("loading...")
    if str(path).lower().endswith((".csq", ".seq")):
        frames, t = load_csq(path)
    else:
        frames = load_sequence(path)
        t = build_time_vector(len(frames), fps)

    print("pre-conditioning...")
    frames, t = decimate(frames, t, target_fps=max(8 * f_excite, 1.0))
    frames = remove_global_offsets(frames)
    frames = detrend_per_pixel(frames, t)
    frames, t = trim_to_whole_cycles(frames, t, f_excite)

    print("lock-in...")
    amp, phase = lockin(frames, t, f_excite)
    nf = noise_floor(frames, t, f_excite)
    print(f"  noise floor (off-frequency amplitude): {nf:.4g}")
    print(f"  peak amplitude / noise floor: {amp.max() / nf:.1f}")

    # 2f check: strong 2f with weak f means the excitation was amplitude
    # modulated rather than switched on/off, so power varied at twice the rate.
    amp2f, _ = lockin(frames, t, 2 * f_excite)
    print(f"  median amplitude at 2f / at f: {np.median(amp2f) / np.median(amp):.2f}"
          "   (should be well under 1)")

    if fiducial_roi is not None:
        y0, y1, x0, x1 = fiducial_roi
        ref = np.median(phase[y0:y1, x0:x1])
        phase = np.angle(np.exp(1j * (phase - ref)))
        print(f"  phase referenced to fiducial ({np.degrees(ref):.1f} deg)")

    alpha = 5e-7                                    # m^2/s, glass
    mu_mm = np.sqrt(alpha / (np.pi * f_excite)) * 1000
    sigma_px = 2.5 * mu_mm / mm_per_px
    print(f"  diffusion length {mu_mm:.2f} mm -> highpass sigma {sigma_px:.1f} px")

    part = part_mask(amp)
    filled = np.where(part, amp, np.median(amp[part]))   # kill the edge step
    amp_hp = spatial_highpass(filled, sigma_px)

    disp = np.where(part, amp_hp, np.nan)
    lo, hi = np.nanpercentile(disp, [2, 98])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    im0 = ax[0].imshow(amp);     ax[0].set_title("amplitude (raw)")
    im1 = ax[1].imshow(disp, vmin=lo, vmax=hi)
    ax[1].set_title("amplitude (spatial highpass, masked)")
    im2 = ax[2].imshow(np.degrees(phase), cmap="twilight")
    ax[2].set_title("phase [deg]")
    for a, im in zip(ax, (im0, im1, im2)):
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig("lockin_images.png", dpi=150)

    if line_endpoints is None:
        line_endpoints = find_deletion_line(amp_hp, part)
        (x0, y0), (x1, y1) = line_endpoints
        length_mm = np.hypot(x1 - x0, y1 - y0) * mm_per_px
        print(f"  auto-detected line: ({x0:.0f}, {y0:.0f}) -> ({x1:.0f}, {y1:.0f})"
              f"  ({length_mm:.0f} mm)")

    p0, p1 = line_endpoints
    off, xprof = cross_line_profile(amp_hp, p0, p1)
    pos, aprof = along_line_profile(amp_hp, p0, p1)

    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4.5))
    ax2[0].plot(off * mm_per_px, xprof)
    ax2[0].axhline(0, lw=0.5, color="k")
    ax2[0].axvspan(-mu_mm, mu_mm, alpha=0.15,
                   label=f"+/- diffusion length ({mu_mm:.1f} mm)")
    ax2[0].set_xlabel("distance across line [mm]")
    ax2[0].set_ylabel("lock-in amplitude")
    ax2[0].legend()
    ax2[1].plot(pos * mm_per_px, aprof)
    ax2[1].set_xlabel("position along line [mm]  (busbar to busbar)")
    ax2[1].set_ylabel("integrated amplitude")
    fig2.tight_layout()
    fig2.savefig("lockin_profiles.png", dpi=150)

    return amp, amp_hp, phase, nf


# ----------------------------------------------------------------------------
# WHAT THE OUTPUT MEANS
# ----------------------------------------------------------------------------
#
# CROSS-LINE PROFILE WIDTH
#   A real buried heat source produces a peak whose width is comparable to the
#   thermal diffusion length -- millimetres, not pixels.  A feature much
#   narrower than that is a surface artifact: contamination, a reflection, or
#   registration error from the part physically moving.  This width check is
#   your best single discriminator against false positives.
#
# ALONG-LINE PROFILE SHAPE
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
# PHASE
#   Prefer phase for detection.  Amplitude scales with emissivity, view angle,
#   and reflected radiation; phase is a ratio of two quantities that scale
#   identically, so those multiplicative nuisances cancel.  Phase depends on
#   the thermal path -- source depth and lateral distance -- not on how shiny
#   the surface is.  On a low-emissivity coated face, amplitude images are
#   dominated by reflection artifacts and phase images largely are not.
#
# NULL TEST
#   Run the identical procedure with the supply off.  Anything that still
#   appears is an artifact of your setup, and its magnitude is the real
#   detection floor -- usually higher than the off-frequency estimate.
#
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    analyse(
        path="FLIR0022.csq",
        fps=30.0,
        f_excite=0.1,
        fiducial_roi=(150, 100, 150, 100),
        mm_per_px=3,
        # line_endpoints left as None -- auto-detected by find_deletion_line().
        # Pass explicit ((x0, y0), (x1, y1)) here to override if detection
        # picks up the wrong feature.
    )
