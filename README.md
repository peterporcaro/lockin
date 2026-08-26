# Lock-In Thermography for Deletion-Line Inspection

Tools for detecting resistive leakage across laser-ablated "deletion lines" in
a transparent conductive coating (e.g. an aircraft transparency heater), using
lock-in thermography: power the part with a modulated supply, record an IR
sequence, and extract the component of the thermal response that is coherent
with the modulation frequency.

Two scripts:

| File | Role |
|---|---|
| `psu_control.py` | Drives the bench power supply through a square-wave on/off modulation cycle and logs current draw. |
| `lockin_thermography.py` | Loads the recorded IR sequence, runs the lock-in, and pulls a symmetric leakage signal out from underneath the part's normal (and much larger) zone-to-zone power-density structure. |

This document covers what each script does, how its pieces fit together, why
they're built the way they are, and how to run the test suite.

---

## 1. Power supply control — `psu_control.py`

A short, linear script (no functions, runs top-to-bottom) that talks to a
programmable AC supply over SCPI/VISA and produces the square-wave excitation
the lock-in in `lockin_thermography.py` is built to extract.

### What it does

1. Opens the instrument via `pyvisa` using its VISA resource string
   (`USB0::0x0A69::0x0883::96160900000067::INSTR` — a specific unit; change
   this for your own supply) and clears any stale error state (`*CLS`).
2. Configures the **carrier**: the supply's own AC output frequency and
   voltage (`FREQ`, `VOLT:AC`), current limit (`CURR:LIM`), and AC coupling
   (`OUTP:COUP AC`). This is the actual waveform the heater coating runs on —
   it has nothing to do with the lock-in frequency.
3. Checks `SYST:ERR?` after configuration and aborts (closing the instrument
   cleanly) if the supply rejected anything, rather than silently proceeding
   on a misconfigured unit.
4. Enables the output (`OUTP ON`), starting from `VOLT:AC 0` so enabling
   itself doesn't produce a transient.
5. Runs the **modulation**: alternates the AC voltage between `V_OP` ("on")
   and `0` ("off") for `CYCLES` full periods of `F_MOD`, holding each state
   for `DUTY` / `1 - DUTY` of the period. This on/off cycling *is* the signal
   the lock-in in the analysis script demodulates — `F_MOD` here must match
   `f_excite` passed to `analyse()`.
6. While each state holds, polls `MEAS:CURR:AC?` once a second and appends a
   row to `excitation_log.csv` — a live record of what the part actually
   drew, not just what was commanded.
7. In a `finally` block: forces the output to 0 V and off, and closes the
   VISA session — this runs even if the loop is interrupted, so the supply
   never gets left live.

### Configuration constants (top of file)

| Constant | Meaning |
|---|---|
| `F_MOD` | Modulation frequency, Hz — the lock-in frequency. Must match `f_excite` in the analysis. |
| `DUTY` | Fraction of each modulation period the supply is on (0.5 = square wave). |
| `CYCLES` | Number of modulation periods to run. |
| `F_LINE` | Carrier frequency the supply's AC output runs at (Hz) — what the heater physically sees. |
| `V_OP` | Carrier RMS voltage during the "on" phase. |
| `I_LIM` | Current limit set on the supply — set above expected draw, below anything alarming. |

Total run time and the on/off split are computed and printed at startup
(`period ... -> ... on, ... off`, `total run ... min`) so a mis-set `DUTY` or
`CYCLES` is obvious before the supply is even opened.

### Output — `excitation_log.csv`

Columns: `t_s` (seconds since start), `cycle` (0-indexed modulation cycle),
`state` (`"on"` / `"off"`), `current_A` (live AC current reading). One row
per second while a state holds. This is a diagnostic record — e.g. to
confirm current draw actually tracked the commanded state, or to catch a
supply that tripped its current limit mid-run — not an input to
`lockin_thermography.py` (which works from the recorded IR sequence and its
own `f_excite` parameter, not from this log).

### Dependencies

`pyvisa`, plus a working VISA backend (NI-VISA, Keysight IO Libraries, or
`pyvisa-py`) that can see the instrument. No GPIO/serial code — everything
goes through the VISA resource manager.

### Design notes

- **Linear script, not a library.** There's nothing here another script
  needs to import — it's meant to be edited (resource string, constants) and
  run directly for a given test, not parameterized as a reusable module.
- **Error checking after configuration, not after every write.** SCPI
  `*OPC?`/`SYST:ERR?` round-trips are slow; checking once after the whole
  configuration block (rather than after each individual `write`) keeps the
  startup fast while still catching a rejected setting before power is ever
  applied.
- **The `finally` block is the important part.** Any exception during the
  modulation loop (Ctrl-C, a VISA timeout, a bad current reading) still
  drives the output to 0 V and off before the script exits — a bench supply
  left live at `V_OP` because a script crashed is the failure mode this
  guards against.

---

## 2. Lock-in thermography analysis — `lockin_thermography.py`

This is the larger, more heavily documented script, and where inline
docstrings explain *why* each step exists — the summary below extracts the
essentials, but the docstrings in the file are the authoritative reference and
go into more depth on the physics behind each choice.

### The physical picture this script is built around

- The part is imaged **through** something (a glass ply, in the case that
  drove several of the design decisions below) — any real thermal signature
  is blurred by *at least* that thickness before it reaches the camera, and
  by the thermal diffusion length at the excitation frequency on top of that.
  It physically cannot be narrower than either.
- Deletion lines divide the coating into zones that **legitimately** run at
  different power density (different heater geometry, graded coating
  thickness) — the lock-in amplitude has a genuine, physical **step** at
  every line, and that step is *not* itself a defect.
- A resistive leakage defect is a much smaller, **symmetric** heat bump
  riding on top of that step: real heat spreads into both neighbouring zones
  alike, since heat doesn't know which zone it started in.

Most of the interesting design decisions in this script exist to separate
that small symmetric bump from the (much larger) step, from part motion, and
from camera/data artefacts — without ever discarding the step, since it's
diagnostic in its own right.

### Pipeline overview

`analyse()` is the entry point and runs ten stages in order:

```
loading -> [checking for corrupted frames] -> [registering frames]
  -> pre-conditioning -> lock-in -> geometry setup -> phase referencing
  -> spatial processing -> rendering diagnostic images
  -> symmetric/antisymmetric line analysis
```

(Stages in brackets are optional, gated by `reject_outliers=True` and
`register=False` respectively.) Each stage is wrapped in the `_stage` context
manager, which prints a `"<name>..."` header immediately and a
`"done (elapsed)"` footer on exit — so a slow step shows up as a status line
rather than as silence, and `_progress()` gives the same treatment to tight
per-frame loops (frame decode, per-frame anomaly scan, registration) with a
throttled progress line (redrawn in place on an interactive terminal, one
line every few seconds when output is redirected).

The sections below follow the file's own numbered section comments.

#### Utilities

- **`_fmt_dt(seconds)`** — formats a duration as `"1m23.4s"` / `"3.2s"`.
- **`_stage`** — the per-phase progress context manager described above.
- **`_progress(i, n, t0, last, prefix)`** — throttled progress line for a
  tight loop; returns the new `last`-reported timestamp so the caller
  threads it through the loop without re-checking the clock every iteration.

#### 1. Loading

- **`load_sequence(path)`** — loads a saved export (`.npy` by default). The
  body is meant to be replaced with whatever your own export format is; nothing
  downstream cares how the frames arrived, only that they end up as a
  `(n_frames, h, w)` float array.
- **`_decode_raw_record(fff)`** — decodes one FLIR FFF record's raw pixel data.
  *Design decision:* the FFF record's own "subtype" byte is not trusted to say
  how the data is packed (it's inconsistent across camera/firmware
  combinations); instead the function sniffs the data itself — a PNG
  signature, a JPEG SOI marker, or a byte count that matches `h*w*2` exactly —
  the same approach ExifTool's FLIR.pm parser uses, since that parser has seen
  the most real-world FLIR export variants.
- **`load_csq(path)`** — loads a FLIR `.csq`/`.seq` recording via `flirpy`
  (container/calibration metadata) plus the decoding above (pixel data),
  converting to °C with the camera's own Planck-law constants. Falls back
  from per-frame timestamps to an assumed constant frame rate when the
  camera's own timestamps are too coarse (whole seconds) to resolve a fast
  recording — detected by checking whether consecutive timestamps are
  monotonically increasing.
- **`build_time_vector(n_frames, fps)`** — a plain constant-rate time axis,
  for use only when the source doesn't carry real per-frame timestamps.
- **`decimate(frames, t, target_fps)`** — block-averages frames down to a
  target rate. *Design decision:* block-averaging (not sub-sampling) doubles
  as an anti-alias filter and lowers the noise floor by `sqrt(block)` before
  the lock-in even runs, and it's what makes a 30 GB raw recording tractable
  in memory at a lock-in frequency that only needs a handful of Hz.

#### 2. Frame quality & motion registration

This section exists because two independent physical/data problems can
masquerade as a real defect signal, and each needed its own detector:

- **`reject_outlier_frames(frames, t, ...)`** — drops frames corrupted by a
  camera glitch (typically a NUC shutter recalibration event). *Design
  decision:* scores each frame by the **99.5th percentile** of
  `|frame[i] - frame[i-1]|` (not a mean) — a percentile catches a small,
  spatially localized corrupted patch without needing to move the whole
  frame's average. A frame is only flagged if **both** its neighbouring jumps
  are anomalous (checked via a MAD-based robust z-score against the whole
  recording), so a genuine persistent step (real drift, a real fast thermal
  transient) isn't mistaken for a one-frame glitch. Frames are dropped
  outright rather than interpolated, since `lockin()` is just a weighted dot
  product against whatever time vector it's given — removing a frame and its
  timestamp together keeps the `2/n` normalization consistent with no gap to
  fill. Raises rather than silently dropping more than `max_reject_frac` of
  the recording, on the theory that's more likely a systematic problem than a
  run of isolated glitches.
- **`detect_frozen_frames(frames, t, ...)`** — flags contiguous runs of
  frames suspiciously *similar* to their neighbour, the mirror-image failure
  mode `reject_outlier_frames()` is explicitly blind to (a camera or loader
  repeating stale frame data, not a jump). Uses the **median** (not a
  percentile) of the same frame-to-frame difference — a global similarity
  measure, since the concern here is "did the whole frame stop updating," not
  a localized spike. Report-only: it does not drop anything, since how much
  of a long run to discard (or whether it's a genuinely quiet period, e.g.
  after excitation was switched off) is a judgement call left to the caller.
- **`diagnose_pixel(frames, t, y, x, f_excite, path=...)`** — plots one
  pixel's raw time series with its lock-in sinusoid fit overlaid, saved to a
  PNG. Not called by `analyse()` automatically; it exists as a standalone tool
  for when a specific location's amplitude number is suspicious and you need
  to see the underlying trace (a clean sinusoid vs. a step vs. a spike) to
  tell a real signal from a data artefact.
- **`_coherent_amplitude(signal, t, f)`** — the same single-bin DFT
  `lockin()` does, applied to a 1D signal (used to summarize how much of a
  registration offset time series is coherent with the excitation, as
  opposed to a real drift or plain jitter).
- **`register_frames(frames, t, f_excite, ...)`** — sub-pixel-registers every
  frame to a common reference via phase correlation
  (`skimage.registration.phase_cross_correlation`). Several design decisions
  here, in order of how they were arrived at:
  - **`normalization=None`** rather than skimage's own default of `"phase"`:
    phase-only normalization equalizes every spatial frequency's magnitude
    before correlating, which is fine for genuinely broadband, richly
    textured images, but on smooth thermal imagery it hands high-frequency
    sensor noise equal weight to the real signal and badly underestimates
    exactly the sub-pixel shifts (a few tenths of a pixel) this function
    exists to catch. Plain cross-correlation weights each frequency by its
    actual power instead, which recovers small shifts reliably here.
  - **`reference="middle"`** (not `"first"`, the naive choice): if the part
    has a real monotonic drift (rig settling, a cold-start transient) on top
    of the oscillatory "breathing" this function targets, anchoring to frame
    0 pins the *whole* corrected sequence to one end of that drift, so once
    it's removed, every image can look shifted by the drift's full range
    relative to the raw footage. Anchoring to the temporally central frame
    instead centres the correction in that range.
  - The function decomposes the resulting offsets into a **linear drift
    trend** (fit and reported separately) and the **residual** — only the
    residual's excitation-coherent component (via `_coherent_amplitude`) is
    the actual dipole-causing artefact this function targets; a large offset
    *range* on its own conflates "the part genuinely drifted a lot" with
    "registration is misbehaving," and the two need different responses.
  - Registration is **off by default** in `analyse()` (`register=False`) —
    see the Design Decisions section below.
- **`track_part_centroid(frames, ...)`** — per-frame intensity centroid of
  the hot region, independent of `register_frames()` (no phase correlation
  involved) — used as a second, cross-checking motion signal in the
  `lockin_motion.png` diagnostic when registration is enabled.

#### 3. Pre-conditioning

*(Unmodified by design across this project's history — deliberately kept
simple and linear.)*

- **`remove_global_offsets(frames)`** — subtracts each frame's spatial median
  from itself, killing the offset steps produced by the camera's internal NUC
  shutter and any global ambient drift (both shift the *whole* image at once,
  unlike a line-shaped defect).
- **`detrend_per_pixel(frames, t)`** — removes a per-pixel linear trend in
  time (the part slowly equilibrating with the room). A ramp has energy at
  every frequency including the lock-in's, so removing it lowers the noise
  floor.
- **`trim_to_whole_cycles(frames, t, f, skip_cycles=2)`** — discards the
  initial thermal transient (the part warming to cyclic steady state) and
  truncates to an integer number of excitation periods, avoiding the
  spectral leakage a partial cycle would cause.

#### 4. The lock-in itself

*(Also deliberately unmodified — this is the mathematical core.)*

- **`lockin(frames, t, f)`** — projects every pixel's time series onto
  `sin(2*pi*f*t)` and `cos(2*pi*f*t)`: a single-bin discrete Fourier
  transform, equivalently a bandpass filter centred at `f` with bandwidth set
  by the total record length. Returns `(amplitude, phase_radians)`.
- **`noise_floor(frames, t, f, factor=1.37)`** — amplitude at a frequency
  deliberately *not* commensurate with the excitation or its harmonics
  (`f * factor`); since there's no real signal there, whatever comes back is
  the measurement noise floor in the same units as the real result.

#### 5. Spatial processing

- **`spatial_highpass(amp, sigma_px)`** — subtracts a Gaussian-blurred copy
  of the amplitude image. **Not used by `analyse()`'s default pipeline** —
  see Design Decisions — but still used internally by the automatic
  line-detection functions below, and available standalone for a part with no
  real zone-to-zone steps (only isolated line features), where it's still the
  right tool.
- **`part_mask(amp, ...)`** — binary footprint of the part: threshold at a
  fraction of the bright end of the amplitude distribution, fill holes
  (closing over the deletion lines themselves), then erode inward so the true
  part edge — which would otherwise ring against the high-pass filter — can't
  contaminate the interior.
- **Automatic line detection** (`_tls_axis`, `_hough_peak`, `_extract_segment`,
  `find_deletion_lines`, `find_deletion_line`) — a Hough-transform-based
  detector. `find_deletion_lines` is what `auto_setup()` calls, and is used
  by `analyse()`'s default pipeline as of `auto_geometry=True` (§6) — see
  Design Decisions for the risk that comes with that and how
  `lockin_line_candidates.png` mitigates it. `find_deletion_line` (the
  single-strongest-line convenience wrapper) isn't called by the pipeline
  itself, but remains available standalone:
  - `_tls_axis` fits a line to a weighted point cloud via total least squares
    (the major axis of the weighted covariance) rather than ordinary
    y-on-x regression, so it doesn't blow up on a near-vertical line.
  - `_hough_peak` is what makes multi-line parts work at all: fitting a
    single axis to *all* ridge pixels at once averages several lines
    together into an axis on none of them, while a Hough accumulator lets
    each line vote into its own `(theta, rho)` bin, keeping them separate.
    Votes are weighted by lock-in amplitude, so the winner is the line
    carrying the most signal.
  - `_extract_segment` turns one Hough peak into concrete endpoints:
    refits on its own inliers, then finds where the line actually starts
    and stops by binning inliers along the axis and keeping amplitude-
    weighted-strong bins, joining bins separated by a small gap (a
    deletion line can legitimately go quiet over part of its length — a
    single discrete bridge defect, say — without that being evidence of
    two separate lines).
  - `find_deletion_lines` repeatedly extracts the strongest remaining line
    and *erases* its axis from the candidate pool before searching again —
    erasing (not just excluding the segment) is what keeps multiple
    near-parallel lines from being rediscovered as fragments of each
    other.
  - `find_deletion_line` is the convenience wrapper returning just the
    single strongest line (optionally with the full ranked list too).
- **`_rasterize_segment(mask, p0, p1)`** / **`auto_fiducial_roi(part, lines, ...)`**
  — automatic placement of the fiducial/phase-reference ROI in the part's
  quiet zone, farthest from every detected line *and* from the part's own
  boundary (a scalloped edge or busbar is exactly where motion artefacts and
  emissivity variation are worst). Driven automatically by `auto_setup()`,
  and used by `analyse()`'s default pipeline for the same reason as
  automatic line detection above (§6); still available standalone, and
  still what a manually-clicked config falls back to only if you choose
  `auto_geometry=False`. Two specific correctness details worth
  noting: the "farthest point" search
  averages the whole near-max **plateau** rather than taking the first tied
  pixel (a naive `argmax` would bias the ROI to one edge of a quiet region
  instead of its centre), and the edge-clearance distance is *not* derived
  from the same length scale as the line-exclusion distance — a busbar or
  scalloped region is a physically different, typically much bigger feature
  than the thermal diffusion length, so conflating the two under-protects the
  ROI at a coarse mm/px.
- **`pick_line_endpoints(amp)`** — a minimal interactive click-two-points
  helper, superseded for normal use by the fuller `interactive_setup()` in
  §6, kept as a lightweight standalone option.
- **`cross_line_profile(image, p0, p1, ...)`** — the original
  nearest-neighbour cross-line profile (rounds every sample to its nearest
  pixel). Superseded by `slanted_edge_profile()` for the main pipeline (see
  §7 and Design Decisions), but still available and still what
  `along_line_profile` internally mirrors the sampling geometry of.
- **`along_line_profile(image, p0, p1, ...)`** — integrates a narrow band
  centred on the line as a function of position *along* it, for reading off
  the failure-mode shapes documented in the "WHAT THE OUTPUT MEANS" trailer
  comment at the bottom of the file (smooth hump vs. localized spike vs. flat
  elevation). Not called by `analyse()` by default; call it directly on `amp`
  for a chosen line if you want this view.

#### Sub-pixel geometry helpers

- **`_line_angle_deg(p0, p1)`** — a line's angle from horizontal, folded into
  `[0, 180)`.
- **`check_line_angle(p0, p1, bin_px, critical_tol_deg=3.0)`** — warns
  (without raising) when a line's angle is within `critical_tol_deg` of 0, 45,
  or 90 degrees. *Design decision:* the sub-pixel gain in
  `slanted_edge_profile()` comes from phase diversity — a tilted line crosses
  the pixel grid at a different sub-pixel offset at every position along it.
  At exactly 0/90 degrees that diversity vanishes entirely: moving along an
  exactly horizontal line never changes which row you're sampling, so every
  along-line position reads the identical sub-pixel row phase and pooling
  only reduces noise, not resolution. Near 45 degrees the along-line steps
  advance both axes in lockstep, a known degenerate case for the same kind of
  phase-diversity argument in slant-edge imaging practice. The function
  prints an estimate of the bin size actually achievable and lets the caller
  continue regardless — this is a warning about trustworthiness, not a hard
  failure.
- **`slanted_edge_profile(image, p0, p1, half_width_px, bin_px=0.2, n_along=800)`**
  — the sub-pixel cross-line sampler that replaced `cross_line_profile()` in
  the default pipeline. Samples a dense `(n_bins x n_along)` grid — offsets
  from `-half_width_px` to `+half_width_px` in `bin_px` steps, `n_along`
  positions along the line — using **continuous bilinear interpolation**
  (`scipy.ndimage.map_coordinates`, `order=1`), with no rounding anywhere in
  the sampling path, then averages along the line at each offset. See Design
  Decisions for why this exists.

#### 6. Geometry selection (interactive + persisted JSON config)

- **`auto_setup(amp, f_excite, mm_per_px, n_lines=1, ...)`** — what
  `analyse()` uses by default (`auto_geometry=True`): runs
  `find_deletion_lines()` (on a spatially high-passed copy of the amplitude
  image, at `highpass_mu_factor` — default 2.5 — times the diffusion length)
  and `auto_fiducial_roi()` to build a geometry config with no clicking and
  no interactive backend required — a full run can go headless end to end.
  Requires `mm_per_px` up front, since there's no calibration click in this
  path. Returns `(config, candidates, n_selected)`: `candidates` is the
  *full* ranked list `find_deletion_lines()` found (not just the `n_lines`
  kept), which `plot_line_candidates()` visualizes. Every line comes back
  untagged as a reference — there's no human judgement call being made here
  — so tag one by hand in `roi_config.json` afterwards if the summary table
  should reference against it. See Design Decisions for the risk that comes
  with defaulting to this and how the diagnostic below mitigates it.
- **`interactive_setup(image, n_lines=1, calibrate=False)`** — the manual
  fallback, for when `auto_setup()`'s pick is wrong (check
  `lockin_line_candidates.png`) or `mm_per_px` isn't known yet. One
  click-through session (via `matplotlib`'s `ginput`) on a displayed frame
  that defines every piece of geometry the pipeline needs: `n_lines`
  deletion lines (two endpoint clicks each, with the line's angle computed
  and printed immediately, plus a console prompt for whether it's a
  "reference" — believed-sound — line), one fiducial/phase-reference ROI
  (two opposite-corner clicks, explicitly guided to clean mid-zone coating
  away from every line and the part edge), and optionally a calibration pair
  (two points spanning a known physical distance, prompted for at the
  console) to compute `mm_per_px` directly rather than hard-coding it. Each
  selection is confirmed visually before moving to the next. Requires an
  interactive backend — will not work headless. Used by `analyse()` when
  `auto_geometry=False`.
- **`plot_line_candidates(amp, part, candidates, n_selected, fiducial_roi, path="lockin_line_candidates.png")`**
  — the diagnostic companion to `auto_setup()`: every candidate line found,
  drawn on the masked amplitude image, faint red and unlabelled for the
  ones *not* kept and solid green for the ones that were, plus the
  auto-placed fiducial ROI box. Deliberately a **separate figure** from
  `lockin_images.png` rather than added to it — `lockin_images.png`'s own
  line-overlay omission (§ rendering diagnostic images) was about
  decluttering routine per-run output where the geometry is already known
  and trusted; here the whole point is reviewing an *automatic* pick before
  trusting it, so the overlay — runners-up included, kept visible even
  though faint — is the diagnostic, not clutter. Called automatically by
  `analyse()` whenever `auto_geometry=True`.
- **`save_roi_config(config, path="roi_config.json")`** /
  **`load_roi_config(path="roi_config.json")`** — persist/reload the
  geometry (and, once `analyse()` has run once, the processing parameters —
  see §5 of Design Decisions) as JSON, so a recording only needs to be
  clicked through once.

#### 7. Symmetric / antisymmetric line analysis

The core of what makes this pipeline able to separate a real leakage bump
from the part's normal zone-to-zone structure:

- **`symmetric_antisymmetric_profile(image, p0, p1, mu_px, mm_per_px, ...)`**
  — decomposes the sub-pixel cross-line profile of `image` about a deletion
  line into
  ```
  sym  = (right + mirrored_left) / 2   (even about the centre)
  anti = (right - mirrored_left) / 2   (odd about the centre)
  ```
  `sym` is the candidate leakage signal; `anti` is the zone-to-zone step. Two
  design decisions worth calling out:
  - **Centring.** The given line position is treated as only an initial
    estimate, refined by finding where the profile is **steepest** near that
    estimate (the peak of `|d profile / d offset|`), not by finding where the
    raw profile crosses the midpoint between its two wing levels. The
    midpoint-crossing approach reads back the step's own structure correctly
    in isolation, but a real leakage bump at the same location adds a
    symmetric offset to the raw profile there too, which shifts a level
    crossing away from the true centre — a symmetric bump's slope is exactly
    zero at its own centre by construction, so the steepest-point method is
    unbiased by however large a co-located symmetric signal happens to be.
    Because bilinear interpolation is piecewise-linear between native pixel
    columns, the steepest point is usually a short *plateau* of tied gradient
    values (one native-pixel interval) rather than a single isolated peak;
    taking `argmax` alone would bias the estimate toward whichever end of
    that plateau comes first in scan order, so the whole tied plateau is
    averaged instead (the same fix used, for the same reason, in
    `auto_fiducial_roi()`'s tie-breaking). The line is then shifted along its
    own normal by that amount and resampled **once** — not iterated to
    convergence, per the physical picture that a single clean step has one
    well-defined location.
  - **Physical plausibility.** The function also computes the FWHM of the
    `sym` peak (via a linearly-interpolated half-max crossing), returned for
    the caller to compare against the ply thickness and diffusion length —
    see `analyse_deletion_line()` below and Design Decisions.
- **`analyse_deletion_line(amp, phase, p0, p1, mu_mm, mm_per_px, ply_thickness_mm=3.0, ...)`**
  — the full per-line analysis: checks the line's angle first
  (`check_line_angle`), recentres once on the **amplitude** channel (higher
  SNR on the step than phase usually gives) via
  `symmetric_antisymmetric_profile`, then reuses that corrected line —
  unchanged — for the **phase** channel, so amplitude and phase are decomposed
  about the same physical point and stay directly comparable rather than each
  finding a slightly different centre from its own noise. Produces a verdict
  by comparing the amplitude channel's FWHM against `ply_thickness_mm` (too
  narrow) and `5 * mu_mm` (too wide, i.e. likely zone-scale structure rather
  than a compact defect) — see Design Decisions for the physical reasoning.
- **`print_line_summary(results, is_reference=None)`** — the per-line summary
  table: sym peak, wing RMS, their ratio, and the physical width/verdict for
  both amplitude and phase, plus — when any line is tagged as reference — each
  line's amplitude peak as a ratio to the mean of the reference lines' peaks,
  so an elevated line stands out as a number rather than something read off a
  plot by eye.

#### 8. Pipeline

- **`analyse(path, fps, f_excite, ...)`** — orchestrates all of the above.
  See its docstring in the file for the complete parameter list; the Design
  Decisions section below covers the choices behind the defaults. Returns
  `(amp, phase, nf, results)`, where `results` is one dict per line (in
  `roi_config["lines"]` order) in the shape `analyse_deletion_line()`
  produces.

### Output files

| File | Produced when | Contents |
|---|---|---|
| `roi_config.json` | Always (geometry setup stage) | Line endpoints/angles/reference tags, fiducial ROI, `mm_per_px`, and the processing parameters used for this run. |
| `lockin_images.png` | Always | Three panels: raw amplitude, masked amplitude, masked phase (degrees), with the fiducial ROI overlaid. |
| `lockin_line_profiles.png` | Always | One row per line, sym/anti curves for amplitude and phase, with the diffusion-length band shaded and the run's processing parameters stamped at the bottom. |
| `lockin_motion.png` | Only if `register=True` | Registration offset vs. time, and part centroid before/after registration (an independent motion check). |
| `lockin_line_candidates.png` | Whenever `auto_geometry=True` (the default) and `use_saved_config=False` | Every line candidate `find_deletion_lines()` found (faint red) with the `n_lines` actually selected (solid green) and the auto-placed fiducial ROI, overlaid on the masked amplitude image — review this before trusting an automatic pick. |
| `lockin_pixel_diag.png` | Only if `diagnose_pixel()` called manually | One pixel's raw trace with its lock-in fit overlaid. |

### Design decisions

A few choices in this script came directly from real failure modes observed
on real data during development, not just from first-principles reasoning —
worth calling out specifically:

1. **Registration defaults to off (`register=False`).** Sub-pixel
   registration exists because a part that moves coherently with the
   excitation (thermal expansion against its fixture) produces a dipole —
   positive on one side of a line, negative on the other — that lock-in
   amplifies exactly as it would real signal. But `phase_cross_correlation`
   can occasionally lock onto a spurious large "shift" on quiet, low-texture
   data even when the true motion is zero, which does more damage than the
   motion it was meant to fix; this was observed directly (a ~120 px spurious
   spike on synthetic data with zero injected motion). For a low-CTE part
   (glass), genuine motion is expected to be a small fraction of a pixel —
   below what registration can usefully resolve anyway — so the expected
   value of turning it on is negative for that material. It remains available
   and is the right choice for a part whose material plausibly moves a
   measurable fraction of a pixel.
2. **`spatial_highpass()` is not in the default pipeline.** On a part where
   adjacent zones have a genuine physical step in power density, subtracting
   a blurred copy of the image turns that step into a spurious antisymmetric
   dipole (a Gaussian blur of a step is itself a smoothed step, and the
   difference between a sharp and a smoothed step is approximately the
   step's own derivative — an odd function about its centre) — which can
   swamp the much smaller symmetric leakage signal entirely.
   `symmetric_antisymmetric_profile()` replaces it by decomposing the step
   and the candidate signal directly, rather than filtering first.
3. **Sub-pixel sampling (`slanted_edge_profile`) rather than
   nearest-neighbour.** The heat source is imaged through the full glass ply,
   so any real signature is blurred by at least the ply thickness before it
   reaches the surface — at typical mm/px sampling that blurred feature can
   be sub-pixel wide, and rounding every sample to its nearest pixel
   (`cross_line_profile`'s approach) turns that into single-sample spiking
   artefacts that look like signal but are quantization noise. Pooling many
   along-line crossings at different sub-pixel phases (viable only because
   the line is tilted relative to the pixel grid — see `check_line_angle`)
   recovers genuinely finer effective resolution.
4. **Geometry defaults to automatic detection (`auto_geometry=True`), paired
   with a mandatory review image.** The Hough-based line detector (§5) was
   built around finding the strongest sharp ridge in a high-passed image —
   exactly the signal a genuine zone-to-zone step also produces, which is
   *not* what should be selected, so an automatic pick always carries some
   risk of locking onto the step instead of the real line. That risk is
   accepted in exchange for headless, no-clicking operation on the common
   case, on the condition that it's never silent: `lockin_line_candidates.png`
   is produced on every automatic run specifically so a human can see every
   candidate the detector considered (faint) against the one(s) it picked
   (solid) and catch a wrong pick before trusting the run's results — this is
   the same trade `find_deletion_line()`'s own printed runner-up margin makes
   in text form. When the plot shows the wrong pick, or `mm_per_px` isn't
   known yet (automatic mode has no calibration click), fall back to
   `interactive_setup()` via `analyse(auto_geometry=False)` — full manual
   control, still available, just no longer the default.
5. **Processing parameters are recorded and checked, not just used.**
   `half_width_mm`, `bin_px`, `n_along`, `f_excite`, `mm_per_px`, and
   `ply_thickness_mm` are stamped into `roi_config.json` and onto every
   figure; re-analysing the same config with different values prints an
   explicit warning naming every parameter that changed. Different window
   widths across runs make their figures/results incomparable, and this is
   meant to catch that before it's noticed only by eye.
6. **Physical plausibility, not just statistical significance.** A peak's
   ratio to the wing noise floor answers "is this distinguishable from
   noise"; it does not answer "is this physically possible." The FWHM
   verdict (narrower than the ply thickness → surface artefact; wider than
   ~5 diffusion lengths → zone-scale structure, not a compact defect)
   answers the second question independently of the first — a residual
   mis-centring artefact can produce a modestly-elevated ratio while being
   far too broad to be a real localized defect, and the width check catches
   that where the ratio alone would not (observed directly during testing:
   a clean, defect-free line showed ratio 3.08 — arguably concerning in
   isolation — but a 7.3 mm width against a ~1.3 mm diffusion length
   correctly flagged it as not a real line defect).
7. **Frame-quality detectors are separate because their failure signatures
   are opposite.** `reject_outlier_frames()` catches a frame that jumps far
   from its neighbours (a percentile-based, single-frame test);
   `detect_frozen_frames()` catches a run of frames that stay suspiciously
   *close* to their neighbours (a median-based, run-length test). Neither
   detects what the other is built for — a corrupted-then-repeated frame can
   pass the outlier check cleanly while still being wrong.
8. **Phase is documented as the preferred detection channel**, and is
   decomposed the same symmetric/antisymmetric way as amplitude, side by
   side, for that reason — amplitude scales with emissivity, view angle, and
   reflected radiation, while phase is a ratio of two quantities that scale
   identically, so those multiplicative nuisances cancel. See the "WHAT THE
   OUTPUT MEANS" comment block at the bottom of the file for the full
   physical interpretation notes (including along-line profile shape
   diagnostics, and the recommended null test with the supply off).
9. **The phase panel in `lockin_images.png` re-centres its own +-180 deg
   wraparound point before display.** `phase` is referenced to the
   fiducial ROI (0 deg there) once, upstream, and stays that way
   everywhere else in the pipeline. But `np.angle()`'s wraparound point is
   fixed at +-180 deg *from that reference* — if the part's real phase
   spread happens to reach anywhere near there (a real, physical
   zone-to-zone phase difference, not an error), the wrap falls INSIDE the
   occupied data instead of in the empty part of the circle, and a
   perfectly continuous physical quantity shows up in the figure as a
   sharp, unphysical colour seam across part of the part (observed
   directly on real data). The panel now re-wraps around the *circular
   mean* of the part's own masked phase instead of around the fiducial
   reference — for the roughly unimodal distribution expected here (one
   coherent physical process across the part), that puts the branch cut
   in the least-populated part of the circle, as far from the real data as
   the distribution allows. This is display-only: the numeric `phase`
   array used by the per-line symmetric/antisymmetric analysis is
   unaffected.

### Quick start

```python
from lockin_thermography import analyse

# First run on a new recording: auto_geometry=True (the default) detects
# geometry automatically -- no clicking, no interactive backend needed.
# mm_per_px is required up front (no calibration click in this path).
# ALWAYS review lockin_line_candidates.png afterwards, before trusting the
# result -- see Design Decisions #4.
analyse(
    path="FLIR0022.csq",
    fps=30.0,
    f_excite=0.1,        # must match F_MOD in psu_control.py
    mm_per_px=2.23,
    n_lines=3,
    use_saved_config=False,
)

# Every later run of the SAME recording: reload the saved geometry and
# processing parameters instead of re-detecting -- also works headless.
analyse(
    path="FLIR0022.csq",
    fps=30.0,
    f_excite=0.1,
    use_saved_config=True,
)

# When the candidates plot shows the wrong pick, or mm_per_px isn't known
# yet: fall back to clicking through geometry by hand (n_lines deletion
# lines, fiducial ROI, and a calibration pair since mm_per_px is left
# unset here) with auto_geometry=False.
analyse(
    path="FLIR0023.csq",
    fps=30.0,
    f_excite=0.1,
    n_lines=3,
    use_saved_config=False,
    auto_geometry=False,
)
```

### Dependencies

`numpy`, `scipy`, `matplotlib`, `scikit-image` (for `register_frames()`).
Optional: `flirpy`, `pillow`, `pylibjpeg`, `pylibjpeg-libjpeg` (for
`load_csq()` — direct `.csq`/`.seq` import).

---

## 3. Tests

Nine standalone test scripts (run with `python <file>.py`; each prints
`PASS`/`FAIL` per check and an `N/M passed` summary — there is no pytest
harness, by design, matching the rest of the project's plain-script style)
plus two end-to-end smoke tests. All were written to reproduce a specific,
previously-observed failure mode on synthetic data with a known ground
truth, not just to exercise code paths generically.

| File | Checks | Cases |
|---|---|---|
| `test_find_line.py` | `find_deletion_line()` / `find_deletion_lines()` pick exactly one real line and rank correctly among several: parallel lines of different brightness, crossing lines, a line with a gap vs. a dim solid line, near-axis-aligned vs. near-vertical lines, a line vs. a bright compact blob, and that two equal-strength lines are returned as two separate lines rather than one averaged (meaningless) axis between them. | 6/6 |
| `test_auto_fiducial.py` | `auto_fiducial_roi()` lands between two lines, clears a single line by the requested exclusion distance, falls back to the part's most interior point when there are no lines, raises on an empty part mask, scales its box size with `exclusion_px` (not a flat pixel cap), and — the regression case — clears the part's physical edge by a real margin even when `exclusion_px` is tiny (the original reported bug: edge clearance defaulting to the thermal-diffusion-length scale, not the physically distinct busbar/edge scale). | 7/7 |
| `test_registration.py` | Proves the actual mechanism behind the "dipole instead of a symmetric bump" symptom: a static, high-contrast ridge that moves coherently with the excitation produces a dipole (antiphase flanks around a null at the ridge centre) under lock-in, not the symmetric bump a real heat source gives — and that `register_frames()` removes it, recovering the injected sub-pixel motion accurately. | 3/3 |
| `test_registration_drift.py` | Reproduces the reported symptom directly: with a real monotonic drift (~20 px) on top of oscillatory motion (~0.35 px), registering to frame 0 anchors the whole corrected sequence to one end of that drift; `reference="middle"` roughly halves the net apparent shift. | 1/1 |
| `test_outlier_frames.py` | Reproduces the "pasted patch" artefact: a single frame with a localized corrupted region, invisible in raw amplitude but sharp after spatial high-pass. Confirms `reject_outlier_frames()` has zero false positives on clean data, detects and drops exactly the corrupted frame, and that doing so collapses the artefact. | 3/3 |
| `test_frozen_frames.py` | Confirms `detect_frozen_frames()` has zero false positives on clean data, correctly detects an injected run of duplicated frames, that `reject_outlier_frames()` does *not* catch the same run (proving the two detectors cover genuinely different failure modes), and a smoke check that `diagnose_pixel()` produces a non-empty plot file. | 4/4 |
| `test_sym_anti.py` | Checks `symmetric_antisymmetric_profile()` against a synthetic field with a known step and a known symmetric bump, on a genuinely tilted line with a matching rotated field (not axis-aligned, since that's the degenerate case `check_line_angle()` exists to flag): a pure step decomposes to `sym ~= 0`; a step+bump at the exact true centre recovers the bump amplitude in `sym` and the step in `anti`; a line deliberately mis-given by 3 px is corrected by recentring; recentring is measurably more accurate against the known ground truth than not recentring; and `analyse_deletion_line()` shares one recentred line — and correctly reports its angle — across the amplitude and phase channels. | 10/10 |
| `test_slanted_edge.py` | Demonstrates the actual sub-pixel resolution gain: on a transition narrower than one native pixel, `slanted_edge_profile()` recovers the true 10-90% rise width meaningfully more accurately than nearest-neighbour `cross_line_profile()` does. Also checks `check_line_angle()` fires for a range of angles at/near 0, 45, and 90 degrees and stays quiet for a spread of well-tilted angles. | 14/14 |
| `test_roi_config.py` | `save_roi_config()`/`load_roi_config()` JSON roundtrip: exact roundtrip of a multi-line config, per-line reference tag and angle preserved, and processing parameters preserved. | 5/5 |
| `test_pipeline_smoke.py` | End-to-end `analyse()` run (fully headless, via a hand-built `roi_config`) on a synthetic sequence with two deletion lines splitting genuinely different power-density zones and a leakage bump on only one of them: confirms the defect line's peak and ratio clearly separate from the clean line's, the clean line correctly reads as "not a line defect" via the width check, all expected output files are produced, an inverted/empty fiducial ROI is rejected with a clear error, and background-masked phase spread is much tighter than the unmasked full-frame spread (confirming the masking in the diagnostic figure is doing real work). | assertions, no numbered count |
| `test_auto_geometry.py` | End-to-end `analyse(auto_geometry=True)` run (no `roi_config`, no clicking) on the same two-zone synthetic sequence as the smoke test above: confirms `auto_setup()` finds the requested number of lines and places a fiducial ROI with no geometry supplied up front, and that every expected output file — including `lockin_line_candidates.png` — is produced. Not a claim that auto-detection picks the *intended* line (it can legitimately lock onto the scene's own zone-boundary step instead — see Design Decisions #4); the point is proving the headless wiring runs end to end and that the candidates review image exists to catch exactly that failure mode. | assertions, no numbered count |

### Running the tests

Each file is self-contained:

```bash
python test_find_line.py
python test_pipeline_smoke.py
# ...etc
```

There is no single "run everything" entry point in the repo; run the files
you care about, or loop over all of them:

```bash
for f in test_*.py; do echo "=== $f ==="; python "$f"; done
```

### Why plain scripts instead of a test framework

Consistent with the rest of the project (`psu_control.py` and
`lockin_thermography.py`'s own `__main__` block are both plain, run-directly
scripts), the tests avoid a pytest/unittest dependency in favour of scripts
that print their own pass/fail summary and can be run with nothing but the
interpreter. Each one builds its synthetic ground truth inline (a known step,
a known bump, a known injected motion) rather than using fixtures, so the
physical claim being tested and the assertion checking it sit next to each
other in the same file.
