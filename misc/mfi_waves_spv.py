#!/usr/bin/env python3
"""
COSMOS visibility from Paranal (VISTA), over SPV.

Conditions per night:
- Start after evening astronomical twilight end
- End at morning astronomical twilight start
- Target "visible" if altitude > 30 deg
- Case A: moon illumination fraction < 0.1
- Case B: moon illumination >= 0.1 AND moon altitude < 0 deg

Requires: astropy, astroplan, numpy
Install:
  pip install astropy astroplan numpy
"""

from __future__ import annotations

import argparse
import numpy as np

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation

from astroplan import Observer
from astroplan.moon import moon_illumination

def count_continuous_blocks(mask: np.ndarray, step_min: float, block_min: float = 20.0) -> float:
    """
    Count time (in hours) only in FULL `block_min`-minute blocks where `mask` is
    continuously True.

    Example: if mask is True for 34 minutes continuously, counts 20 minutes.
             if mask is True for 41 minutes continuously, counts 40 minutes.
    """
    if mask.size == 0:
        return 0.0

    step_min = float(step_min)
    block_min = float(block_min)
    if step_min <= 0 or block_min <= 0:
        return 0.0

    # Require exact step divisibility? Not strictly necessary, but this keeps it clean.
    # If not divisible, we still do it in time-space, not step-space.
    block_steps = int(np.round(block_min / step_min))
    if block_steps <= 0:
        return 0.0

    # Find contiguous True run lengths (in samples)
    m = np.asarray(mask, dtype=bool)

    # Pad with False at both ends so we can detect run boundaries via diff
    padded = np.concatenate(([False], m, [False]))
    d = np.diff(padded.astype(np.int8))

    run_starts = np.where(d == 1)[0]   # index in original m where a True-run starts
    run_ends   = np.where(d == -1)[0]  # index where it ends (exclusive)

    run_lengths = run_ends - run_starts  # lengths in samples

    # Full blocks per run
    full_blocks = np.sum(run_lengths // block_steps)

    # Convert to hours
    return full_blocks * (block_min / 60.0)

def build_observer_paranal() -> Observer:
    # ESO Paranal coords ~ 24°40' S, 70°25' W :contentReference[oaicite:2]{index=2}
    # Use decimal degrees with Paranal altitude ~2635 m (common Paranal value; altitude doesn't affect twilight much).
    loc = EarthLocation.from_geodetic(lon=-70.4167 * u.deg, lat=-24.6667 * u.deg, height=2635 * u.m)
    return Observer(location=loc, name="Paranal (VISTA)", timezone="America/Santiago")


def cosmos_target() -> SkyCoord:
    # COSMOS center (J2000): RA 150.11916667 deg, Dec +2.20583333 deg :contentReference[oaicite:3]{index=3}
    return SkyCoord(ra=150.11916667 * u.deg, dec=2.20583333 * u.deg, frame="icrs")


def time_grid(t_start: Time, t_end: Time, step_minutes: float) -> Time:
    if t_end <= t_start:
        return Time([], format="jd", scale=t_start.scale)
    step = step_minutes * u.min
    n = int(np.floor(((t_end - t_start).to(u.min).value) / step_minutes)) + 1
    return t_start + TimeDelta(np.arange(n) * step.to(u.day).value, format="jd")


def compute_night_window(observer: Observer, anchor: Time) -> tuple[Time, Time] | None:
    """
    Returns (window_start, window_end) for the night whose midnight is the next midnight after 'anchor'.
    Window:
      evening astro twilight end  ->  (morning astro twilight start - 1 hour)
    """
    mid = observer.midnight(anchor, which="next")

    eve_astro_end = observer.twilight_evening_astronomical(mid, which="previous")
    morn_astro_start = observer.twilight_morning_astronomical(mid, which="next")

    window_start = eve_astro_end
    window_end = morn_astro_start #- 1.0 * u.hour

    if window_end <= window_start:
        return None
    return window_start, window_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--start",
        default=None,
        help="Start time (UTC) in ISO format. Default: now + 7 days. Example: 2026-03-03 00:00:00",
    )
    ap.add_argument("--weeks", type=float, default=8.0, help="Number of weeks to evaluate (default 8).")
    ap.add_argument("--step-min", type=float, default=2.0, help="Time grid step in minutes (default 2).")
    ap.add_argument(
        "--alt-min-deg",
        type=float,
        default=50.0,
        help="Minimum target altitude in degrees to count as visible (default 0).",
    )
    ap.add_argument(
        "--print-nightly",
        action="store_true",
        help="Print per-night breakdown.",
    )
    ap.add_argument("--block-min", type=float, default=20.0, help="Continuous block size in minutes (default 20).")
    args = ap.parse_args()

    observer = build_observer_paranal()
    target = cosmos_target()

    # Start "Tomorrow" => now + 7 days (unless user overrides).
    start = Time(args.start, scale="utc") if args.start else Time.now().utc #+ 7.0 * u.day
    ndays = int(np.ceil(args.weeks * 7.0))

    alt_min = args.alt_min_deg * u.deg

    total_caseA_hours = 0.0
    total_caseB_hours = 0.0
    total_caseC_hours = 0.0
    total_open_hours = 0.0  # just for context: total hours in the allowed nightly window with target up

    for i in range(ndays):
        anchor = start + i * u.day
        win = compute_night_window(observer, anchor)
        if win is None:
            continue
        t0, t1 = win

        tt = time_grid(t0, t1, args.step_min)
        if len(tt) == 0:
            continue

        # Target altitude
        targ_alt = observer.altaz(tt, target).alt
        target_up = targ_alt > alt_min

# Moon altitude + illumination
        moon_alt = observer.moon_altaz(tt).alt
        moon_18down = moon_alt < (-18) * u.deg

        mfi = moon_illumination(tt)  # fraction in [0,1]

        # Cases (note strict inequalities per your spec)
        caseA_mask = (mfi < 0.1)
        caseB_mask = (mfi > 0.1) & (mfi < 0.58) & moon_18down
        caseC_mask = (mfi > 0.58) & moon_18down

        # Count minutes using boolean masks
        # Count ONLY full continuous blocks (default: 20 min)
        block_min = args.block_min

        open_hours = count_continuous_blocks(target_up, args.step_min, block_min=block_min)

        caseA_hours = count_continuous_blocks(target_up & caseA_mask, args.step_min, block_min=block_min)
        caseB_hours = count_continuous_blocks(target_up & caseB_mask, args.step_min, block_min=block_min)
        caseC_hours = count_continuous_blocks(target_up & caseC_mask, args.step_min, block_min=block_min)

        total_open_hours += open_hours
        total_caseA_hours += caseA_hours
        total_caseB_hours += caseB_hours
        total_caseC_hours += caseC_hours

        if args.print_nightly:
            # Use date label from the midnight in Chile local time for readability
            mid = observer.midnight(anchor, which="next")
            date_str = mid.isot.split("T")[0]
            # Print UTC timestamps to avoid timezone dependency
            print(
                f"{date_str}: "
                f"Hours up={open_hours:6.2f}, "
                f"Dark={caseA_hours:6.2f}, "
                f"Grey & Moon Down={caseB_hours:6.2f}, "
                f"Bright & Moon Down={caseC_hours:6.2f}"
            )

    print("\n=== Summary ===")
    print(f"Start (UTC): {start.isot}")
    print(f"Target: COSMOS center (RA={target.ra.deg:.3f} deg, Dec={target.dec.deg:.3f} deg)")
    print(f"Days evaluated: {ndays}  (weeks={args.weeks})")
    print(f"Time step: {args.step_min} min")
    print(f"Min target altitude: {args.alt_min_deg} deg")
    print("")
    print(f"Total 'target up' hours in allowed nightly windows: {total_open_hours:.2f} h")
    print(f"Dark (moon illum < 0.1): {total_caseA_hours:.2f} h")
    print(f"Grey (0.58 > moon illum >= 0.1 but moon 18[deg] below horizon): {total_caseB_hours:.2f} h")
    print(f"Bright/Superbright (moon illum >= 0.58 but moon 18[deg] below horizon): {total_caseC_hours:.2f} h")
    print(f"Total dark+grey+bright+ hours: {total_caseA_hours + total_caseB_hours + total_caseC_hours:.2f} h")


if __name__ == "__main__":
    main()