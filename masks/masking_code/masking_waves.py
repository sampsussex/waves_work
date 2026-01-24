"""
Sky masking pipeline using BallTree + Numba (class-based)

Inputs (1–4 are CSV; 5 is Parquet):
  1) stars_path   : CSV with columns [ra, dec, radius]
  2) ghosts_path  : CSV with columns [ra, dec, radius]
  3) ngc_path     : CSV with columns [poly_id, vertices]
                    where `vertices` is a semicolon-separated list of "ra dec" pairs
                    e.g. "10.1 2.0; 11.0 2.2; 10.7 1.8" (first/last may be same or not)
  4) extras_path  : CSV with columns [ra, dec, radius]
  5) (optional) input_catalog_path: Parquet with columns [ra, dec]; if omitted, a random catalog is generated.

Behavior:
  • If input catalog (5) is absent, fills a rectangular sky region with N random points uniformly within
    [ra_min, ra_max] × [dec_min, dec_max] (degrees), using sin(dec) sampling.
  • Builds a BallTree (sklearn) on the main catalog in (dec, ra) radians with the haversine metric.
  • For each of the radial mask CSVs (stars, ghosts, extras), performs a radius query and marks all returned
    main points as masked. No post-refinement — exactly as requested.
  • For NGC polygons: computes a conservative spherical query radius per polygon (Numba), then uses BallTree to
    preselect candidate main points, followed by a fast Numba point-in-polygon test in lon/lat degrees.
  • Appends boolean columns: mask_star, mask_ghost, mask_ngc, mask_extra.
  • Saves the augmented catalog (Parquet/CSV inferred by output extension).

Speed:
  • Heavy lifting for radial searches is in BallTree C/NumPy code.
  • Numba-accelerated geometry (polygon bounding radius, point-in-polygon, haversine, random dec sampling).
  • Processes queries in batches to control memory; minimal Python overhead.

Example:
  python sky_masking.py \
      --stars stars.csv \
      --ghosts ghosts.csv \
      --ngc ngc.csv \
      --extras extras.csv \
      --radius-unit arcsec \
      --ra-min 150 --ra-max 160 --dec-min -1 --dec-max 2 \
      --n-rand 5000000 \
      --out masked_catalog.parquet

Requirements:
  pip install numpy pandas pyarrow numba scikit-learn
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from numba import njit, prange

# ----------------------------
# Numba geometry helpers
# ----------------------------

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

@njit(cache=True)
def deg2rad(a):
    return a * DEG2RAD

@njit(cache=True)
def rad2deg(a):
    return a * RAD2DEG

@njit(cache=True)
def normalize_ra_deg(ra):
    r = ra % 360.0
    if r < 0:
        r += 360.0
    return r

@njit(cache=True)
def sin_random_dec(n, dec_min_deg, dec_max_deg, rng_seed):
    # uniform in sin(dec)
    smin = math.sin(dec_min_deg * DEG2RAD)
    smax = math.sin(dec_max_deg * DEG2RAD)
    out = np.empty(n, dtype=np.float64)
    # Simple LCG for speed/numba (statistically fine for this use); or use np.random.Generator outside Numba.
    a = 1664525
    c = 1013904223
    m = 2**32
    state = np.uint32(rng_seed)
    for i in range(n):
        state = np.uint32((a * state + c) % m)
        u = (state / float(m))
        s = smin + (smax - smin) * u
        out[i] = math.asin(max(-1.0, min(1.0, s))) * RAD2DEG
    return out

@njit(cache=True)
def haversine_deg(ra1, dec1, ra2, dec2):
    dlon = (ra2 - ra1) * DEG2RAD
    dlat = (dec2 - dec1) * DEG2RAD
    a = (math.sin(dlat/2.0)**2
         + math.cos(dec1*DEG2RAD) * math.cos(dec2*DEG2RAD) * math.sin(dlon/2.0)**2)
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return c * RAD2DEG

@njit(cache=True)
def polygon_center_and_radius_deg(ra_deg, dec_deg):
    """Compute an approximate spherical center and a conservative bounding radius for a polygon.
    Center: mean on unit sphere (normalized). Radius: max great-circle distance from center to vertices (deg),
    inflated slightly for safety.
    """
    n = ra_deg.size
    # mean on unit sphere
    sx = 0.0; sy = 0.0; sz = 0.0
    for i in range(n):
        lon = ra_deg[i] * DEG2RAD
        lat = dec_deg[i] * DEG2RAD
        cl = math.cos(lat)
        sx += cl * math.cos(lon)
        sy += cl * math.sin(lon)
        sz += math.sin(lat)
    sx /= n; sy /= n; sz /= n
    # normalize
    r = math.sqrt(sx*sx + sy*sy + sz*sz) + 1e-30
    sx /= r; sy /= r; sz /= r
    # center lon/lat
    cen_lat = math.atan2(sz, math.sqrt(max(0.0, 1.0 - sz*sz)))
    cen_lon = math.atan2(sy, sx)
    cen_ra = cen_lon * RAD2DEG
    cen_dec = cen_lat * RAD2DEG
    # max distance to vertices
    dmax = 0.0
    for i in range(n):
        d = haversine_deg(cen_ra, cen_dec, ra_deg[i], dec_deg[i])
        if d > dmax:
            dmax = d
    return cen_ra, cen_dec, dmax * 1.05 + 1e-6  # 5% inflation + tiny epsilon

@njit(cache=True)
def point_in_polygon_lonlat(point_ra, point_dec, poly_ra, poly_dec):
    """Ray-casting test in lon/lat degrees with simple local unwrap (assumes small polygons)."""
    # Unwrap poly RA near point_ra to avoid 0/360 issues
    n = poly_ra.size
    ra_un = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = poly_ra[i]
        # Shift to be close to point_ra
        dx = x - point_ra
        if dx > 180.0:
            x -= 360.0
        elif dx < -180.0:
            x += 360.0
        ra_un[i] = x
    inside = False
    j = n - 1
    for i in range(n):
        yi = poly_dec[i]
        yj = poly_dec[j]
        xi = ra_un[i]
        xj = ra_un[j]
        if ((yi > point_dec) != (yj > point_dec)):
            xint = (xj - xi) * (point_dec - yi) / (yj - yi + 1e-300) + xi
            if point_ra < xint:
                inside = not inside
        j = i
    return inside

# ----------------------------
# Core class
# ----------------------------

class SkyMasker:
    def __init__(self,
                 stars_path: Optional[str] = None,
                 ghosts_path: Optional[str] = None,
                 ngc_path: Optional[str] = None,
                 extras_path: Optional[str] = None,
                 input_catalog_path: Optional[str] = None,
                 radius_unit: str = "arcsec",
                 ra_min: Optional[float] = None,
                 ra_max: Optional[float] = None,
                 dec_min: Optional[float] = None,
                 dec_max: Optional[float] = None,
                 n_rand: int = 0,
                 seed: int = 12345):
        self.stars_path = stars_path
        self.ghosts_path = ghosts_path
        self.ngc_path = ngc_path
        self.extras_path = extras_path
        self.input_catalog_path = input_catalog_path
        self.radius_unit = radius_unit.lower()
        self.ra_min = ra_min
        self.ra_max = ra_max
        self.dec_min = dec_min
        self.dec_max = dec_max
        self.n_rand = n_rand
        self.seed = seed

        self.catalog = None  # pandas DataFrame with ra, dec
        self.tree = None     # BallTree on (dec, ra) radians

    # ---------- IO helpers ----------
    @staticmethod
    def _read_catalog_parquet(path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        ra_col = cols.get('ra', None) or [c for c in df.columns if c.lower() in ('ra','ra_deg','alpha')][0]
        dec_col = cols.get('dec', None) or [c for c in df.columns if c.lower() in ('dec','dec_deg','delta')][0]
        return df[[ra_col, dec_col]].rename(columns={ra_col:'ra', dec_col:'dec'})

    @staticmethod
    def _read_radial_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # find cols
        def pick(df, names):
            for n in names:
                if n in df.columns: return n
                low = n.lower()
                for c in df.columns:
                    if c.lower() == low: return c
            raise KeyError(f"Missing columns {names} in {path}")
        ra_c = pick(df, ['ra','RA','ra_deg','alpha'])
        dec_c = pick(df, ['dec','DEC','dec_deg','delta'])
        r_c = pick(df, ['radius','rad','r','mask_radius'])
        out = df[[ra_c, dec_c, r_c]].rename(columns={ra_c:'ra', dec_c:'dec', r_c:'radius'})
        return out

    @staticmethod
    def _read_ngc_csv(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        df = pd.read_csv(path)
        if 'vertices' not in df.columns:
            # try case-insensitive
            verts_col = None
            for c in df.columns:
                if c.lower() == 'vertices':
                    verts_col = c
                    break
            if verts_col is None:
                raise ValueError("NGC CSV must contain a 'vertices' column")
        else:
            verts_col = 'vertices'
        polys: List[Tuple[np.ndarray, np.ndarray]] = []
        for _, row in df.iterrows():
            verts = str(row[verts_col]).strip()
            pts = [p.strip() for p in verts.split(';') if p.strip()]
            ra_list = []
            dec_list = []
            for p in pts:
                parts = p.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid vertex '{p}' — expected 'ra dec'")
                ra_list.append(float(parts[0]))
                dec_list.append(float(parts[1]))
            polys.append((np.asarray(ra_list, dtype=np.float64), np.asarray(dec_list, dtype=np.float64)))
        return polys

    # ---------- Random catalog ----------
    def _generate_random_catalog(self) -> pd.DataFrame:
        assert self.ra_min is not None and self.ra_max is not None and self.dec_min is not None and self.dec_max is not None
        n = int(self.n_rand)
        rng = np.random.default_rng(self.seed)
        # RA uniform with wrap handling
        ra_min = normalize_ra_deg(self.ra_min)
        ra_max = normalize_ra_deg(self.ra_max)
        wrap = ra_max < ra_min
        if wrap:
            width = (ra_max + 360.0) - ra_min
            ra = ra_min + rng.random(n) * width
            ra = np.where(ra >= 360.0, ra - 360.0, ra)
        else:
            width = ra_max - ra_min
            ra = ra_min + rng.random(n) * width
        dec = sin_random_dec(n, self.dec_min, self.dec_max, np.uint32(self.seed))
        return pd.DataFrame({'ra': ra.astype(np.float64), 'dec': dec.astype(np.float64)})

    # ---------- Build BallTree ----------
    def build_tree(self):
        if self.input_catalog_path and os.path.exists(self.input_catalog_path):
            self.catalog = self._read_catalog_parquet(self.input_catalog_path)
        else:
            if self.n_rand <= 0:
                raise ValueError("No input catalog provided; please set --n-rand and bounds to generate a random catalog.")
            self.catalog = self._generate_random_catalog()
        # BallTree needs (lat, lon) radians with haversine metric
        lat = np.deg2rad(self.catalog['dec'].to_numpy(np.float64))
        lon = np.deg2rad(self.catalog['ra'].to_numpy(np.float64))
        X = np.vstack([lat, lon]).T
        self.tree = BallTree(X, metric='haversine')

    # ---------- Mask helpers ----------
    @staticmethod
    def _unit_to_rad(unit: str) -> float:
        u = unit.lower()
        if u in ('deg','degree','degrees'): return np.deg2rad(1.0)
        if u in ('arcmin','amin','minutes'): return np.deg2rad(1.0/60.0)
        if u in ('arcsec','asec','seconds'): return np.deg2rad(1.0/3600.0)
        raise ValueError(f"Unknown radius unit: {unit}")

    def _apply_radial_mask(self, df: pd.DataFrame, batch: int = 200_000) -> np.ndarray:
        assert self.tree is not None and self.catalog is not None
        rad_scale = self._unit_to_rad(self.radius_unit)
        out = np.zeros(len(self.catalog), dtype=bool)
        ra = df['ra'].to_numpy(np.float64)
        dec = df['dec'].to_numpy(np.float64)
        rad = df['radius'].to_numpy(np.float64) * rad_scale  # radians
        # Query BallTree built on main catalog, passing mask centers as query points
        for i0 in range(0, ra.size, batch):
            i1 = min(ra.size, i0 + batch)
            lat = np.deg2rad(dec[i0:i1])
            lon = np.deg2rad(ra[i0:i1])
            Y = np.vstack([lat, lon]).T
            r = rad[i0:i1]
            # query_radius supports array of radii (since sklearn 1.4). To stay robust, loop per sub-batch.
            for j in range(Y.shape[0]):
                ind = self.tree.query_radius(Y[j:j+1], r[j], count_only=False, return_distance=False)
                if len(ind) and ind[0].size:
                    out[ind[0]] = True
        return out

    def _apply_ngc_mask(self, polys: List[Tuple[np.ndarray, np.ndarray]], batch: int = 10_000) -> np.ndarray:
        assert self.tree is not None and self.catalog is not None
        # Pre-build arrays of query centers and radii from polygons (Numba)
        centers = []  # (dec_rad, ra_rad)
        radii = []    # radians
        poly_arr = [] # keep as numpy for later precise test
        for (ra_v, dec_v) in polys:
            cen_ra, cen_dec, rad_deg = polygon_center_and_radius_deg(ra_v, dec_v)
            centers.append([math.radians(cen_dec), math.radians(cen_ra)])  # (lat, lon)
            radii.append(math.radians(rad_deg))
            poly_arr.append((ra_v, dec_v))
        if len(centers) == 0:
            return np.zeros(len(self.catalog), dtype=bool)
        centers = np.array(centers, dtype=np.float64)
        radii = np.array(radii, dtype=np.float64)

        # Candidate preselection via BallTree
        candidates: List[np.ndarray] = []
        for i0 in range(0, centers.shape[0], batch):
            i1 = min(centers.shape[0], i0 + batch)
            Y = centers[i0:i1]
            rr = radii[i0:i1]
            for j in range(Y.shape[0]):
                ind = self.tree.query_radius(Y[j:j+1], rr[j], count_only=False, return_distance=False)
                if len(ind) and ind[0].size:
                    candidates.append(ind[0])
                else:
                    candidates.append(np.empty(0, dtype=int))

        # Precise point-in-polygon for candidates only (Numba)
        cat_ra = self.catalog['ra'].to_numpy(np.float64)
        cat_dec = self.catalog['dec'].to_numpy(np.float64)
        out = np.zeros(len(self.catalog), dtype=bool)
        c = 0
        for k, idx in enumerate(candidates):
            if idx.size == 0:
                continue
            ra_v, dec_v = poly_arr[k]
            for ii in idx:
                if point_in_polygon_lonlat(cat_ra[ii], cat_dec[ii], ra_v, dec_v):
                    out[ii] = True
        return out

    # ---------- Public API ----------
    def run(self) -> pd.DataFrame:
        self.build_tree()
        assert self.catalog is not None
        # Initialize mask columns as False
        self.catalog['mask_star'] = False
        self.catalog['mask_ghost'] = False
        self.catalog['mask_ngc'] = False
        self.catalog['mask_extra'] = False

        if self.stars_path and os.path.exists(self.stars_path):
            stars = self._read_radial_csv(self.stars_path)
            self.catalog['mask_star'] = self._apply_radial_mask(stars)
        if self.ghosts_path and os.path.exists(self.ghosts_path):
            ghosts = self._read_radial_csv(self.ghosts_path)
            self.catalog['mask_ghost'] = self._apply_radial_mask(ghosts)
        if self.extras_path and os.path.exists(self.extras_path):
            extras = self._read_radial_csv(self.extras_path)
            self.catalog['mask_extra'] = self._apply_radial_mask(extras)
        if self.ngc_path and os.path.exists(self.ngc_path):
            polys = self._read_ngc_csv(self.ngc_path)
            self.catalog['mask_ngc'] = self._apply_ngc_mask(polys)

        return self.catalog

# ----------------------------
# CLI
# ----------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="Sky masking with BallTree + Numba")
    ap.add_argument('--stars', type=str, default=None, help='CSV with [ra, dec, radius] for star masks')
    ap.add_argument('--ghosts', type=str, default=None, help='CSV with [ra, dec, radius] for ghost masks')
    ap.add_argument('--ngc', type=str, default=None, help='CSV with polygons: columns [poly_id, vertices]')
    ap.add_argument('--extras', type=str, default=None, help='CSV with [ra, dec, radius] for extra sources')
    ap.add_argument('--catalog', type=str, default=None, help='Parquet with main sources [ra, dec] (optional)')

    # Random catalog options if --catalog omitted
    ap.add_argument('--n-rand', type=int, default=0, help='Number of random points to generate if no catalog is given')
    ap.add_argument('--ra-min', type=float, default=None)
    ap.add_argument('--ra-max', type=float, default=None)
    ap.add_argument('--dec-min', type=float, default=None)
    ap.add_argument('--dec-max', type=float, default=None)
    ap.add_argument('--seed', type=int, default=12345)

    ap.add_argument('--radius-unit', type=str, default='arcsec', choices=['arcsec','arcmin','deg','degrees'])
    ap.add_argument('--out', type=str, required=True, help='Output path (.parquet or .csv)')
    return ap.parse_args()


def main():
    args = _parse_args()
    sm = SkyMasker(
        stars_path=args.stars,
        ghosts_path=args.ghosts,
        ngc_path=args.ngc,
        extras_path=args.extras,
        input_catalog_path=args.catalog,
        radius_unit=args.radius_unit,
        ra_min=args.ra_min, ra_max=args.ra_max,
        dec_min=args.dec_min, dec_max=args.dec_max,
        n_rand=args.n_rand,
        seed=args.seed,
    )
    df = sm.run()
    out = args.out
    if out.lower().endswith('.parquet'):
        df.to_parquet(out, index=False)
    elif out.lower().endswith('.csv'):
        df.to_csv(out, index=False)
    else:
        raise ValueError("Output must be .parquet or .csv")


if __name__ == '__main__':
    main()
