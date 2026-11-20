# -*- coding: utf-8 -*-
"""
00 - TWI i TPI iz postojecih DEM rastera (pokrece se jednom, pre 03). Izlaz ide u TERRAIN_DIR, pa ga 03 sam uzima.
  TWI = ln(SCA / (tan(slope) + 0.001)),  SCA = (flow_acc + 1) * velicina celije   [flow_acc u broju celija, D8]
        epsilon 0.001 sprecava eksploziju na ravnom terenu (isto kao u analizi vetrozastitnog pojasa)
  TPI_r = dem_filled - srednja visina u krugu poluprecnika r (bez NaN); vise razmera: lokalni -> sirok
Ulaz : TERRAIN_DIR/dem_filled_aoi.tif, slope_aoi.tif (stepeni), flow_accumulation_d8_aoi.tif
Izlaz: TERRAIN_DIR/twi_aoi.tif, tpi_<r>m_aoi.tif
"""
import os
import numpy as np
import rasterio
from scipy.signal import fftconvolve
from common import TERRAIN_DIR

DEM, SLOPE, FACC = "dem_filled_aoi.tif", "Slope_aoi.tif", "flow_accumulation_d8_aoi.tif"
FACC_JEDINICA = "cells"        # "cells" (broj celija) ili "area" (m2)
SLOPE_JEDINICA = "degrees"     # "degrees" ili "percent"
TPI_R = [50, 150, 500]         # poluprecnici [m]: mikroreljef, unutar parcele, sirok (depresije/grede)
EPS = 0.001


def citaj(ime):
    with rasterio.open(os.path.join(TERRAIN_DIR, ime)) as s:
        a = s.read(1).astype("float64")
        if s.nodata is not None: a[a == s.nodata] = np.nan
        a[a <= -9999] = np.nan
        return a, s.profile, abs(s.transform.a)


def snimi(a, prof, ime):
    prof = {**prof, "dtype": "float32", "nodata": -9999.0, "count": 1, "compress": "deflate"}
    with rasterio.open(os.path.join(TERRAIN_DIR, ime), "w", **prof) as d:
        d.write(np.where(np.isfinite(a), a, -9999.0).astype("float32"), 1)
    print(f"  {ime}: min={np.nanmin(a):.3f} | median={np.nanmedian(a):.3f} | max={np.nanmax(a):.3f}")


def tpi(dem, r_px):
    y, x = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
    k = (x ** 2 + y ** 2 <= r_px ** 2).astype(float)
    ok = np.isfinite(dem)
    suma = fftconvolve(np.where(ok, dem, 0), k, mode="same")
    n = fftconvolve(ok.astype(float), k, mode="same")
    sr = np.where(n > 0.5 * k.sum(), suma / np.maximum(n, 1e-9), np.nan)   # min. pola kruga validno (ivice AOI)
    return np.where(ok, dem - sr, np.nan)


if __name__ == "__main__":
    dem, prof, cs = citaj(DEM)
    slope, _, _ = citaj(SLOPE)
    facc, _, _ = citaj(FACC)
    print(f"DEM {dem.shape}, celija {cs} m")
    tan_s = slope / 100 if SLOPE_JEDINICA == "percent" else np.tan(np.deg2rad(slope))
    sca = (facc + 1) * cs if FACC_JEDINICA == "cells" else facc / cs
    snimi(np.log(sca / (tan_s + EPS)), prof, "twi_aoi.tif")
    for r in TPI_R:
        snimi(tpi(dem, max(1, round(r / cs))), prof, f"tpi_{r}m_aoi.tif")
