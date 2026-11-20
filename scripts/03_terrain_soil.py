# -*- coding: utf-8 -*-
"""
03 - DEM derivati (10 m) i SoilGrids (250 m) -> kolone u pixel datasetu (uzorkovanje u centru piksela).
  - svaki .tif u TERRAIN_DIR / SOIL_DIR postaje kolona (ime = ime fajla)
  - nodata / -9999 / -32768 -> NaN, SoilGrids skaliranje u standardne jedinice
  - fizicki opsezi iz common.RASPON (negativne vrednosti -> NaN, osim TPI/curvature koji su po definiciji +/-)
  - aspect -> northness/eastness (ravno = 0), flow_acc -> log1p
Ulaz : OUT/02_pixels_<godina>.csv
Izlaz: OUT/03_pixels_ts_<godina>.csv, OUT/03_terrain_soil_log.csv
"""
import os, glob
import numpy as np
import pandas as pd
from common import *

# SoilGrids mapped units -> standardne: bdod cg/cm3->g/cm3, cec mmol/kg->cmol/kg, cfvo/clay/sand/silt ->%,
# nitrogen cg/kg->g/kg, phh2o pHx10->pH, soc dg/kg->g/kg, ocd hg/m3->kg/m3
SOIL_SCALE = {"bdod": 100, "cec": 10, "cfvo": 10, "clay": 10, "sand": 10, "silt": 10,
              "nitrogen": 100, "phh2o": 10, "soc": 10, "ocd": 10}
FLAT_SLOPE = 0.5   # [deg] ispod ovoga aspect nema smisla -> northness = eastness = 0
SOIL_ARTEFAKT = {"phh2o": 6.5, "ocs": 30}   # ispod ovoga -> piksel u prelaznoj zoni ka vodi (T-120 jezero)
# Ne uzorkuju se: D8 smer toka je kategorija (1,2,4..128), sirovi DEM je ~ isti kao dem_filled
PRESKOCI = {"flow_direction_d8", "dem", "flow_accumulation_d8"}


def rasteri(folder):
    r = {os.path.splitext(os.path.basename(f))[0].lower().replace("_aoi", ""): f
         for f in sorted(glob.glob(os.path.join(folder, "*.tif")))}
    return {k: v for k, v in r.items() if k not in PRESKOCI}


def obradi(df, log, year):
    xs, ys = df["X"].values, df["Y"].values
    nove, soil = {}, []
    for izvor, folder in (("terrain", TERRAIN_DIR), ("soil", SOIL_DIR)):
        for name, f in rasteri(folder).items():
            v = sample_raster(f, xs, ys)
            v[v <= -9999] = np.nan
            if izvor == "soil":
                v = v / next((s for k, s in SOIL_SCALE.items() if name.startswith(k)), 1)
                soil.append(name)
            lo, hi = raspon(name)
            van = np.zeros(len(v), bool)
            if lo is not None: van |= v < lo
            if hi is not None: van |= v > hi
            q1, q3 = np.nanpercentile(v, [25, 75]) if np.isfinite(v).any() else (np.nan, np.nan)
            iqr_out = int(((v < q1 - 3 * (q3 - q1)) | (v > q3 + 3 * (q3 - q1))).sum())
            v[van] = np.nan
            nove[name] = v
            log.append({"year": year, "kolona": name, "izvor": izvor, "nan_pct": round(100 * np.isnan(v).mean(), 2),
                        "van_opsega": int(np.sum(van)), "iqr3_ekstremi_zadrzani": iqr_out,
                        "min": np.nanmin(v) if np.isfinite(v).any() else np.nan,
                        "max": np.nanmax(v) if np.isfinite(v).any() else np.nan})
    n = pd.DataFrame(nove, index=df.index)
    art = np.zeros(len(n), bool)
    for k, prag in SOIL_ARTEFAKT.items():
        c = next((c for c in soil if c.startswith(k)), None)
        if c: art |= (n[c] < prag).values
    n.loc[art, soil] = np.nan
    print(f"  SoilGrids artefakti {SOIL_ARTEFAKT} -> NaN: {art.sum():,} piksela {pd.Series(df['parcel'][art]).value_counts().to_dict()}")
    asp = next((c for c in n if c.startswith("aspect")), None)
    slp = next((c for c in n if c.startswith("slope")), None)
    if asp:
        a = np.deg2rad(n[asp])
        ravno = n[slp] < FLAT_SLOPE if slp else pd.Series(False, n.index)
        n["northness"] = np.where(ravno, 0, np.cos(a))
        n["eastness"] = np.where(ravno, 0, np.sin(a))
    for c in [c for c in n if c.startswith("flow_acc") and "log" not in c and f"{c}_log" not in n
              and "flow_accumulation_log" not in n]:
        n[f"{c}_log"] = np.log1p(n[c])
    tex = [c for c in n if c.split("_")[0] in ("clay", "sand", "silt")]
    if len(tex) == 3:
        suma = n[tex].sum(axis=1, min_count=3)
        los = (suma - 100).abs() > 10
        n.loc[los, tex] = np.nan
        print(f"  Tekstura clay+sand+silt: mean={suma.mean():.1f}% | |suma-100|>10 -> NaN: {los.sum():,}")
    return pd.concat([df, n], axis=1)


if __name__ == "__main__":
    log = []
    for year in YEARS:
        p = put(f"02_pixels_{year}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        print(f"\n{'=' * 60}\n{year}: {len(df):,} piksela")
        df = obradi(df, log, year)
        df.to_csv(put(f"03_pixels_ts_{year}.csv"), index=False)
        print(f"  Sacuvano: {df.shape[0]:,} x {df.shape[1]}")
    log = pd.DataFrame(log)
    log.to_csv(put("03_terrain_soil_log.csv"), index=False)
    if len(log):
        print("\n", log.groupby("kolona")[["nan_pct", "van_opsega", "iqr3_ekstremi_zadrzani", "min", "max"]]
              .agg({"nan_pct": "max", "van_opsega": "sum", "iqr3_ekstremi_zadrzani": "sum", "min": "min", "max": "max"})
              .round(3).to_string())
