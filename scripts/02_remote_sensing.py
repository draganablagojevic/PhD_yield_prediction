# -*- coding: utf-8 -*-
"""
02 - Sentinel-2 + Landsat-8 + Landsat-7 -> pixel-level dataset (10 m, mreza poravnata sa S2).
  1) YM tacke se usrednjavaju po pikselu (pikseli sa tackama iz >1 parcele se izbacuju)
  2) za svaki datum: reflektanse (B02,B03,B04,B08,B11), maska oblaka (S2 SCL / Landsat QA), 15 indeksa
     isti datum sa vise senzora: prioritet S2 > L8 > L7, prazni (oblacni/SLC-off) pikseli se popunjavaju nizim
  3) anomalije: fizicki opsezi + robusni outlier unutar parcele-datuma -> NaN
Ulaz : OUT/01_yield_points_<godina>.csv, S2_DIR, L8_DIR, L7_DIR
Izlaz: OUT/02_pixels_<godina>.csv  (kolone <INDEKS>_<YYYYMMDD>), OUT/02_scene_log.csv (po parceli i datumu),
       OUT/02_pokrivenost_<godina>.csv (parcela x datum, % cistih piksela)
S2_DIR / L8_DIR / L7_DIR se pretrazuju rekurzivno (svi podfolderi); godina se cita iz datuma u imenu fajla.
Nazivi: S2  ..._T34TCR_20190512T094041_B04_10m.tif (B11, SCL: _20m)
        L8  LC08_L2SP_186028_20190512_20200828_02_T1_SR_B4.TIF (C2) ili LC08_..._01_T1_sr_band4.tif (C1); QA_PIXEL / pixel_qa
        L7  LE07_L2SP_186028_20190512_20200828_02_T1_SR_B3.TIF (C2) ili LE07_..._01_T1_sr_band3.tif (C1); QA_PIXEL / pixel_qa
Kanali:      plava  zelena  crvena  NIR  SWIR1      L7 ETM+ nema coastal kanal, pa su brojevi pomereni za 1 u odnosu na L8
  S2 MSI     B02    B03     B04     B08  B11
  L8 OLI     B2     B3      B4      B5   B6
  L7 ETM+    B1     B2      B3      B4   B5         (B6 termalni, B7 SWIR2 - ne koriste se)
"""
import os, re, glob
import numpy as np
import pandas as pd
import rasterio
from common import *

MIN_PTS        = 3      # min. YM tacaka po pikselu
MIN_DATES      = 5      # min. validnih GNDVI datuma po pikselu
MAD_K          = 4.0    # robusni z-prag za outlier unutar parcele-datuma (NDVI)
MIN_CLEAR_FRAC = 0.3    # ako je < 30% piksela parcele cisto tog datuma, ceo parcela-datum -> NaN
S2_OFFSET      = 0      # BOA_ADD_OFFSET; 0 za snimke procesirane pre 25.1.2022 (baseline < 04.00)
S2_SCL_LOS     = [0, 1, 3, 8, 9, 10, 11]   # nodata, saturated, senka, oblak (sr/vis), cirrus, sneg
# Usklađivanje sa S2 (nagib, odsecak) po kanalu; identitet = bez korekcije (npr. Claverie 2018 za OLI, Roy 2016 za ETM+)
#HARM = {s: {b: (1.0, 0.0) for b in ["B02", "B03", "B04", "B08", "B11"]} for s in ("L8", "L7")} # prvi put napisano
HARM = {"L8": {"B02": (1.0990, +0.0148), "B03": (1.2356, -0.0052), "B04": (1.1710, -0.0006), "B08": (1.1081, -0.0301), "B11": (1.0169, +0.0206)},
        "L7": {"B02": (1.2670, -0.0031), "B03": (1.1668, -0.0040), "B04": (1.1705, -0.0063), "B08": (0.9795, +0.0162), "B11": (1.0132, +0.0286)}}
BANDS = ["B02", "B03", "B04", "B08", "B11"]
L_BAND = {"L8": {"2": "B02", "3": "B03", "4": "B04", "5": "B08", "6": "B11"},
          "L7": {"1": "B02", "2": "B03", "3": "B04", "4": "B08", "5": "B11"}}
PRIOR = {"S2": 0, "L8": 1, "L7": 2}
S2_RE = re.compile(r"(T\d{2}[A-Z]{3})_(\d{8})T\d{6}_(B02|B03|B04|B08|B11|SCL)_\d{2}m\.(tif|jp2)$", re.I)
L_RE = re.compile(r"(L[CE]0[789])_\w{4}_(\d{6})_(\d{8})_\d{8}_(\d{2})_\w{2}_(SR_B\d|sr_band\d|QA_PIXEL|pixel_qa)\.tif$", re.I)
OPSEG = {"NDVI": (-1, 1), "GNDVI": (-1, 1), "NDMI": (-1, 1), "ARVI": (-1, 1), "WDRVI": (-1, 1), "OSAVI": (-1, 1),
         "EVI": (-1, 1), "MSAVI2": (-1, 1), "SAVI": (-1.5, 1.5), "IPVI": (0, 1), "TVI": (0, 1.23),
         "DVI": (-1, 1), "SR": (0, 30), "GCI": (-1, 20), "CVI": (0, 30)}


def skeniraj(year):
    """{(datum, senzor, scena): {band: putanja}}"""
    sc = {}
    for f in glob.glob(os.path.join(S2_DIR, "**", "*"), recursive=True):
        if (m := S2_RE.search(os.path.basename(f))) and m[2][:4] == str(year):
            d = sc.setdefault((m[2], "S2", m[1]), {})
            b = m[3].upper()
            if b not in d or m[4].lower() == "tif": d[b] = f
    for folder in (L8_DIR, L7_DIR):
        for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
            if not ((m := L_RE.search(os.path.basename(f))) and m[3][:4] == str(year)):
                continue
            s = "L7" if m[1].upper().startswith("LE") else "L8"
            d = sc.setdefault((m[3], s, m[2]), {"_c2": m[4] == "02"})
            b = m[5]
            if "qa" in b.lower(): d["QA"] = f
            elif b[-1] in L_BAND[s]: d[L_BAND[s][b[-1]]] = f
    return sc


def reflektanse(b, sensor, xs, ys):
    if any(k not in b for k in BANDS + (["SCL"] if sensor == "S2" else ["QA"])):
        return None
    R = {k: sample_raster(b[k], xs, ys) for k in BANDS}
    if sensor == "S2":
        R = {k: (v + S2_OFFSET) / 10000 for k, v in R.items()}
        los = np.isin(sample_raster(b["SCL"], xs, ys), S2_SCL_LOS)
    else:
        s, o = (2.75e-5, -0.2) if b["_c2"] else (1e-4, 0.0)
        R = {k: (v * s + o) * HARM[sensor][k][0] + HARM[sensor][k][1] for k, v in R.items()}
        qa = np.nan_to_num(sample_raster(b["QA"], xs, ys), nan=1).astype(int)
        bits = (0, 1, 2, 3, 4, 5) if b["_c2"] else (0, 3, 4, 5)   # C2: fill, dilated, cirrus(L8), oblak, senka, sneg
        los = np.bitwise_and(qa, sum(1 << i for i in bits)) > 0
    los |= np.any([~((v > 0) & (v <= 1)) for v in R.values()], axis=0)   # SLC-off pruge (0) i nevalidno -> NaN
    for v in R.values(): v[los] = np.nan
    return R


def indeksi(R):
    B, G, Rd, N, S = (R[k].astype("float64") for k in BANDS)
    with np.errstate(all="ignore"):
        ndvi = (N - Rd) / (N + Rd)
        I = {"NDVI": ndvi, "SAVI": 1.5 * (N - Rd) / (N + Rd + 0.5),
             "MSAVI2": (2 * N + 1 - np.sqrt((2 * N + 1) ** 2 - 8 * (N - Rd))) / 2,
             "EVI": 2.5 * (N - Rd) / (N + 6 * Rd - 7.5 * B + 1), "GNDVI": (N - G) / (N + G),
             "IPVI": N / (N + Rd), "GCI": N / G - 1, "NDMI": (N - S) / (N + S), "OSAVI": (N - Rd) / (N + Rd + 0.16),
             "DVI": N - Rd, "ARVI": (N - (2 * Rd - B)) / (N + (2 * Rd - B)), "SR": N / Rd,
             "WDRVI": (0.1 * N - Rd) / (0.1 * N + Rd), "TVI": np.sqrt(ndvi + 0.5), "CVI": N * Rd / G ** 2}
    for k, (lo, hi) in OPSEG.items():
        I[k][~((I[k] >= lo) & (I[k] <= hi))] = np.nan
    return I


def anomalije(ndvi, parcel):
    """True za piksel-datum koji je outlier u parceli ili parcela-datum skoro potpuno oblacan."""
    s = pd.Series(ndvi)
    g = s.groupby(parcel)
    med = g.transform("median")
    mad = (s - med).abs().groupby(parcel).transform("median").mul(1.4826).clip(lower=0.02)
    cnt, frac = g.transform("count"), g.transform("count") / g.transform("size")
    return (((s - med).abs() > MAD_K * mad) & (cnt >= 10) | (frac < MIN_CLEAR_FRAC)).values


def pikseli(year, x0, y0):
    ym = pd.read_csv(put(f"01_yield_points_{year}.csv"))
    ym["ix"] = np.floor((ym["X"] - x0) / RES).astype(int)
    ym["iy"] = np.floor((y0 - ym["Y"]) / RES).astype(int)
    mix = ym.groupby(["ix", "iy"])["parcel"].transform("nunique") > 1
    px = (ym[~mix].groupby(["parcel", "crop", "year", "ix", "iy"])
          .agg(yield_=("yield", "mean"), moisture=("moisture", "mean"), n_points=("yield", "size"))
          .reset_index().rename(columns={"yield_": "yield"}))
    print(f"  YM: {len(ym):,} tacaka -> {len(px):,} piksela | mesoviti (2+ parcele): {mix.sum():,} tacaka | "
          f"< {MIN_PTS} tacke: {(px['n_points'] < MIN_PTS).sum():,} piksela")
    px = px[px["n_points"] >= MIN_PTS].reset_index(drop=True)
    px["X"] = x0 + (px["ix"] + 0.5) * RES
    px["Y"] = y0 - (px["iy"] + 0.5) * RES
    return px


def obradi(year, scenes, log):
    s2 = [b["B04"] for (d, s, _), b in scenes.items() if s == "S2" and "B04" in b]
    if s2:
        with rasterio.open(s2[0]) as src: x0, y0 = src.transform.c % RES, src.transform.f % RES
    else:
        x0 = y0 = 0.0
    px = pikseli(year, x0, y0)
    xs, ys, parcel = px["X"].values, px["Y"].values, px["parcel"].values
    kolone = {}
    for date in sorted({d for d, _, _ in scenes}):
        R, used = None, []
        for (d, s, _), b in sorted(scenes.items(), key=lambda kv: PRIOR[kv[0][1]]):
            if d != date or (Rs := reflektanse(b, s, xs, ys)) is None:
                continue
            used.append(s)
            R = Rs if R is None else {k: np.where(np.isnan(R[k]), Rs[k], R[k]) for k in BANDS}
        sensor = "+".join(sorted(set(used), key=PRIOR.get))
        if R is None:
            print(f"    {date}: nepotpuni bendovi -> preskacem"); continue
        I = indeksi(R)
        los = anomalije(I["NDVI"], parcel) | np.isnan(I["NDVI"])
        for k, v in {**R, **I}.items():
            v[los] = np.nan
            kolone[f"{k}_{date}"] = v.astype("float32")
        n_ok = int((~los).sum())
        pp = pd.Series(~los).groupby(parcel).agg(["size", "sum"])
        log += [{"year": year, "date": date, "sensor": sensor, "parcel": p, "n_pix": r["size"], "n_cisto": int(r["sum"]),
                 "pct_cisto": round(100 * r["sum"] / r["size"], 1)} for p, r in pp.iterrows()]
        print(f"    {date} {sensor:6s}: cisto {n_ok:>7,} / {len(px):,} ({100 * n_ok / len(px):5.1f}%)")
    df = pd.concat([px.drop(columns=["ix", "iy"]), pd.DataFrame(kolone)], axis=1).dropna(axis=1, how="all")
    n_valid = df.filter(regex=r"^GNDVI_\d{8}$").notna().sum(axis=1)
    print(f"  Pikseli sa < {MIN_DATES} validnih datuma: {(n_valid < MIN_DATES).sum():,} -> izbaceni")
    return df[n_valid >= MIN_DATES].reset_index(drop=True)


if __name__ == "__main__":
    log = []
    for year in YEARS:
        if not os.path.exists(put(f"01_yield_points_{year}.csv")):
            continue
        scenes = skeniraj(year)
        n = pd.Series([s for _, s, _ in scenes]).value_counts().to_dict()
        print(f"\n{'=' * 60}\n{year}: scene po senzoru {n}")
        df = obradi(year, scenes, log)
        df.to_csv(put(f"02_pixels_{year}.csv"), index=False)
        print(f"  Sacuvano: {df.shape[0]:,} piksela x {df.shape[1]} kolona")
    log = pd.DataFrame(log)
    log.to_csv(put("02_scene_log.csv"), index=False)
    for y, g in log.groupby("year"):   # tabela parcela x datum, % cistih piksela
        g.pivot_table(index="parcel", columns="date", values="pct_cisto").to_csv(put(f"02_pokrivenost_{y}.csv"))
