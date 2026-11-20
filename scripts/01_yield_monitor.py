# -*- coding: utf-8 -*-
"""
01 - Sirovi yield monitor CSV-ovi -> ociscen point dataset po godini.
Ulaz : YM_RAW/<godina>/*.csv (jedan fajl po parceli; parcela = prva rec imena fajla, usev u imenu fajla)
Izlaz: OUT/01_yield_points_<godina>.csv  (parcel, crop, year, X, Y [UTM34N], yield [t/ha], moisture [%], speed)
       OUT/01_yield_filter_log.csv       (koliko tacaka je izbaceno u kom koraku, po parceli)
       OUT/01_crop_provera.csv           (usev iz imena fajla vs. kolona Product, po fajlu)
Filtri: drugi usev u Product -> poligon (unutrasnji bafer) -> agronomski min/max -> IQR prinosa.
        Vlaga i brzina se NE filtriraju (senzori nepouzdani, brzina u 2019/2021 u drugim jedinicama); kolone ostaju.
"""
import os, glob, unicodedata
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from common import *

KOLONE = {"Yld_Mass_D": "yield", "Moisture__": "moisture", "Speed_km_h": "speed", "X": "X", "Y": "Y"}
# Kljucne reci (EN / SR latinica i cirilica / HU); poredi se bez dijakritika i velikih slova, kao deo teksta
USEV_KLJUC = {"wheat":     ["wheat", "psen", "пшен", "buza"],                       # búza, őszi búza
              "corn":      ["corn", "maize", "kukur", "кукур", "kukor"],            # kukorica
              "sunflower": ["sunfl", "suncok", "сунцо", "napraf"],                  # napraforgó
              "soybeans":  ["soy", "soja", "соја", "szoja"],                        # szója
              "barley":    ["barley", "jecam", "јечам", "arpa"],                    # árpa (ne prati se)
              "rapeseed":  ["rapeseed", "canola", "repic", "репиц", "repce"]}       # repce (ne prati se)
CROP_IZVOR = "fajl"       # kad se ime fajla i Product ne slazu: "fajl" ili "product"
PRODUCT_COL = "Product"
PRODUCT_IGNORISI = {("T-18-3-3", 2019), ("T-200", 2019)}   # Product pogresan u fajlu (knjige farme) -> sve tacke = usev iz imena
IQR_K     = 1.5           # parcel-level IQR prinosa
EDGE_BUF  = 10.0          # unutrasnji bafer poligona [m] - uklanja uvratine/ivice; 0 = bez bafera

def norm(s):   # bez dijakritika; ostecena slova iz eksporta (\ufffd, ?) se brisu
    s = unicodedata.normalize("NFKD", str(s).lower()).replace("\ufffd", "").replace("?", "")
    return "".join(c for c in s if not unicodedata.combining(c))

def prepoznaj(tekst):
    t = norm(tekst)
    return next((crop for crop, kl in USEV_KLJUC.items() if any(norm(k) in t for k in kl)), None)

def usev(f, df, year, provera):
    ime = os.path.basename(f)
    c_fajl = prepoznaj(ime)
    pcol = next((c for c in df.columns if c.lower() == PRODUCT_COL.lower()), None)
    prod = df[pcol].dropna().astype(str).str.strip().value_counts() if pcol else pd.Series(dtype=int)
    c_prod = {p: prepoznaj(p) for p in prod.index}
    c_glavni = c_prod.get(prod.index[0]) if len(prod) else None
    crop = (c_fajl if CROP_IZVOR == "fajl" else c_glavni) or c_fajl or c_glavni
    ok = c_fajl == c_glavni if c_fajl and c_glavni else None
    nepoznat = [p for p, c in c_prod.items() if c is None]
    provera.append({"year": year, "fajl": ime, "usev_fajl": c_fajl, "usev_product": c_glavni, "poklapanje": ok,
                    "product_vrednosti": "; ".join(f"{p} ({n})" for p, n in prod.items()),
                    "nepoznat_product": "; ".join(nepoznat), "vise_useva_u_fajlu": len({c for c in c_prod.values() if c}) > 1,
                    "koriscen": crop})
    if ok is False:
        print(f"  !! {ime}: ime fajla = {c_fajl}, Product = {c_glavni} ({prod.index[0]}) -> koristim {crop}")
    if nepoznat:
        print(f"  !! {ime}: nepoznat Product {nepoznat} -> dodaj u USEV_KLJUC")
    if provera[-1]["vise_useva_u_fajlu"]:
        print(f"  !! {ime}: vise useva u koloni Product: {prod.to_dict()}")
    return crop

def ucitaj(f):
    with open(f, encoding="utf-8", errors="ignore") as fh:
        h = fh.readline()
    sep = ";" if h.count(";") > h.count(",") else ","
    df = pd.read_csv(f, sep=sep, decimal="," if sep == ";" else ".", low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def poligon(parcel):
    p = os.path.join(POLY_DIR, f"{parcel}.shp")
    if not os.path.exists(p):
        return None
    g = gpd.read_file(p)
    g = g.set_crs(CRS) if g.crs is None else g.to_crs(CRS)
    poly = spoji(g.geometry)
    inner = poly.buffer(-EDGE_BUF) if EDGE_BUF else poly
    return poly if inner.is_empty else inner

def iqr_ok(s, k=IQR_K):
    q1, q3 = s.quantile([0.25, 0.75])
    return s.between(q1 - k * (q3 - q1), q3 + k * (q3 - q1))

def u_poligone(df, year, log):
    """Dodela tacaka (pod)poligonima parcele; tacke van poligona se izbacuju."""
    parent = df["parcel"].iat[0]
    subs = SPLIT.get((parent, year), [parent])
    keep = pd.Series(False, index=df.index)
    for s in subs:
        poly = poligon(s)
        if poly is None:
            if len(subs) == 1:
                print(f"    {s}: SHP ne postoji -> bez klipovanja")
                return df
            print(f"    {s}: SHP pod-poligona ne postoji -> preskacem")
            continue
        m = u_poligonu(poly, df["X"].values, df["Y"].values) & ~keep
        df.loc[m, "parcel"] = s
        keep |= m
    log["rm_poligon"] = int((~keep).sum())
    return df[keep]

def filtriraj(g, crop, year):
    log = {"n_ulaz": len(g)}
    lo, hi = agro(crop, year)
    m = g["yield"].between(lo, hi); log["rm_agro"] = int((~m).sum()); g = g[m]
    m = iqr_ok(g["yield"]) if len(g) >= 20 else pd.Series(True, g.index)
    log["rm_iqr"] = int((~m).sum()); g = g[m]
    log["n_izlaz"] = len(g)
    return g, log

def obradi_fajl(f, year, logs, provera):
    df = ucitaj(f)
    crop = usev(f, df, year, provera)
    if crop not in AGRO:
        print(f"  {os.path.basename(f)}: usev nije prepoznat/pracen ({crop}) -> preskacem"); return None
    ime = os.path.basename(f)
    pcol = next((c for c in df.columns if c.lower() == PRODUCT_COL.lower()), None)
    n_raw, drugi = len(df), pd.Series(False, df.index)
    if pcol and (ime.split()[0], year) not in PRODUCT_IGNORISI:   # tacke sa DRUGIM prepoznatim usevom (npr. kukuruz u uvratinama psenice); nepoznat/prazan Product ostaje
        p = df[pcol].astype(str).str.strip()
        p = p.map({v: prepoznaj(v) for v in p.unique()})
        drugi = p.notna() & (p != crop)
        if drugi.any():
            print(f"    {ime}: {drugi.sum():,} tacaka sa drugim usevom u Product {p[drugi].value_counts().to_dict()} -> izbaceno")
        df = df[~drugi]
    nema = [c for c in ("Yld_Mass_D", "X", "Y") if c not in df.columns]
    if nema:
        print(f"  {ime}: nedostaju kolone {nema} -> preskacem"); return None
    df = df[[c for c in KOLONE if c in df.columns]].rename(columns=KOLONE)
    for c in ("moisture", "speed"):
        if c not in df: df[c] = np.nan
    df["parcel"] = ime.split()[0]                        # parcela iz imena fajla (kolona Field se ne koristi)
    for c in ("yield", "moisture", "speed", "X", "Y"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df[df["yield"].gt(0) & df[["X", "Y"]].notna().all(axis=1)].copy()

    if df["yield"].median() > 100:                       # kg/ha -> t/ha
        df["yield"] /= 1000; print(f"    {ime}: prinos u kg/ha -> /1000")
    if df["X"].abs().median() < 180:                     # WGS84 -> UTM 34N
        df["X"], df["Y"] = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(df["X"].values, df["Y"].values)
    df["crop"], df["year"] = crop, year

    out = []
    for parcel, g in df.groupby("parcel"):
        if iskljucen(parcel, year):
            print(f"    {parcel}: u IZUZMI -> preskacem"); continue
        log = {"year": year, "parcel": parcel, "crop": crop, "n_raw": n_raw, "rm_drugi_usev": int(drugi.sum()),
               "rm_nevalidno": n0 - len(df)}
        g = u_poligone(g, year, log)
        if g.empty:
            logs.append({**log, "n_ulaz": 0, "n_izlaz": 0}); print(f"    {parcel}: sve tacke van poligona -> preskacem"); continue
        for sub, gs in g.groupby("parcel"):
            gs, l = filtriraj(gs, crop, year)
            logs.append({**log, **l, "parcel": sub})
            out.append(gs)
            print(f"    {sub:10s} {crop:9s}: {l['n_ulaz']:>7,} -> {l['n_izlaz']:>7,} | van poligona {log.get('rm_poligon', 0):>6,} "
                  f"| mean={gs['yield'].mean():.2f} t/ha")
    return pd.concat(out) if out else None


if __name__ == "__main__":
    logs, provera = [], []
    for year in YEARS:
        dirs = [d for d in glob.glob(os.path.join(YM_RAW, f"*{year}*")) if os.path.isdir(d)]
        files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.csv")))
        print(f"\n{'=' * 60}\n{year}: {len(files)} fajlova")
        parts = [p for f in files if (p := obradi_fajl(f, year, logs, provera)) is not None]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)[["parcel", "crop", "year", "X", "Y", "yield", "moisture", "speed"]]
        df.to_csv(put(f"01_yield_points_{year}.csv"), index=False)
        print(f"  Sacuvano: {len(df):,} tacaka | {df['parcel'].nunique()} parcela")
    pr = pd.DataFrame(provera)
    pr.to_csv(put("01_crop_provera.csv"), index=False)
    if len(pr):
        print(f"\nProvera useva: {len(pr)} fajlova | poklapa se {int((pr['poklapanje'] == True).sum())} | "
              f"NE poklapa se {int((pr['poklapanje'] == False).sum())} | bez Product kolone/prepoznavanja "
              f"{int(pr['poklapanje'].isna().sum())} | nepracen/neprepoznat usev {int((~pr['koriscen'].isin(list(AGRO))).sum())}")
    log = pd.DataFrame(logs)
    log.to_csv(put("01_yield_filter_log.csv"), index=False)
    if len(log):
        print("\nIzbaceno po koracima:\n", log.filter(like="rm_").sum().to_string())
