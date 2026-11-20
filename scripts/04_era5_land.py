# -*- coding: utf-8 -*-
"""
04 - ERA5-Land (vec preuzeti era5_land_<godina>.nc) -> dnevne vrednosti i GDD po parceli od setve do zetve.
Ispravke u odnosu na era5_land_per_parcel.py:
  - padavine: ERA5-Land 'tp' je AKUMULIRAN od 00 UTC; dnevna suma = vrednost u 00 UTC narednog dana
    (staro sabiranje satnih vrednosti je precenjivalo padavine ~10x). Akumulacija se detektuje automatski.
  - satne serije svih potrebnih godina se spajaju PRE dnevne agregacije (31.12. dobija 00 UTC od 1.1.)
  - NetCDF se otvara jednom po godini; uklonjene nekoriscene funkcije i preuzimanje (podaci postoje)
Izlaz: OUT/04_era5_daily_per_parcel.csv
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from common import *

KULTURE = {"пшеница": "wheat", "кукуруз": "corn", "сунцокрет": "sunflower", "соја": "soybeans"}
GDD = {"wheat": (0.0, 30.0), "corn": (10.0, 30.0), "sunflower": (6.0, 30.0), "soybeans": (10.0, 30.0)}  # (T_base, T_upper)
COL = {"parcel": "Парцела", "year": "Година", "crop": "Култура", "sow": "Датум садње", "harv": "Датум жетве",
       "lat": "Latitude", "lon": "Longitude"}


def ucitaj_setve():
    df = pd.read_excel(SETVE_XLSX)
    df[[COL["parcel"], COL["lat"], COL["lon"]]] = df[[COL["parcel"], COL["lat"], COL["lon"]]].ffill()
    df = df[df[COL["crop"]].isin(KULTURE)].rename(columns={v: k for k, v in COL.items()})
    df["crop"] = df["crop"].map(KULTURE)
    df["year"] = df["year"].astype(int)
    df[["sow", "harv"]] = df[["sow", "harv"]].apply(pd.to_datetime)
    df = df[[not iskljucen(p, y) for p, y in zip(df["parcel"], df["year"])]]
    print(f"Tabela setve: {len(df)} unosa")
    return df.reset_index(drop=True)


def centroid(parcel, row):
    shp = os.path.join(POLY_DIR, f"{parcel}.shp")
    if os.path.exists(shp):
        g = gpd.read_file(shp)
        g = g.set_crs(CRS) if g.crs is None else g
        c = g.to_crs("EPSG:4326").geometry.union_all().centroid
        return c.x, c.y
    print(f"  {parcel}: SHP ne postoji -> koordinate iz Excela")
    return float(row["lon"]), float(row["lat"])


def satno(ds, lon, lat):
    p = ds.sel(latitude=lat, longitude=lon, method="nearest").squeeze(drop=True)
    t = (p["t2m"].to_series() - 273.15).rename("T")
    tp = (p["tp"].to_series().clip(lower=0) * 1000).rename("tp")
    out = pd.concat([t, tp], axis=1)
    out.index = pd.to_datetime(out.index)
    return out


def dnevno(h):
    h = h[~h.index.duplicated()].sort_index()
    d = h["T"].resample("D").agg(["min", "max", "mean"]).set_axis(["Tmin_C", "Tmax_C", "Tmean_C"], axis=1)
    dif = h["tp"].diff()[h.index.hour != 1]
    akum = (dif[dif.abs() > 1e-6] > 0).mean() > 0.9
    if akum:   # akumulirano od 00 UTC -> vrednost u 00 UTC = suma prethodnog dana
        p = h["tp"][h.index.hour == 0]
        p.index = p.index.normalize() - pd.Timedelta(days=1)
    else:
        p = h["tp"].resample("D").sum()
    d["prec_mm"] = p
    return d.dropna(subset=["Tmean_C"]), akum


if __name__ == "__main__":
    setve = ucitaj_setve()
    godine = sorted(set(setve["year"]) | set(setve["sow"].dt.year))
    ds = {}
    for y in godine:
        f = os.path.join(ERA5_DIR, f"era5_land_{y}.nc")
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} ne postoji (preuzmi starom skriptom era5_land_per_parcel.py)")
        ds[y] = xr.open_dataset(f)
    sve, cent = [], {}
    for _, r in setve.iterrows():
        if r["parcel"] not in cent:
            cent[r["parcel"]] = centroid(r["parcel"], r)
        lon, lat = cent[r["parcel"]]
        h = pd.concat([satno(ds[y], lon, lat) for y in range(r["sow"].year, r["year"] + 1)])
        d, akum = dnevno(h)
        d = d.loc[r["sow"]:r["harv"]].copy()
        if d.empty:
            print(f"  {r['parcel']} {r['year']}: nema ERA5 podataka -> preskacem"); continue
        tb, tu = GDD[r["crop"]]
        d["GDD_day"] = ((d["Tmax_C"].clip(upper=tu) + d["Tmin_C"].clip(upper=tu)) / 2 - tb).clip(lower=0)
        d["GDD_cum"] = d["GDD_day"].cumsum()
        d["prec_cum"] = d["prec_mm"].cumsum()
        d = d.rename_axis("date").reset_index()
        d.insert(0, "parcel", r["parcel"]); d.insert(1, "crop", r["crop"]); d.insert(2, "year", r["year"])
        d["sowing_date"], d["harvest_date"] = r["sow"].date(), r["harv"].date()
        sve.append(d)
        print(f"  {r['parcel']:10s} {r['crop']:9s} {r['year']} | {len(d)} dana | GDD={d['GDD_cum'].iat[-1]:.0f} | "
              f"P={d['prec_cum'].iat[-1]:.0f} mm{'' if akum else ' (tp nije akumuliran)'}")
    df = pd.concat(sve, ignore_index=True)
    df.to_csv(put("04_era5_daily_per_parcel.csv"), index=False)
    print(f"\nSacuvano: {len(df):,} dana | {df.groupby(['parcel', 'year']).ngroups} parcela-sezona")
    print(df.groupby(["year", "crop", "parcel"])[["GDD_cum", "prec_cum"]].last()
          .groupby(["year", "crop"]).mean().round(0).to_string())
