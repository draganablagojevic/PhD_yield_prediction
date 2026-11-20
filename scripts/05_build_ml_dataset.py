# -*- coding: utf-8 -*-
"""
05 - ML dataset u termalnom vremenu (GDD od setve, ERA5 po parceli).
  A) BBCH faze Phase3-5 definisane GDD pragovima -> <INDEKS>_<Faza>_mean/max/min/std + n_obs_<Faza>
     pragovi: GDD_FAZE (rucno) ili kalibracija: stare frakcije sezone (phenology.py) na referentnoj godini -> GDD, medijana parcela
  B) fenometrika na GLAVNI indeksu: PCHIP po GDD osi, POS u prozoru Phase3-Phase5 (od pocetka Phase3 do zetve); SOS/POS/EOS u GDD i datumu,
     indeksi na POS-u, kontrola kvaliteta (gap_POS_days, POS_edge); nepouzdana fenometrika -> NaN
  C) klima: sezonski ERA5 + po fazama (padavine, Tmax, dani > T_STRES, trajanje faze u danima)
  Snimci se nikad ne interpoliraju u fazne statistike: faza bez snimka = NaN (n_obs = 0).
Ulaz : OUT/03_pixels_ts_<godina>.csv, OUT/04_era5_daily_per_parcel.csv
Izlaz: OUT/05_ml_pixel_dataset.csv, OUT/05_ml_parcel_dataset.csv, OUT/05_gdd_faze.csv
"""
import os, re, warnings
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from common import *
warnings.simplefilter("ignore", RuntimeWarning)

GLAVNI   = "GNDVI"
INDEKSI  = ["ARVI", "B02", "B03", "B04", "B08", "B11", "CVI", "DVI", "EVI", "GCI", "GNDVI", "IPVI",
            "MSAVI2", "NDMI", "NDVI", "OSAVI", "SAVI", "SR", "TVI", "WDRVI"]
FAZE     = ["Phase3", "Phase4", "Phase5"]
SKALA    = {"wheat": "dani", "sunflower": "dani", "corn": "gdd", "soybeans": "gdd"}
PRAGOVI  = {"wheat": (150, 185, 220),       # dani od setve: vlatanje-butanje / klasanje-cvetanje (POS) / nalivanje-zrenje
            "sunflower": (40, 78, 100),     # dani od setve: butonizacija / cvetanje R5 (POS) / nalivanje-zrenje
            "corn": (264, 631, 1040),       # GDD 10/30 °C: V6 / VT-R1 (NCH-40) / R3 = R1 (778) + 20 dana (NDSU)
            "soybeans": (400, 684, 1084)}   # GDD 10/30 °C: vegetativni porast / R1-R5 (POS +-200) / R6-R8
STD_MIN  = 3           # std samo ako faza ima >= 3 snimka
MAX_GAP  = 30          # [dana] maks. razmak do susednog snimka oko POS-a
SOS_FRAC = 0.2         # SOS/EOS prag = min + 0.2 * amplituda
T_STRES  = 30.0        # [°C] toplotni stres
MIROVANJE = {}               # npr. {"wheat": 5.0}: zimsko mirovanje bez GDD; iskljuceno jer je psenica po danima (opcija B)
MIROVANJE_DANA = 14          # prolecni prelaz je stabilan ako 5-dnevni prosek ostane iznad praga >= 14 dana (topli januar ne prekida mirovanje)
DATUM_RE = re.compile(r"^(.+)_(\d{8})$")
FENO = ["SOS", "POS", "EOS", "SOS_GDD", "POS_GDD", "EOS_GDD", "SOS_date", "POS_date", "EOS_date",
        "Season_Length_days", "Season_Length_GDD", "Seasonal_Integral", "Max_Slope_100GDD",
        "POS_days_from_sowing", "prec_to_POS_mm", "gap_POS_days", "POS_edge"]


def mirovanje(t, year, prag, n_dana):
    """True za dane zimskog mirovanja: od prvog dana kad 5-dnevni prosek Tmean padne ispod praga do stabilnog
    prolecnog prelaza: prvi dan posle 1.1. (godina zetve) od kog 5-dnevni prosek ostaje iznad praga >= n_dana uzastopno."""
    t5 = t.rolling(5).mean()
    ispod = t5.index[t5 < prag]
    if not len(ispod):
        return pd.Series(False, t.index)
    od = ispod[0]
    iznad = (t5 > prag) & (t5.index >= max(od, pd.Timestamp(year, 1, 1)))
    niz = iznad[::-1].groupby((~iznad[::-1]).cumsum()).cumsum()[::-1]   # broj uzastopnih dana iznad praga od tog dana
    do = niz.index[niz >= n_dana][0] if (niz >= n_dana).any() else t.index[-1]
    return pd.Series((t.index >= od) & (t.index < do), t.index)

def ucitaj_era5():
    e = pd.read_csv(put("04_era5_daily_per_parcel.csv"), parse_dates=["date", "sowing_date", "harvest_date"])
    era, rows = {}, []
    for (p, y), g in e.groupby(["parcel", "year"]):
        g = g.set_index("date")
        m = mirovanje(g["Tmean_C"], y, MIROVANJE[g["crop"].iat[0]], MIROVANJE_DANA) if g["crop"].iat[0] in MIROVANJE else pd.Series(False, g.index)
        g["GDD_day"] = g["GDD_day"].where(~m, 0.0)
        g["GDD_cum"] = g["GDD_day"].cumsum()
        g["T_faze"] = (g.index - g["sowing_date"].iat[0]).days if SKALA[g["crop"].iat[0]] == "dani" else g["GDD_cum"]
        era[(p, y)] = g
        rows.append({"parcel": p, "year": y, "crop": g["crop"].iat[0], "sowing_date": g["sowing_date"].iat[0],
                     "harvest_date": g["harvest_date"].iat[0], "GDD_total": g["GDD_day"].sum(),
                     "prec_total_mm": g["prec_mm"].sum(), "Tmax_mean": g["Tmax_C"].mean(), "Tmean_mean": g["Tmean_C"].mean(),
                     "dormancy_days": int(m.sum()), "_od": m.idxmax() if m.any() else pd.NaT,
                     "_do": m[::-1].idxmax() if m.any() else pd.NaT})
    sez = pd.DataFrame(rows).set_index(["parcel", "year"])
    z = sez[sez["dormancy_days"] > 0]
    if len(z):
        print("Zimsko mirovanje (medijana po godini):\n" + z.groupby(level="year").agg(
            od=("_od", "median"), do=("_do", "median"), dana=("dormancy_days", "median")).to_string() + "\n")
    return era, sez.drop(columns=["_od", "_do"] + ([] if MIROVANJE else ["dormancy_days"]))

def kalibrisi():
    t = pd.DataFrame([[c, SKALA[c], *p] for c, p in PRAGOVI.items()], columns=["crop", "skala", *FAZE]).set_index("crop")
    t.to_csv(put("05_gdd_faze.csv"))
    print(f"Pragovi faza (pocetak faze; dani ili °C·dan od setve):\n{t.to_string()}\n")
    return t


def fazne_statistike(g, tag, faza):
    out = {}
    for f in FAZE:
        cols = [t for t, ff in zip(tag, faza) if ff == f]
        for ind in INDEKSI:
            M = g.reindex(columns=[f"{ind}_{t}" for t in cols]).to_numpy(float)
            n = np.isfinite(M).sum(1)
            out[f"{ind}_{f}_mean"] = np.nanmean(M, 1) if cols else np.nan
            out[f"{ind}_{f}_max"] = np.nanmax(M, 1) if cols else np.nan
            out[f"{ind}_{f}_min"] = np.nanmin(M, 1) if cols else np.nan
            out[f"{ind}_{f}_std"] = np.where(n >= STD_MIN, np.nanstd(M, 1, ddof=1), np.nan) if cols else np.nan
        out[f"n_obs_{f}"] = np.isfinite(g.reindex(columns=[f"{GLAVNI}_{t}" for t in cols]).to_numpy(float)).sum(1)
    return pd.DataFrame(out, index=g.index)


def klima_po_fazama(d, prag):
    out, gc = {}, d["T_faze"]
    for f, lo, hi in zip(FAZE, prag, list(prag[1:]) + [np.inf]):
        m = (gc >= lo) & (gc < hi)
        out.update({f"days_{f}": int(m.sum()), f"prec_{f}": d.loc[m, "prec_mm"].sum(),
                    f"Tmax_{f}": d.loc[m, "Tmax_C"].mean(), f"hot_days_{f}": int((d.loc[m, "Tmax_C"] > T_STRES).sum())})
    return out


def fenometrika(v, G, dat, gd, dd, prec, win, sow):
    ok = np.isfinite(v)
    if ok.sum() < 4:
        return None
    x, inv = np.unique(G[ok], return_inverse=True)
    y = np.bincount(inv, v[ok]) / np.bincount(inv)
    if len(x) < 4:
        return None
    m = (gd >= x[0]) & (gd <= x[-1])
    g, days = gd[m], dd[m]
    f = PchipInterpolator(x, y)(g)
    w = np.flatnonzero((g >= win[0]) & (g < win[1]))
    if len(w) < 3:
        return None
    ip = w[np.argmax(f[w])]
    edge = ip in (w[0], w[-1])
    vd = dat[ok]
    pre, post = vd[vd < days[ip]], vd[vd > days[ip]]
    gap = max((days[ip] - pre.max()).days if len(pre) else 999, (post.min() - days[ip]).days if len(post) else 999)
    thr = f.min() + SOS_FRAC * (f[ip] - f.min())
    i_s = int(np.argmax(f[:ip + 1] >= thr))
    pad = np.flatnonzero(f[ip:] <= thr)
    i_e = ip + int(pad[0]) if len(pad) else len(f) - 1
    gu, iu = np.unique(g, return_index=True)
    rast = np.interp(g[i_s:ip + 1] + 100, gu, f[iu]) - f[i_s:ip + 1]   # porast krive na narednih 100 GDD (SOS -> POS)    j = np.argmin(np.abs((vd - days[ip]).days))
    j = np.argmin(np.abs((vd - days[ip]).days))
    return dict(SOS=f[i_s], POS=f[ip], EOS=f[i_e], SOS_GDD=g[i_s], POS_GDD=g[ip], EOS_GDD=g[i_e],
                SOS_date=days[i_s], POS_date=days[ip], EOS_date=days[i_e],
                Season_Length_days=(days[i_e] - days[i_s]).days, Season_Length_GDD=g[i_e] - g[i_s],
                Seasonal_Integral=f[i_s:i_e + 1].sum(),
                Max_Slope_100GDD=rast.max() if len(rast) else np.nan,
                POS_days_from_sowing=(days[ip] - sow).days, prec_to_POS_mm=prec[:np.searchsorted(dd, days[ip]) + 1].sum(),
                gap_POS_days=gap, POS_edge=edge, _j=np.flatnonzero(ok)[j])


def obradi_parcelu(g, year, crop, datumi, era, sez, prag):
    p = g["parcel"].iat[0]
    key = (ALIAS.get(p, p), year)
    if key not in era:
        print(f"  {p}: nema ERA5/datuma setve -> preskacem ({len(g)} piksela)"); return None
    d, s = era[key], sez.loc[key]
    G, TF = d["GDD_cum"].reindex(datumi).to_numpy(), d["T_faze"].reindex(datumi).to_numpy()
    ok = np.isfinite(G)
    dat, G, TF = pd.DatetimeIndex(np.array(datumi)[ok]), G[ok], TF[ok]
    tag = [x.strftime("%Y%m%d") for x in dat]
    faza = np.select([TF >= prag[2], TF >= prag[1], TF >= prag[0]], FAZE[::-1], "pre")
    p3 = d.loc[d["T_faze"] >= prag[0], "GDD_cum"]
    win = (p3.iat[0] if len(p3) else np.inf, np.inf)   # POS se trazi od pocetka Phase3 (u GDD) do zetve
    out = fazne_statistike(g, tag, faza)
    V = g.reindex(columns=[f"{GLAVNI}_{t}" for t in tag]).to_numpy(float)
    gd, dd, pc = d["GDD_cum"].to_numpy(), d.index, d["prec_mm"].to_numpy()
    rec, at_pos = [], {ind: np.full(len(g), np.nan) for ind in INDEKSI if ind != GLAVNI}
    for i, v in enumerate(V):
        r = fenometrika(v, G, dat, gd, dd, pc, (win), s["sowing_date"])
        if r is None:
            rec.append({}); continue
        j = r.pop("_j")
        if r["POS_edge"] or r["gap_POS_days"] > MAX_GAP:
            r.update({k: np.nan for k in FENO if k not in ("gap_POS_days", "POS_edge")})
        else:
            for ind in at_pos:
                c = f"{ind}_{tag[j]}"
                at_pos[ind][i] = g[c].iat[i] if c in g else np.nan
        rec.append(r)
    feno = pd.DataFrame(rec, index=g.index).reindex(columns=FENO)
    feno = feno.assign(**{f"{ind}_at_POS": a for ind, a in at_pos.items()})
    for k, v in {**s.drop("crop").to_dict(), **klima_po_fazama(d, prag)}.items():
        feno[k] = v
    return pd.concat([out, feno], axis=1)


def obradi_godinu(year, era, sez, pragovi):
    df = pd.read_csv(put(f"03_pixels_ts_{year}.csv"))
    df = df[[not iskljucen(p, year) for p in df["parcel"]]].reset_index(drop=True)
    vremenske = [c for c in df.columns if DATUM_RE.match(c)]
    datumi = sorted({pd.to_datetime(DATUM_RE.match(c)[2]) for c in vremenske})
    print(f"{year}: {len(df):,} piksela | {len(datumi)} datuma")
    parts = [r for (_, crop), g in df.groupby(["parcel", "crop"])
             if (r := obradi_parcelu(g, year, crop, datumi, era, sez, pragovi.loc[crop, FAZE].to_numpy(float)))
             is not None]
    feat = pd.concat(parts)
    out = df.drop(columns=vremenske).join(feat, how="inner")
    n_obs = out[[f"n_obs_{f}" for f in FAZE]].sum(axis=1)
    pouzdano = out["POS"].notna()
    print(f"  bez ijednog snimka u Phase3-5: {(n_obs == 0).sum():,} -> izbaceni | "
          f"pouzdana fenometrika: {pouzdano.mean() * 100:.1f}% piksela")
    print("  % piksela bez snimka po fazi:\n" + out.groupby("crop")[[f"n_obs_{f}" for f in FAZE]]
          .agg(lambda s: round((s == 0).mean() * 100, 1)).to_string())
    return out[n_obs > 0]


if __name__ == "__main__":
    era, sez = ucitaj_era5()
    pragovi = kalibrisi()
    df = pd.concat([obradi_godinu(y, era, sez, pragovi) for y in YEARS if os.path.exists(put(f"03_pixels_ts_{y}.csv"))],
                   ignore_index=True).sort_values(["year", "parcel"]).reset_index(drop=True)
    t = np.where(df["crop"].map(SKALA) == "dani", df["POS_days_from_sowing"], df["POS_GDD"])
    pr = pragovi.loc[df["crop"], FAZE].to_numpy(float)
    poz = pd.Series(np.select([t < pr[:, 1], t < pr[:, 2]], ["pre Phase4", "Phase4"], "posle Phase4"), index=df.index).where(df["POS"].notna())
    print("\nProvera: gde pada POS (% piksela sa pouzdanom fenometrikom; cilj = Phase4):")
    print(pd.crosstab([df["crop"], df["year"]], poz, normalize="index").mul(100).round(1).to_string())
    df.to_csv(put("05_ml_pixel_dataset.csv"), index=False)
    num = df.select_dtypes("number").columns.drop("year")
    parc = df.groupby(["year", "parcel", "crop"])[list(num)].mean()
    parc["n_pixels"] = df.groupby(["year", "parcel", "crop"]).size()
    parc.reset_index().to_csv(put("05_ml_parcel_dataset.csv"), index=False)
    print(f"\nML pixel: {df.shape[0]:,} x {df.shape[1]} | ML parcel: {len(parc)} redova")
    print(df.groupby(["year", "crop"])["yield"].describe().round(2).to_string())
