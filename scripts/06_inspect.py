# -*- coding: utf-8 -*-
"""
06 - Inspekcija svih faza: prinos (tacke i pikseli), NaN, min/max i fizicki opsezi, ERA5 logika,
     fenometrika (POS van sezone, rani POS psenice), tekstura tla, broj piksela, korelacije sa prinosom.
Izlaz: OUT/06_inspekcija/ (izvestaj.xlsx, PNG grafici, sumnjivi_POS_qgis.csv)
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

DIR = put("06_inspekcija"); os.makedirs(DIR, exist_ok=True)
CROPS = ["wheat", "corn", "sunflower", "soybeans"]
BOJE = {2017: "#4472C4", 2018: "#ED7D31", 2019: "#70AD47", 2021: "#C00000"}
CV_PRAG = 0.25
tabele = {}


def citaj(ime):
    p = put(ime)
    return pd.read_csv(p, low_memory=False) if os.path.exists(p) else None


def sacuvaj(fig, ime):
    fig.tight_layout(); fig.savefig(os.path.join(DIR, ime), dpi=150, bbox_inches="tight"); plt.close(fig)


def prinos(df, nivo):
    s = df.groupby(["crop", "year"])["yield"].describe().round(2)
    v = df.groupby(["crop", "year", "parcel"])["yield"].agg(["count", "mean", "std", "min", "max"]).round(3)
    v["CV"] = (v["std"] / v["mean"]).round(3)
    lo_hi = df.apply(lambda r: agro(r["crop"], r["year"]), axis=1, result_type="expand")
    van = ((df["yield"] < lo_hi[0]) | (df["yield"] > lo_hi[1])).groupby([df["crop"], df["year"]]).sum()
    s["van_agro"] = van
    tabele[f"prinos_{nivo}"], tabele[f"CV_parcela_{nivo}"] = s, v
    print(f"\n--- Prinos ({nivo}) ---\n{s.to_string()}")
    print(f"Parcele CV>{CV_PRAG}: {(v['CV'] > CV_PRAG).sum()} / {len(v)}")


def grafici_prinos(df):
    cr = [c for c in CROPS if c in df["crop"].unique()]
    fig, ax = plt.subplots(2, len(cr), figsize=(5 * len(cr), 9), squeeze=False)
    for j, c in enumerate(cr):
        sub = df[df["crop"] == c]; god = sorted(sub["year"].unique())
        ax[0, j].boxplot([sub.loc[sub["year"] == y, "yield"] for y in god], tick_labels=[str(y) for y in god])
        for y in god:
            ax[1, j].hist(sub.loc[sub["year"] == y, "yield"], bins=40, alpha=0.5, color=BOJE.get(y), label=str(y))
        for v in AGRO[c]:
            ax[0, j].axhline(v, color="r", ls="--"); ax[1, j].axvline(v, color="r", ls="--")
        ax[0, j].set_title(c); ax[0, j].set_ylabel("t/ha"); ax[1, j].legend(fontsize=8); ax[1, j].set_xlabel("t/ha")
    sacuvaj(fig, "prinos_box_hist.png")
    fig, ax = plt.subplots(1, len(cr), figsize=(5 * len(cr), 8), squeeze=False)
    for j, c in enumerate(cr):
        pv = df[df["crop"] == c].groupby(["parcel", "year"])["yield"].mean().unstack()
        im = ax[0, j].imshow(pv.values, aspect="auto", cmap="RdYlGn", vmin=AGRO[c][0], vmax=AGRO[c][1])
        ax[0, j].set_xticks(range(pv.shape[1]), pv.columns); ax[0, j].set_yticks(range(pv.shape[0]), pv.index, fontsize=7)
        for (i, k), val in np.ndenumerate(pv.values):
            if np.isfinite(val): ax[0, j].text(k, i, f"{val:.1f}", ha="center", va="center", fontsize=6)
        ax[0, j].set_title(c); plt.colorbar(im, ax=ax[0, j], shrink=0.7)
    sacuvaj(fig, "prinos_heatmap_parcela.png")


def kolone(df):
    num = df.select_dtypes("number")
    t = pd.DataFrame({"nan_pct": (num.isna().mean() * 100).round(2), "min": num.min(), "max": num.max(),
                      "mean": num.mean(), "neg": (num < 0).sum()})
    def van(c):
        if not any(c.lower().startswith(k) for k in RASPON):
            return np.nan
        lo, hi = raspon(c)
        return int((num[c] < lo).sum() if lo is not None else 0) + int((num[c] > hi).sum() if hi is not None else 0)
    t["van_opsega"] = [van(c) for c in num]
    tabele["kolone"] = t.round(4)
    print(f"\n--- Kolone sa NaN ---\n{t[t['nan_pct'] > 0]['nan_pct'].to_string()}")
    print(f"--- Van fizickog opsega ---\n{t[t['van_opsega'] > 0]['van_opsega'].to_string()}")


def logika(df):
    k = {}
    for c in ["POS_date", "sowing_date", "harvest_date"]:
        df[c] = pd.to_datetime(df.get(c), errors="coerce")
    k["POS van setva-zetva"] = ((df["POS_date"] < df["sowing_date"]) | (df["POS_date"] > df["harvest_date"])).sum()
    k["POS na granici prozora (fenometrika NaN)"] = df["POS_edge"].astype(bool).sum()
    k["gap_POS > 30 dana (fenometrika NaN)"] = (df["gap_POS_days"] > 30).sum()
    k["Season_Length_days < 30"] = (df["Season_Length_days"] < 30).sum()
    k["SOS >= POS"] = (df["SOS"] >= df["POS"]).sum()
    k["POS_GDD > GDD_total"] = (df["POS_GDD"] - df["GDD_total"] > 0.01).sum()
    k["prec_to_POS > prec_total"] = (df["prec_to_POS_mm"] - df["prec_total_mm"] > 0.01).sum()
    k["Tmax_mean < Tmean_mean"] = (df["Tmax_mean"] < df["Tmean_mean"]).sum()
    k["prec_total_mm > 1500 (sumnjivo)"] = (df["prec_total_mm"] > 1500).sum()
    tex = [c for c in df if c.split("_")[0] in ("clay", "sand", "silt")]
    if len(tex) == 3:
        k["|clay+sand+silt - 100| > 5"] = ((df[tex].sum(axis=1, min_count=3) - 100).abs() > 5).sum()
    tabele["logicke_provere"] = pd.Series(k, name="n").to_frame()
    print(f"\n--- Logicke provere ---\n{tabele['logicke_provere'].to_string()}")
    tabele["n_piksela"] = df.groupby(["year", "crop"]).size().unstack(fill_value=0)
    print(f"\n--- Broj piksela ---\n{tabele['n_piksela'].to_string()}")
    nobs = [c for c in df if c.startswith("n_obs_")]
    tabele["faze_bez_snimka_pct"] = df.groupby(["year", "crop"])[nobs].agg(lambda s: round((s == 0).mean() * 100, 1))
    tabele["faze_n_obs_medijana"] = df.groupby(["year", "crop"])[nobs].median()
    tabele["fenometrika_kvalitet"] = df.groupby(["year", "crop"]).agg(
        pouzdano_pct=("POS", lambda s: round(s.notna().mean() * 100, 1)),
        gap_POS_medijana=("gap_POS_days", "median"), POS_GDD_medijana=("POS_GDD", "median"))
    for t in ("faze_bez_snimka_pct", "fenometrika_kvalitet"):
        print(f"\n--- {t} ---\n{tabele[t].to_string()}")
    bad = df[df["POS_edge"].astype(bool) | (df["gap_POS_days"] > 30)]
    bad[["parcel", "year", "crop", "X", "Y", "gap_POS_days", "POS_edge", "yield"]].to_csv(
        os.path.join(DIR, "sumnjivi_POS_qgis.csv"), index=False)


def korelacije(df):
    feat = [c for c in df.select_dtypes("number") if c not in ("yield", "year", "X", "Y", "n_points")]
    r = pd.DataFrame({c: df[df["crop"] == c][feat + ["yield"]].corr("spearman")["yield"].drop("yield")
                      for c in CROPS if (df["crop"] == c).sum() > 30})
    tabele["spearman_yield"] = r.round(3)
    fig, ax = plt.subplots(1, r.shape[1], figsize=(5 * r.shape[1], 7), squeeze=False)
    for j, c in enumerate(r):
        top = r[c].dropna().reindex(r[c].abs().sort_values(ascending=False).index).dropna().head(20)[::-1]
        ax[0, j].barh(top.index, top.values, color=np.where(top > 0, "#2E8B57", "#C00000"))
        ax[0, j].set_title(f"{c}: Spearman r sa prinosom"); ax[0, j].tick_params(labelsize=7)
    sacuvaj(fig, "korelacije_top20.png")


def profili(indeks="GNDVI"):
    fig, ax = plt.subplots(len(CROPS), 1, figsize=(14, 3.2 * len(CROPS)), sharex=False)
    for y in YEARS:
        df = citaj(f"02_pixels_{y}.csv")
        if df is None: continue
        cols = sorted((c for c in df if re.match(rf"^{indeks}_\d{{8}}$", c)), key=lambda c: c[-8:])
        doy = [pd.to_datetime(c[-8:]).dayofyear for c in cols]
        for a, c in zip(ax, CROPS):
            sub = df[df["crop"] == c]
            if len(sub):
                m, s = sub[cols].mean(), sub[cols].std()
                a.plot(doy, m, "o-", ms=3, color=BOJE.get(y), label=str(y))
                a.fill_between(doy, m - s, m + s, alpha=0.12, color=BOJE.get(y))
                a.set_title(f"{c} - {indeks} (mean +/- std piksela)", fontsize=9); a.set_xlabel("DOY")
    for a in ax:
        a.grid(alpha=0.3)
        if a.lines: a.legend(fontsize=7)
    sacuvaj(fig, f"profil_{indeks}.png")


if __name__ == "__main__":
    pts = [d for y in YEARS if (d := citaj(f"01_yield_points_{y}.csv")) is not None]
    if pts: prinos(pd.concat(pts), "tacke")
    log = citaj("01_yield_filter_log.csv")
    if log is not None: tabele["filter_log_YM"] = log
    ml = citaj("05_ml_pixel_dataset.csv")
    if ml is not None:
        prinos(ml, "pikseli"); grafici_prinos(ml); kolone(ml); logika(ml); korelacije(ml)
    profili()
    for ime in ["02_scene_log.csv", "03_terrain_soil_log.csv"]:
        if (d := citaj(ime)) is not None: tabele[ime[:-8]] = d
    with pd.ExcelWriter(os.path.join(DIR, "izvestaj.xlsx")) as w:
        for ime, t in tabele.items():
            t.to_excel(w, sheet_name=ime[:31])
    print(f"\nIzvestaj i grafici: {DIR}")
