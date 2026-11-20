# -*- coding: utf-8 -*-
"""
Zbirni rezultati Exp1 (LOPO u godini) i Exp2 (LOYO, pixel/parcel) iz <MODEL>_all_predictions.csv.
Metrike se racunaju iz predvidjanja: R2, RMSE, nRMSE, MAE, bias, udeo biasa u MSE, R2 bez biasa, Pearson.
Izlaz: jedan Excel (ukupno, pivot tabele, po parceli).
"""
import os, glob
import numpy as np
import pandas as pd

EXP1 = r"D:\PhD ML models\Proces\Exp1_Dv4_results"     # <kultura>/<godina>/<MODEL>_all_predictions.csv
EXP2 = r"D:\PhD ML models\Proces\Exp2_Dv4_results"     # <kultura>_all_years/<pixel|parcel>/<MODEL>_all_predictions.csv
IZLAZ = r"D:\PhD ML models\Proces\zbirni_rezultati_Dv4.xlsx"
# Exp2 pusten pre ispravke train.py: oznaka godine je pomerena (2018 -> 2019, 2019 -> 2021). False posle ispravke.
ISPRAVI_GODINU_EXP2 = {"soybeans", "sunflower"}   # kulture ciji je Exp2 pusten PRE ispravke train.py
GODINE = {"wheat": [2017, 2018, 2019, 2021], "sunflower": [2017, 2018, 2019, 2021],
          "soybeans": [2017, 2018, 2019, 2021], "corn": [2018, 2019]}


def metrike(x):
    y, p = x["y_test"].to_numpy(float), x["y_pred"].to_numpy(float)
    e, ss = p - y, ((y - y.mean()) ** 2).sum()
    mse = (e ** 2).mean()
    return pd.Series({"n": len(y), "R2": 1 - (e ** 2).sum() / ss if ss > 0 else np.nan,
                      "RMSE": np.sqrt(mse), "nRMSE_pct": 100 * np.sqrt(mse) / y.mean(), "MAE": np.abs(e).mean(),
                      "bias": e.mean(), "udeo_biasa": e.mean() ** 2 / mse if mse > 0 else np.nan,
                      "R2_bez_biasa": 1 - ((e - e.mean()) ** 2).sum() / ss if ss > 0 else np.nan,
                      "Pearson": np.corrcoef(y, p)[0, 1] if len(y) > 2 and y.std() > 0 and p.std() > 0 else np.nan,
                      "y_mean": y.mean(), "pred_mean": p.mean()})


delovi = []
for f in glob.glob(os.path.join(EXP1, "*", "*", "*_all_predictions.csv")):
    crop, god = f.split(os.sep)[-3:-1]
    d = pd.read_csv(f)[["parcel", "y_test", "y_pred"]]
    delovi.append(d.assign(exp="Exp1_LOPO", crop=crop, nivo="pixel", godina=int(god),
                           model=os.path.basename(f).replace("_all_predictions.csv", "")))
for f in glob.glob(os.path.join(EXP2, "*_all_years", "*", "*_all_predictions.csv")):
    crop, nivo = f.split(os.sep)[-3:-1]
    crop = crop.replace("_all_years", "")
    d = pd.read_csv(f)
    god = d["year"].astype(int)
    if crop in ISPRAVI_GODINU_EXP2:
        g = GODINE[crop]
        god = god.map(lambda v: g[g.index(v) + 1])
    delovi.append(d[["parcel", "y_test", "y_pred"]].assign(exp="Exp2_LOYO", crop=crop, nivo=nivo, godina=god.values,
                                                           model=os.path.basename(f).replace("_all_predictions.csv", "")))
    print(f"  {f}: {len(d):,} redova")
P = pd.concat(delovi, ignore_index=True)
print(f"Ucitano {len(delovi)} fajlova, {len(P):,} predvidjanja")

K = ["exp", "crop", "nivo", "godina", "model"]
po_god = P.groupby(K).apply(metrike).reset_index()                                     # svaka test godina
sve2 = P[P["exp"] == "Exp2_LOYO"].groupby(["exp", "crop", "nivo", "model"]).apply(metrike).reset_index().assign(godina="sve")
ukupno = pd.concat([po_god.astype({"godina": str}), sve2], ignore_index=True)[K + list(po_god.columns[len(K):])]
po_parc = P[P["nivo"] == "pixel"].groupby(K + ["parcel"]).apply(metrike).reset_index()  # parcela nema smisla na nivou parcele

pv = lambda m: ukupno.pivot_table(index=["exp", "crop", "nivo", "godina"], columns="model", values=m).round(3)
with pd.ExcelWriter(IZLAZ) as w:
    ukupno.round(4).to_excel(w, sheet_name="ukupno", index=False)
    for m in ["R2", "RMSE", "nRMSE_pct", "Pearson", "bias", "R2_bez_biasa"]:
        pv(m).to_excel(w, sheet_name=f"pivot_{m}")
    po_parc.round(4).to_excel(w, sheet_name="po_parceli", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(f"\n-> {IZLAZ}\n\nR2:\n{pv('R2').to_string()}\n\nPearson:\n{pv('Pearson').to_string()}")