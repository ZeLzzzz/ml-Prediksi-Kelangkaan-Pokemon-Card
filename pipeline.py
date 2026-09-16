"""Prediksi Kartu Pokemon TCG Ultra Rare - pipeline lengkap untuk tugas ML.

Dataset : pokemon-tcg-data-master 1999-2023.csv (17.172 kartu, 1999-2023)
Subset  : hanya kartu bersupertype "Pokemon" (14.497 baris)
Target  : is_ultra_rare (1 = Secret/Rainbow/Illustration/VMAX/GX/EX/dll)

Jalankan: python pipeline.py
Output  : figures/*.png (bahan PPT), model.joblib (untuk app.py), hasil.md
"""
import ast
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

DATA = "pokemon-tcg-data-master 1999-2023.csv"
TARGET = "is_ultra_rare"
# Pola nama rarity yang dihitung sebagai ultra rare.
POLA_ULTRA = (r"Ultra|Secret|Rainbow|Illustration|Shiny|VMAX|VSTAR|GX|EX\b"
              r"|Amazing|Radiant|Double Rare|Hyper|Prime|LEGEND|Star")
# Kolom yang SENGAJA dibuang: isinya mekanik kartu (V/VMAX/GX/EX) yang juga
# tertulis di nama rarity, jadi modelnya cuma mencocokkan label. Lihat bocor().
BOCOR = ["subtypes", "rules"]

NUM = ["hp", "convertedRetreatCost", "n_attacks", "max_damage", "total_energy",
       "n_types", "panjang_teks", "punya_ability", "punya_flavor", "tahun"]
KAT = ["type_1", "weak_type", "artist", "set", "series", "regulationMark"]
SKOR = ["accuracy", "precision", "recall", "f1", "roc_auc"]
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
CV5 = StratifiedKFold(5, shuffle=True, random_state=42)
# CV internal Stacking WAJIB shuffle: data CSV urut kronologis 1999->2023.
# Tanpa shuffle, meta-learner dilatih lintas era dan kolaps (f1 0.02).
CV3 = StratifiedKFold(3, shuffle=True, random_state=42)
# ponytail: 4 worker, bukan -1. Di mesin 16-core tiap worker memegang salinan
# matriks one-hot -> CPU 100% dan WinError 1450. Data cuma 14k baris, 4 cukup.
NJOB = 4


def simpan(nama):
    plt.tight_layout()
    plt.savefig(FIG / f"{nama}.png", dpi=150, bbox_inches="tight")
    plt.close()


def lit(s):
    """Kolom list/dict tersimpan sebagai string literal Python."""
    try:
        return ast.literal_eval(s) if isinstance(s, str) else []
    except (ValueError, SyntaxError):
        return []


def angka(x):
    """'30', '120+', '×2' -> 30.0, 120.0, 2.0. Damage bisa punya imbuhan."""
    digit = re.sub(r"\D", "", str(x or ""))
    return float(digit) if digit else 0.0


# ---------- 1. Load & feature engineering ----------
def muat(path=DATA):
    d = pd.read_csv(path, low_memory=False)
    p = d[d.supertype == "Pokémon"].copy()

    at = p["attacks"].map(lit)
    p["n_attacks"] = at.map(len)
    p["max_damage"] = at.map(lambda a: max([angka(x.get("damage")) for x in a],
                                           default=0.0))
    p["total_energy"] = at.map(
        lambda a: sum(x.get("convertedEnergyCost") or 0 for x in a))
    p["panjang_teks"] = at.map(
        lambda a: sum(len(str(x.get("text") or "")) for x in a))

    tipe = p["types"].map(lit)
    p["n_types"] = tipe.map(len)
    p["type_1"] = tipe.map(lambda l: l[0] if l else "none")
    p["weak_type"] = p["weaknesses"].map(lit).map(
        lambda l: l[0].get("type", "none") if l else "none")

    # NaN di sini artinya "kartu tidak punya", bukan data hilang.
    p["punya_ability"] = p["abilities"].notna().astype(int)
    p["punya_flavor"] = p["flavorText"].notna().astype(int)
    p["regulationMark"] = p["regulationMark"].fillna("none")
    p["tahun"] = pd.to_datetime(p["release_date"], errors="coerce").dt.year

    p[TARGET] = p["rarity"].fillna("").str.contains(
        POLA_ULTRA, case=False, regex=True).astype(int)
    # Acak urutan baris: CSV aslinya urut rilis 1999->2023, dan CV apa pun yang
    # lupa shuffle akan melatih lintas era. Sekali di sini, aman untuk semua.
    return p.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------- 2. EDA ----------
def eda(d):
    y = d[TARGET]

    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x=y)
    for c in ax.containers:
        ax.bar_label(c)
    ax.set(title=f"Distribusi Target (minoritas {y.mean():.1%})",
           xlabel="0 = Biasa   |   1 = Ultra Rare", ylabel="Jumlah kartu")
    simpan("01_distribusi_target")

    penting = ["hp", "max_damage", "total_energy", "convertedRetreatCost",
               "panjang_teks", "n_attacks"]
    f, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, k in zip(axes.flat, penting):
        sns.kdeplot(data=d, x=k, hue=TARGET, fill=True, common_norm=False, ax=ax)
        ax.set_title(k)
    f.suptitle("Distribusi Fitur: Ultra Rare vs Biasa", fontsize=13)
    simpan("02_distribusi_fitur")

    plt.figure(figsize=(9, 7))
    sns.heatmap(d[NUM].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                annot_kws={"size": 7})
    plt.title("Matriks Korelasi Fitur Numerik")
    simpan("03_korelasi")

    plt.figure(figsize=(10, 4.5))
    tren = d.groupby("tahun")[TARGET].agg(["mean", "size"])
    tren = tren[tren["size"] >= 30]
    plt.plot(tren.index, tren["mean"], marker="o", lw=2)
    plt.title("Proporsi Kartu Ultra Rare per Tahun Rilis (1999-2023)")
    plt.ylabel("proporsi ultra rare")
    plt.xlabel("tahun")
    simpan("04_tren_tahun")

    f, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    urut = d.groupby("series")[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=urut.values, y=urut.index, ax=a1)
    a1.set(title="Proporsi Ultra Rare per Series", xlabel="proporsi")
    top = d.artist.value_counts().head(15).index
    ua = d[d.artist.isin(top)].groupby("artist")[TARGET].mean().sort_values()
    sns.barplot(x=ua.values, y=ua.index, ax=a2)
    a2.set(title="Proporsi Ultra Rare - 15 Artist Terproduktif", xlabel="proporsi")
    simpan("05_series_artist")


def bocor(d):
    """Bukti kenapa subtypes & rules dibuang: keduanya nyaris menyebut label.

    Dipakai sebagai satu slide metodologi di PPT.
    """
    y = d[TARGET]
    skor = {}
    for kol in BOCOR + ["artist", "set", "hp"]:
        X = d[[kol]].astype(str).fillna("NA")
        m = Pipeline([("e", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                      ("c", RandomForestClassifier(n_estimators=100, random_state=42,
                                                   n_jobs=1))])
        skor[kol] = cross_val_score(m, X, y, cv=CV5, scoring="roc_auc",
                                    n_jobs=NJOB).mean()
    s = pd.Series(skor).sort_values()

    plt.figure(figsize=(7, 4))
    warna = ["#d62728" if k in BOCOR else "#1f77b4" for k in s.index]
    plt.barh(s.index, s.values, color=warna)
    plt.axvline(.5, ls="--", c="gray", lw=1)
    for i, v in enumerate(s.values):
        plt.text(v + .01, i, f"{v:.3f}", va="center", fontsize=9)
    plt.xlim(0, 1.05)
    plt.xlabel("ROC-AUC memakai SATU kolom saja")
    plt.title("Deteksi Kebocoran Data (merah = dibuang)")
    simpan("06_kebocoran")
    return s.sort_values(ascending=False)


# ---------- 3. Preprocessing ----------
def praproses():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        # min_frequency: artist ada 285 nilai, yang jarang digabung jadi "infrequent".
        ("kat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), KAT),
    ])


def bungkus(model, smote=False):
    """Prapros + (opsional) SMOTE + model.

    SMOTE di dalam pipeline, jadi hanya kena data train tiap fold dan tidak
    bocor ke data validasi.
    """
    langkah = [("prep", praproses())]
    if smote:
        langkah.append(("smote", SMOTE(random_state=42, k_neighbors=5)))
    return ImbPipeline(langkah + [("clf", model)])


# ---------- 4. Baseline ----------
def baseline(X, y):
    lr = LogisticRegression(max_iter=3000, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
    knn = KNeighborsClassifier()
    kandidat = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "K-Nearest Neighbors": knn,
        "Voting (soft)": VotingClassifier(
            [("lr", lr), ("rf", rf), ("knn", knn)], voting="soft"),
        "Stacking": StackingClassifier(
            [("lr", lr), ("rf", rf)],
            final_estimator=LogisticRegression(max_iter=3000), cv=CV3),
    }
    baris = []
    for nama, m in kandidat.items():
        r = cross_validate(bungkus(m), X, y, cv=CV, scoring=SKOR, n_jobs=NJOB)
        baris.append({"Model": nama, **{s: r[f"test_{s}"].mean() for s in SKOR}})
        print(f"  {nama:22} f1={baris[-1]['f1']:.3f}  recall={baris[-1]['recall']:.3f}")
    t = pd.DataFrame(baris).set_index("Model").sort_values("f1", ascending=False)

    plt.figure(figsize=(9, 4.5))
    t[["precision", "recall", "f1"]].plot.bar(ax=plt.gca(), rot=20)
    plt.title("Perbandingan Baseline (5-fold x 3 repeat)")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    simpan("07_baseline")
    return t


# ---------- 5. Hyperparameter tuning ----------
def tuning(X, y):
    grid = {"clf__n_estimators": [300],
            "clf__max_depth": [15, 25, None],
            "clf__min_samples_split": [2, 5],
            "clf__max_features": ["sqrt", 0.5],
            "clf__class_weight": [None, "balanced"]}
    gs = GridSearchCV(bungkus(RandomForestClassifier(random_state=42, n_jobs=1)),
                      grid, cv=CV5, scoring="f1", n_jobs=NJOB, verbose=0)
    gs.fit(X, y)
    print(f"  best f1={gs.best_score_:.3f}")
    print(f"  {gs.best_params_}")
    return gs


# ---------- 6. Resampling ----------
def dengan_smote(X, y, params):
    p = {k.replace("clf__", ""): v for k, v in params.items()}
    m = bungkus(RandomForestClassifier(random_state=42, n_jobs=1, **p), smote=True)
    r = cross_validate(m, X, y, cv=CV, scoring=SKOR, n_jobs=NJOB)
    return m, {s: r[f"test_{s}"].mean() for s in SKOR}


# ---------- 7. Evaluasi ----------
def evaluasi(model, X, y, nama):
    p = cross_val_predict(model, X, y, cv=CV5, method="predict_proba", n_jobs=NJOB)[:, 1]
    pred = (p > .5).astype(int)
    print("\n" + classification_report(y, pred,
                                       target_names=["Biasa", "Ultra Rare"],
                                       digits=3))

    f, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["Biasa", "Ultra Rare"], cmap="Blues",
        ax=a1, colorbar=False)
    a1.set_title("Confusion Matrix")
    fpr, tpr, _ = roc_curve(y, p)
    a2.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y, p):.3f}")
    a2.plot([0, 1], [0, 1], "k--", lw=1)
    a2.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Kurva ROC")
    a2.legend(loc="lower right")
    simpan(f"08_evaluasi_{nama}")

    pr, rc, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(6, 4.5))
    plt.plot(rc, pr, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Kurva Precision-Recall")
    simpan(f"09_pr_curve_{nama}")
    return p


def kepentingan_fitur(model, X, y):
    model.fit(X, y)
    prep = model.named_steps["prep"]
    nama = list(NUM) + list(
        prep.named_transformers_["kat"].get_feature_names_out(KAT))
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=nama)

    plt.figure(figsize=(7, 5.5))
    imp.nlargest(15).sort_values().plot.barh()
    plt.title("15 Fitur Paling Berpengaruh (Random Forest)")
    plt.xlabel("importance")
    simpan("10_feature_importance")
    return imp.sort_values(ascending=False)


def main():
    d = muat()
    X, y = d[NUM + KAT], d[TARGET]
    print(f"Data: {len(d):,} kartu Pokemon | ultra rare {y.sum():,} ({y.mean():.1%})\n")

    print("[1/6] EDA ...")
    eda(d)

    print("[2/6] Cek kebocoran data ...")
    s = bocor(d)
    print("  AUC satu-kolom:", "  ".join(f"{k}={v:.3f}" for k, v in s.items()))

    print("[3/6] Baseline ...")
    tabel = baseline(X, y)

    print("[4/6] Hyperparameter tuning ...")
    gs = tuning(X, y)

    print("[5/6] SMOTE ...")
    model_smote, skor_smote = dengan_smote(X, y, gs.best_params_)
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in skor_smote.items()))

    print("[6/6] Evaluasi akhir ...")
    evaluasi(model_smote, X, y, "smote")
    imp = kepentingan_fitur(model_smote, X, y)

    ringkas = pd.concat([
        tabel[SKOR],
        pd.DataFrame([skor_smote], index=["RF tuned + SMOTE"]),
    ]).round(3)
    joblib.dump({
        "model": model_smote, "fitur_num": NUM, "fitur_kat": KAT,
        "opsi": {k: sorted(d[k].dropna().astype(str).unique()) for k in KAT},
        "median": d[NUM].median().to_dict(),
    }, "model.joblib")
    Path("hasil.md").write_text(
        f"# Hasil - Prediksi Kartu Pokemon TCG Ultra Rare\n\n"
        f"Data: {len(d):,} kartu | ultra rare {y.sum():,} ({y.mean():.1%})\n\n"
        "## Perbandingan Model\n\n" + ringkas.to_markdown()
        + "\n\n## Best Params\n\n```\n" + str(gs.best_params_)
        + "\n```\n\n## Cek Kebocoran (AUC satu kolom)\n\n"
        + s.round(3).to_markdown()
        + f"\n\nKolom dibuang: {BOCOR}\n\n"
        + "## Top 10 Fitur\n\n" + imp.head(10).round(4).to_markdown() + "\n",
        encoding="utf-8")

    print("\n=== RINGKASAN ===")
    print(ringkas.to_string())
    print(f"\nTersimpan: {len(list(FIG.glob('*.png')))} gambar di figures/, "
          "model.joblib, hasil.md")


if __name__ == "__main__":
    main()
