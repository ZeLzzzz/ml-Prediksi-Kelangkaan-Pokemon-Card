"""Cek apakah sebuah CSV cocok untuk tugas klasifikasi (template PPT bootcamp).

Pakai:  python cek_dataset.py data.csv [nama_kolom_target]
Tanpa nama target, skrip menebak kolom biner yang paling mirip target.
Tanpa argumen sama sekali, menjalankan self-check pada dataset TCG.
"""
import sys
from pathlib import Path

import pandas as pd

MIN_ROWS, MIN_MINORITY = 5000, 500
RASIO_OK = (0.10, 0.40)
POLA_ULTRA = (r"Ultra|Secret|Rainbow|Illustration|Shiny|VMAX|VSTAR|GX|EX\b"
              r"|Amazing|Radiant|Double Rare|Hyper|Prime|LEGEND|Star")


def kandidat_target(d):
    """Kolom dengan tepat 2 nilai unik = kandidat target biner."""
    return [c for c in d.columns if d[c].nunique(dropna=True) == 2]


def periksa(path, target=None):
    d = pd.read_csv(path, low_memory=False)
    n, k = d.shape

    kandidat = kandidat_target(d)
    if target is None:
        if not kandidat:
            print(f"[GAGAL] Tidak ada kolom biner sama sekali di {path}.")
            print("        Dataset ini bukan untuk klasifikasi biner.")
            return False
        # ponytail: ambil kandidat terakhir, target biasanya kolom paling kanan.
        target = kandidat[-1]
        print(f"(target ditebak: '{target}' — kandidat lain: "
              f"{kandidat[:-1] or 'tidak ada'})\n")
    elif target not in d.columns:
        print(f"[GAGAL] Kolom '{target}' tidak ada. Pilihan: {list(d.columns)}")
        return False

    vc = d[target].value_counts()
    minoritas = int(vc.min())
    rasio = minoritas / n

    num = d.select_dtypes("number").columns.drop(target, errors="ignore")
    kat = d.select_dtypes(exclude="number").columns
    miss = d.isna().sum()
    miss = miss[miss > 0]

    hasil = [
        (n >= MIN_ROWS, f"Jumlah baris {n:,} (min {MIN_ROWS:,})"),
        (minoritas >= MIN_MINORITY,
         f"Kelas minoritas {minoritas:,} baris (min {MIN_MINORITY:,})"),
        (RASIO_OK[0] <= rasio <= RASIO_OK[1],
         f"Rasio minoritas {rasio:.1%} (ideal {RASIO_OK[0]:.0%}-{RASIO_OK[1]:.0%})"),
        (len(num) >= 3, f"Fitur numerik {len(num)} (min 3)"),
        (len(kat) >= 3, f"Fitur kategorikal {len(kat)} (min 3)"),
        (len(miss) > 0, f"Kolom dengan missing value: {len(miss)}"),
    ]

    print(f"=== {path} — {n:,} baris x {k} kolom, target '{target}' ===\n")
    for ok, teks in hasil:
        print(f"  {'OK  ' if ok else 'KURANG'}  {teks}")

    # Fitur numerik sengaja tidak jadi syarat wajib: kolom list/dict baru
    # berubah jadi numerik setelah feature engineering di pipeline.py.
    wajib = all(ok for ok, _ in hasil[:3]) and hasil[4][0]
    print(f"\n>>> {'LOLOS' if wajib else 'TIDAK LOLOS'} — "
          f"{sum(ok for ok, _ in hasil)}/{len(hasil)} kriteria terpenuhi")
    if not hasil[5][0]:
        print("    (tanpa missing value, slide 'Handling Missing Value' kosong)")
    if not hasil[3][0]:
        print("    (fitur numerik sedikit — perlu feature engineering dulu)")
    print(f"\nDistribusi target: {vc.to_dict()}")
    return wajib


def demo():
    """Self-check: dataset TCG harus LOLOS (target dibuat sementara)."""
    d = pd.read_csv("pokemon-tcg-data-master 1999-2023.csv", low_memory=False)
    p = d[d.supertype == "Pokémon"].copy()
    p["is_ultra_rare"] = p.rarity.fillna("").str.contains(
        POLA_ULTRA, case=False, regex=True).astype(int)
    tmp = Path("_cek_sementara.csv")
    p.to_csv(tmp, index=False)
    try:
        assert periksa(str(tmp), "is_ultra_rare") is True
    finally:
        tmp.unlink(missing_ok=True)
    print("\n[demo ok] dataset TCG lolos kriteria")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        demo()
    else:
        periksa(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
