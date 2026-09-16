# Prediksi Kelangkaan Kartu Pokémon TCG dengan Machine Learning

Klasifikasi biner untuk memperkirakan apakah sebuah kartu Pokémon TCG tergolong
**ultra rare** berdasarkan atribut yang terlihat di kartunya.

Dataset: **Pokémon TCG Data 1999–2023** — 17.172 kartu, disaring menjadi 14.497
kartu bersupertype *Pokémon*.
Target: `is_ultra_rare` — 2.179 kartu (15,0%).

---

## Hasil

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0,957 | 0,888 | 0,815 | 0,850 | 0,969 |
| Stacking (LR + RF) | 0,955 | 0,890 | 0,799 | 0,842 | 0,977 |
| Voting (soft) | 0,951 | 0,890 | 0,769 | 0,825 | 0,972 |
| Random Forest | 0,948 | 0,877 | 0,765 | 0,817 | 0,970 |
| K-Nearest Neighbors | 0,931 | 0,818 | 0,693 | 0,750 | 0,921 |
| **RF tuned + SMOTE** | **0,958** | **0,870** | **0,848** | **0,859** | **0,980** |

Dievaluasi dengan `RepeatedStratifiedKFold` 5-fold × 3 repeat.

Model akhir: Random Forest hasil GridSearchCV + SMOTE.
Recall kelas ultra rare naik dari **0,765 → 0,848** dibanding Random Forest
baseline, dengan precision hanya turun tipis.

**Fitur paling berpengaruh:** `hp` (0,342), `punya_flavor` (0,128),
`set_Shiny Vault` (0,095), `tahun` (0,048), `max_damage` (0,043).

---

## Dua temuan metodologi

### 1. Kebocoran data pada `subtypes` dan `rules`

Kolom `subtypes` **sendirian** mencapai ROC-AUC **0,902**, dan `rules` 0,887.

Penyebabnya nama rarity memuat mekanik kartunya sendiri — "Rare Holo **VMAX**"
bisa ditebak dari kolom yang isinya `['VMAX']`. Itu mencocokkan label, bukan
memprediksi. Kedua kolom dibuang meski skor mentahnya jadi turun.

### 2. CV internal Stacking wajib di-shuffle

`StackingClassifier(cv=3)` awalnya menghasilkan **f1 = 0,015** — praktis
menebak semua kartu sebagai biasa.

Sebabnya `cv=3` memakai `StratifiedKFold` tanpa shuffle, sedangkan CSV-nya urut
kronologis 1999→2023. Meta-learner jadi dilatih lintas era, dan koefisien
Random Forest-nya berbalik **negatif** (`[3,31, -2,86]`).

Setelah diganti `StratifiedKFold(3, shuffle=True)`: **f1 = 0,842**, koefisien
kembali positif (`[5,18, 3,10]`).

---

## Cara menjalankan

```bash
pip install -r requirements.txt

python pipeline.py      # latih model + hasilkan semua grafik (~45 menit)
streamlit run app.py    # jalankan aplikasi
```

`pipeline.py` menghasilkan `figures/` (10 grafik), `model.joblib`, dan `hasil.md`.
`app.py` memakai `model.joblib` yang sudah jadi, tidak melatih ulang.

---

## Isi repo

| Berkas | Keterangan |
|---|---|
| `pipeline.py` | Pipeline lengkap: EDA → cek kebocoran → preprocessing → 5 baseline → tuning → SMOTE → evaluasi |
| `app.py` | Aplikasi Streamlit, 3 tab: Prediksi, Eksplorasi Data, Tentang |
| `cek_dataset.py` | Validator kelayakan dataset untuk tugas klasifikasi |
| `KERANGKA_PPT.md` | Kerangka presentasi 33 slide, dipetakan ke gambar dan angka |
| `hasil.md` | Tabel hasil yang dihasilkan otomatis |
| `figures/` | 10 grafik siap tempel ke slide |

---

## Preprocessing

Kolom `attacks`, `types`, `weaknesses` tersimpan sebagai *string* berisi
list/dict Python, di-parse dengan `ast.literal_eval` menjadi fitur numerik:

| Fitur | Asal |
|---|---|
| `n_attacks`, `max_damage`, `total_energy`, `panjang_teks` | `attacks` |
| `n_types`, `type_1` | `types` |
| `weak_type` | `weaknesses` |
| `punya_ability`, `punya_flavor` | `abilities`, `flavorText` |
| `tahun` | `release_date` |

Missing value ditangani sesuai artinya, bukan seragam: `abilities` (76,5%
kosong) dan `flavorText` (33%) berarti *kartu tidak punya*, jadi diubah jadi
penanda 0/1; `convertedRetreatCost` diisi median; `regulationMark` diisi
`"none"` karena penanda itu memang baru ada di era tertentu.

`OneHotEncoder(min_frequency=20)` dipakai agar `artist` yang punya 285 nilai
tidak meledak jadi ratusan kolom. SMOTE diletakkan **di dalam** pipeline
`imblearn` sehingga hanya menyentuh data latih tiap fold.

---

## Keterbatasan

Kelangkaan sebagian ditentukan **kebijakan penerbit per set**, bukan murni
atribut kartu. Ada plafon akurasi yang tidak bisa dilewati model sebesar apa
pun, dan sebagian sisa error memang berasal dari sana.

Data berhenti di 2023, sehingga set yang lebih baru perlu latih ulang. Evaluasi
memakai split acak; untuk meniru kondisi nyata memprediksi set yang belum
terbit, seharusnya diuji juga dengan **split berdasarkan waktu**.
