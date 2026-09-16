# Kerangka PPT — Prediksi Kartu Pokémon TCG Ultra Rare

Dipetakan dari template contoh (28 slide). Gambar diambil dari `figures/`,
angka dari `hasil.md`.

---

## 1. Judul
Prediksi Kelangkaan Kartu Pokémon TCG dengan Machine Learning
Nama / NIM · Bootcamp Machine Learning and AI for Beginner

## 2. PENDAHULUAN *(slide pembatas)*

## 3. Latar belakang
- Pokémon TCG berjalan sejak 1996 dan masih terbit sampai sekarang; pasar kartu koleksinya bernilai miliaran dolar.
- Kartu *ultra rare* (Secret, Rainbow, Illustration Rare, VMAX, GX, EX) berharga jauh di atas kartu biasa — selisihnya bisa ratusan kali lipat.
- Masalahnya, kelangkaan tidak tertulis seragam di kartu dan istilahnya berubah tiap era, menyulitkan kolektor pemula menilai kartu.
- Proyek ini membangun model yang memperkirakan apakah sebuah kartu tergolong ultra rare dari atribut yang terlihat di kartunya.

> Ganti angka pasar dengan sumber yang bisa kamu kutip agar bisa dipertanggungjawabkan.

## 4. DATASET *(slide pembatas)*

## 5. Dataset
- **Sumber:** Pokémon TCG Data 1999–2023 (Pokémon TCG API), `pokemon-tcg-data-master 1999-2023.csv`
- **Ukuran awal:** 17.172 kartu × 29 kolom
- **Subset dipakai:** 14.497 kartu bersupertype *Pokémon* (kartu Trainer & Energy dibuang karena tidak punya HP/serangan)
- **Target:** `is_ultra_rare` → 0 = biasa, 1 = ultra rare

## 6. Deskripsi fitur (1) — atribut kartu
`hp`, `types`, `attacks`, `weaknesses`, `retreatCost`, `abilities`, `flavorText`

## 7. Deskripsi fitur (2) — penerbitan
`set`, `series`, `artist`, `release_date`, `regulationMark`, `rarity` (sumber target)

## 8. Data preview & tipe data
- 14.497 baris × 29 kolom, campuran numerik dan objek.
- 16 kolom punya missing value.
- Banyak kolom tersimpan sebagai *string* berisi list/dict Python sehingga harus di-parse dulu.

## 9. Distribusi target
**Gambar:** `01_distribusi_target.png`
2.179 ultra rare (15,0%) vs 12.318 biasa (85,0%) → tidak seimbang, jadi akurasi saja tidak cukup dan perlu resampling.

## 10. Distribusi fitur
**Gambar:** `02_distribusi_fitur.png`
Kartu ultra rare cenderung ber-HP dan berdamage jauh lebih tinggi. Distribusinya miring ke kanan, wajar karena kartu kuat memang sedikit.

## 11. Matriks korelasi
**Gambar:** `03_korelasi.png`
`hp`, `max_damage`, dan `total_energy` saling berkorelasi (kartu kuat mahal di semua sisi), tapi belum sampai taraf multikolinearitas yang mengganggu model pohon.

## 12. Tren waktu *(slide tambahan, tidak ada di contoh)*
**Gambar:** `04_tren_tahun.png`
Proporsi kartu ultra rare naik dari era awal ke era modern — penerbit makin banyak mengeluarkan varian langka.

## 13. Pola per series & artist
**Gambar:** `05_series_artist.png`
Kelangkaan sangat bergantung series dan artist tertentu, jadi keduanya dipakai sebagai fitur.

## 14. DATA PREPROCESSING *(slide pembatas)*

## 15. Feature engineering
Kolom list/dict di-parse dengan `ast.literal_eval` menjadi fitur numerik:

| Fitur baru | Asal |
|---|---|
| `n_attacks`, `max_damage`, `total_energy`, `panjang_teks` | `attacks` |
| `n_types`, `type_1` | `types` |
| `weak_type` | `weaknesses` |
| `punya_ability`, `punya_flavor` | `abilities`, `flavorText` |
| `tahun` | `release_date` |

## 16. Handling missing value
- `abilities` (76,5% kosong) dan `flavorText` (33%) → kosong berarti *kartu tidak punya*, bukan data hilang. Diubah jadi penanda 0/1, bukan dihapus.
- `convertedRetreatCost` (5,6%) → diisi median.
- `regulationMark` (71%) → diisi `"none"`, karena penanda ini memang baru ada di era tertentu.

## 17. Encoding & scaling
- `OneHotEncoder(min_frequency=20)` untuk kategorikal — `artist` punya 285 nilai, yang jarang digabung jadi satu kelompok agar tidak meledak.
- `StandardScaler` untuk numerik.
- Semua dibungkus `ColumnTransformer` di dalam `Pipeline` agar tidak bocor antar fold.

## 18. Deteksi kebocoran data ⭐ *(slide tambahan — nilai plus)*
**Gambar:** `06_kebocoran.png`
- `subtypes` sendirian mencapai ROC-AUC 0,901 dan `rules` 0,886.
- Penyebabnya nama rarity memuat mekanik kartu: "Rare Holo **VMAX**" ditebak dari kolom berisi `['VMAX']`. Itu mencocokkan label, bukan memprediksi.
- Kedua kolom dibuang. Ini menurunkan skor mentah tapi membuat hasilnya sah.

## 19. MODELING *(slide pembatas)*

## 20. Mencari baseline
Lima kandidat, dievaluasi 5-fold × 3 repeat:
1. Logistic Regression
2. Random Forest
3. K-Nearest Neighbors
4. Voting (soft) — LR + RF + KNN
5. Stacking — LR + RF, meta-model Logistic Regression

**Gambar:** `07_baseline.png` · angka di `hasil.md`

## 21. Temuan: CV internal wajib di-shuffle ⭐ *(slide tambahan — nilai plus)*
- Stacking awalnya menghasilkan f1 **0,015** — praktis menebak semua kartu sebagai biasa.
- Penyebabnya `StackingClassifier(cv=3)` memakai StratifiedKFold **tanpa shuffle**, sedangkan data urut kronologis 1999→2023. Meta-learner jadi dilatih lintas era dan koefisien Random Forest-nya berbalik negatif.
- Setelah `StratifiedKFold(3, shuffle=True)`: f1 naik ke **0,859**.

## 22. HYPERPARAMETER TUNING *(slide pembatas)*

## 23. GridSearchCV
- Random Forest, 24 kombinasi, 5-fold, scoring `f1`.
- Grid: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`, `class_weight`.
- Parameter terbaik dan skornya ada di `hasil.md`.

## 24. RESAMPLING *(slide pembatas)*

## 25. SMOTE
- Kelas minoritas hanya 15%, sehingga recall kelas ultra rare tertinggal.
- SMOTE diletakkan **di dalam** pipeline (`imblearn.pipeline`), jadi hanya menyentuh data latih tiap fold dan tidak bocor ke data validasi.

## 26. EVALUASI *(slide pembatas)*

## 27. Hasil akhir
**Gambar:** `08_evaluasi_smote.png` (confusion matrix + ROC), `09_pr_curve_smote.png`
Sebutkan accuracy, precision, recall, f1, dan AUC dari `hasil.md`.

**Argumen metrik:** untuk kolektor, *recall* lebih penting — kartu ultra rare yang terlewat berarti kehilangan nilai. Tapi precision tetap dijaga agar kartu biasa tidak salah dihargai tinggi.

## 28. Feature importance
**Gambar:** `10_feature_importance.png`
Jelaskan 3–5 fitur teratas dan kenapa masuk akal secara domain.

## 29. APLIKASI *(slide pembatas)*

## 30. Demo aplikasi
- Streamlit: form atribut kartu → probabilitas ultra rare.
- Tiga tab: Prediksi, Eksplorasi Data, Tentang.
- Sisipkan tangkapan layar + link GitHub.

## 31. KESIMPULAN & SARAN *(slide pembatas)*

## 32. Kesimpulan
- Random Forest hasil tuning + SMOTE memberi keseimbangan precision–recall terbaik.
- Atribut kartu yang terlihat mata (HP, damage, biaya energi) memang memprediksi kelangkaan.
- Dua kolom dibuang karena terbukti membocorkan label — hasilnya lebih rendah tapi jujur.

## 33. Saran & keterbatasan
- Kelangkaan sebagian ditentukan **kebijakan penerbit per set**, bukan murni atribut kartu, sehingga ada plafon akurasi yang tidak bisa dilewati model sebesar apa pun.
- Data hanya sampai 2023; set baru perlu latih ulang.
- Pengembangan lanjutan: pakai gambar kartu (CNN) atau teks efek serangan (NLP), dan uji dengan **split berdasarkan waktu** untuk meniru kondisi nyata memprediksi set yang belum terbit.
