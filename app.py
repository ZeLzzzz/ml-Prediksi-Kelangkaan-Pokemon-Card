"""Aplikasi prediksi kartu Pokemon TCG Ultra Rare.

Jalankan: streamlit run app.py
Butuh   : model.joblib (hasil dari python pipeline.py)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Kartu Pokemon Ultra Rare",
                   page_icon="🃏", layout="wide")


@st.cache_resource
def muat_model(path="model.joblib"):
    if not Path(path).exists():
        return None
    return joblib.load(path)


bundel = muat_model()
if bundel is None:
    st.error("model.joblib belum ada. Jalankan dulu: `python pipeline.py`")
    st.stop()

model = bundel["model"]
NUM, KAT, OPSI = bundel["fitur_num"], bundel["fitur_kat"], bundel["opsi"]
MEDIAN = bundel["median"]

st.title("🃏 Prediksi Kartu Pokemon TCG Ultra Rare")
st.caption("Random Forest + SMOTE | 14.497 kartu Pokemon, rilis 1999-2023")

tab_prediksi, tab_eda, tab_info = st.tabs(["Prediksi", "Eksplorasi Data", "Tentang"])

with tab_prediksi:
    st.subheader("Masukkan atribut kartu")
    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown("**Statistik kartu**")
        hp = st.slider("HP", 30, 340, 120, step=10)
        retreat = st.slider("Retreat cost", 0, 5, 2)
        n_attacks = st.slider("Jumlah serangan", 0, 4, 2)
        max_damage = st.slider("Damage tertinggi", 0, 300, 100, step=10)
        total_energy = st.slider("Total biaya energi", 0, 12, 4)

    with k2:
        st.markdown("**Tipe & teks**")
        type_1 = st.selectbox("Tipe utama", OPSI["type_1"])
        weak_type = st.selectbox("Tipe kelemahan", OPSI["weak_type"])
        n_types = st.radio("Jumlah tipe", [1, 2], horizontal=True)
        panjang_teks = st.slider("Panjang teks serangan (karakter)", 0, 400, 80,
                                 step=10)
        punya_ability = st.checkbox("Punya Ability", value=False)
        punya_flavor = st.checkbox("Punya flavor text", value=True)

    with k3:
        st.markdown("**Penerbitan**")
        tahun = st.slider("Tahun rilis", 1999, 2023, 2020)
        series = st.selectbox("Series", OPSI["series"])
        kartu_set = st.selectbox("Set", OPSI["set"])
        artist = st.selectbox("Artist", OPSI["artist"])
        regmark = st.selectbox("Regulation mark", OPSI["regulationMark"])

    baris = pd.DataFrame([{
        "hp": hp, "convertedRetreatCost": retreat, "n_attacks": n_attacks,
        "max_damage": max_damage, "total_energy": total_energy,
        "n_types": n_types, "panjang_teks": panjang_teks,
        "punya_ability": int(punya_ability), "punya_flavor": int(punya_flavor),
        "tahun": tahun, "type_1": type_1, "weak_type": weak_type,
        "artist": artist, "set": kartu_set, "series": series,
        "regulationMark": regmark,
    }])[NUM + KAT]

    if st.button("Prediksi", type="primary", use_container_width=True):
        prob = float(model.predict_proba(baris)[0, 1])
        h1, h2 = st.columns([1, 2])
        with h1:
            if prob > .5:
                st.success("### ✨ ULTRA RARE")
            else:
                st.info("### Kartu biasa")
            st.metric("Probabilitas ultra rare", f"{prob:.1%}")
        with h2:
            st.progress(prob)
            st.caption(
                "Ambang 50%. Model dilatih dengan SMOTE agar kelas minoritas "
                "(15% dari data) tidak terabaikan."
            )
        with st.expander("Data yang dikirim ke model"):
            st.dataframe(baris.T.rename(columns={0: "nilai"}),
                         use_container_width=True)

with tab_eda:
    st.subheader("Grafik hasil analisis")
    fig = sorted(Path("figures").glob("*.png")) if Path("figures").exists() else []
    if not fig:
        st.warning("Folder figures/ kosong. Jalankan `python pipeline.py`.")
    for f in fig:
        st.image(str(f), caption=f.stem.replace("_", " "),
                 use_container_width=True)

with tab_info:
    st.subheader("Tentang proyek")
    st.markdown("""
**Masalah.** Kartu Pokemon TCG punya tingkat kelangkaan bertingkat. Kartu
*ultra rare* (Secret, Rainbow, Illustration Rare, VMAX, GX, EX) bernilai jauh
lebih tinggi di pasar kolektor dibanding kartu biasa.

**Target.** `is_ultra_rare` — 2.179 dari 14.497 kartu (15,0%).

**Kebocoran data yang ditangani.** Kolom `subtypes` dan `rules` dibuang karena
isinya mekanik kartu (V, VMAX, GX, EX) yang juga tertulis di nama rarity-nya.
Kolom `subtypes` sendirian sudah mencapai ROC-AUC 0,901 — itu mencocokkan
label, bukan memprediksi. Lihat grafik `06_kebocoran`.

**Keterbatasan.** Kelangkaan sebagian ditentukan kebijakan penerbit per set,
bukan murni atribut kartu. Jadi ada plafon akurasi yang wajar dan tidak semua
sisa error bisa dihilangkan dengan model yang lebih besar.
    """)
    if Path("hasil.md").exists():
        with st.expander("Hasil evaluasi lengkap"):
            st.markdown(Path("hasil.md").read_text(encoding="utf-8"))
