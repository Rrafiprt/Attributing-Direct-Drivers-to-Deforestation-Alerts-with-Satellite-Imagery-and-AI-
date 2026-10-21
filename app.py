import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="ForestGuard AI", page_icon="🌳", layout="wide")

st.title("🌳 ForestGuard AI")
st.caption("Dashboard demonstrasi pemantauan indikasi deforestasi dan klasifikasi kemungkinan penyebab")

DRIVERS = [
    "Perkebunan skala besar",
    "Pertanian berpindah / perambahan",
    "Infrastruktur / permukiman",
    "Pertambangan",
    "Kebakaran hutan / lahan",
]

@st.cache_data
def create_synthetic_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    driver = rng.choice(DRIVERS, n, p=[0.25, 0.24, 0.20, 0.13, 0.18])
    rows = []
    for i, d in enumerate(driver, start=1):
        # These are simulated features and not measurements from satellite imagery.
        ndvi_before = rng.uniform(0.55, 0.90)
        if d == "Kebakaran hutan / lahan":
            ndvi_after = rng.uniform(0.05, 0.30)
            burn_index = rng.uniform(0.60, 1.00)
            road_distance = rng.uniform(0.5, 15)
            area_ha = rng.uniform(2, 120)
        elif d == "Infrastruktur / permukiman":
            ndvi_after = rng.uniform(0.10, 0.38)
            burn_index = rng.uniform(0.00, 0.35)
            road_distance = rng.uniform(0.0, 2.5)
            area_ha = rng.uniform(1, 35)
        elif d == "Pertambangan":
            ndvi_after = rng.uniform(0.02, 0.25)
            burn_index = rng.uniform(0.00, 0.40)
            road_distance = rng.uniform(0.5, 12)
            area_ha = rng.uniform(5, 180)
        elif d == "Perkebunan skala besar":
            ndvi_after = rng.uniform(0.18, 0.42)
            burn_index = rng.uniform(0.00, 0.35)
            road_distance = rng.uniform(0.5, 8)
            area_ha = rng.uniform(30, 300)
        else:
            ndvi_after = rng.uniform(0.12, 0.45)
            burn_index = rng.uniform(0.00, 0.45)
            road_distance = rng.uniform(2, 20)
            area_ha = rng.uniform(1, 25)
        rows.append({
            "ID Alert": f"FG-{i:04d}",
            "Tanggal Alert": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 290))),
            "NDVI Sebelum": round(ndvi_before, 3),
            "NDVI Sesudah": round(ndvi_after, 3),
            "Perubahan NDVI": round(ndvi_before - ndvi_after, 3),
            "Jarak Jalan (km)": round(road_distance, 2),
            "Luas Indikasi (ha)": round(area_ha, 2),
            "Indeks Kebakaran": round(burn_index, 3),
            "Penyebab Referensi": d,
            "Latitude Simulasi": round(float(rng.uniform(-4.8, 4.8)), 5),
            "Longitude Simulasi": round(float(rng.uniform(102.0, 119.0)), 5),
        })
    return pd.DataFrame(rows)

df = create_synthetic_data()
feature_cols = ["NDVI Sebelum", "NDVI Sesudah", "Perubahan NDVI",
                "Jarak Jalan (km)", "Luas Indikasi (ha)", "Indeks Kebakaran"]
X = df[feature_cols]
y = df["Penyebab Referensi"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.warning("Mode demonstrasi: seluruh data, koordinat, dan label dibuat secara sintetis. Jangan gunakan hasil ini sebagai bukti deforestasi nyata atau dasar penegakan hukum.")
st.sidebar.header("Filter Data")
drivers_selected = st.sidebar.multiselect("Kategori penyebab", DRIVERS, default=DRIVERS)
date_range = st.sidebar.date_input(
    "Rentang tanggal",
    value=(df["Tanggal Alert"].min().date(), df["Tanggal Alert"].max().date())
)
filtered = df[df["Penyebab Referensi"].isin(drivers_selected)].copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[filtered["Tanggal Alert"].dt.date.between(date_range[0], date_range[1])]

# Apply trained model to selected records
if not filtered.empty:
    filtered["Prediksi Model"] = model.predict(filtered[feature_cols])
    filtered["Keyakinan Model"] = model.predict_proba(filtered[feature_cols]).max(axis=1)
else:
    filtered["Prediksi Model"] = pd.Series(dtype=str)
    filtered["Keyakinan Model"] = pd.Series(dtype=float)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Alert", f"{len(filtered):,}")
c2.metric("Total Luas Indikasi", f"{filtered['Luas Indikasi (ha)'].sum():,.1f} ha" if len(filtered) else "0 ha")
c3.metric("Perubahan NDVI Rata-rata", f"{filtered['Perubahan NDVI'].mean():.3f}" if len(filtered) else "—")
c4.metric("Akurasi uji model demo", f"{accuracy:.1%}")

st.divider()
if filtered.empty:
    st.info("Tidak ada data untuk filter yang dipilih.")
else:
    left, right = st.columns(2)
    with left:
        st.subheader("Alert menurut kategori")
        counts = filtered["Penyebab Referensi"].value_counts().rename_axis("Kategori").reset_index(name="Jumlah")
        st.plotly_chart(px.bar(counts, x="Kategori", y="Jumlah", color="Kategori"), use_container_width=True)
    with right:
        st.subheader("Distribusi luas indikasi")
        st.plotly_chart(px.histogram(filtered, x="Luas Indikasi (ha)", nbins=20,
                                     labels={"Luas Indikasi (ha)": "Luas indikasi (ha)"}),
                        use_container_width=True)

    st.subheader("Pola perubahan vegetasi")
    st.plotly_chart(px.scatter(
        filtered, x="NDVI Sebelum", y="NDVI Sesudah",
        color="Prediksi Model", size="Luas Indikasi (ha)",
        hover_data=["ID Alert", "Penyebab Referensi", "Tanggal Alert"],
        labels={"NDVI Sebelum": "NDVI sebelum", "NDVI Sesudah": "NDVI sesudah"}
    ), use_container_width=True)

    st.subheader("Lokasi simulasi alert")
    st.caption("Koordinat di bawah ini dibuat secara acak untuk demonstrasi, bukan lokasi alert satelit nyata.")
    st.plotly_chart(px.scatter(
        filtered, x="Longitude Simulasi", y="Latitude Simulasi",
        color="Prediksi Model", hover_name="ID Alert",
        labels={"Longitude Simulasi": "Longitude simulasi", "Latitude Simulasi": "Latitude simulasi"}
    ), use_container_width=True)

    st.subheader("Tabel alert dan hasil klasifikasi")
    show_cols = ["ID Alert", "Tanggal Alert", "Penyebab Referensi", "Prediksi Model",
                 "Keyakinan Model", "Perubahan NDVI", "Jarak Jalan (km)",
                 "Luas Indikasi (ha)", "Indeks Kebakaran"]
    display_df = filtered[show_cols].copy()
    display_df["Keyakinan Model"] = (display_df["Keyakinan Model"] * 100).round(1).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Unduh hasil filter (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="forestguard_alerts_demo.csv",
        mime="text/csv"
    )

with st.expander("Detail evaluasi model demonstrasi"):
    st.write("Model: Random Forest dengan 150 pohon. Fitur dan label adalah data sintetis.")
    st.write(f"Akurasi pada test split sintetis: **{accuracy:.1%}**")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    importance = pd.DataFrame({"Fitur": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    st.plotly_chart(px.bar(importance, x="Importance", y="Fitur", orientation="h",
                           title="Feature importance pada model sintetis"), use_container_width=True)

st.subheader("Interpretasi dan batasan")
st.markdown("""
- **NDVI** pada contoh ini dipakai sebagai indikator simulasi perubahan vegetasi.
- **Atribusi penyebab** adalah prediksi model dari fitur buatan, bukan konfirmasi lapangan.
- Pada penggunaan nyata, ganti data sintetis dengan citra satelit (misalnya Sentinel-2/Landsat), preprocessing yang sesuai, label tervalidasi, dan evaluasi spasial-temporal.
- Alert perlu ditinjau analis atau petugas lapangan sebelum diambil tindakan.
""")
st.caption("ForestGuard AI • Prototipe akademik • Data sintetis")
