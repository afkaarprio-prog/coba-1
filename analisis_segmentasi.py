"""Segmentasi nasabah: K-Means (k=4) pada clustering_data.csv.

Jalankan: python analisis_segmentasi.py
Butuh: pandas, numpy, scikit-learn
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("clustering_data.csv")
produk = [c for c in df.columns if c.startswith("Produk_")]

# Usia kosong hanya pada status Janda/Duda; diisi median agar ikut diklaster
X = pd.DataFrame({
    "usia": df.Usia.fillna(df.Usia.median()),
    "anak": df.Jumlah_Anak,
    "pendapatan": df.Pendapatan_Tahunan,
})
X[produk] = df[produk]
X["log_saldo"] = np.log10(df.Total_Relationship_Balance.clip(lower=1))
Xs = StandardScaler().fit_transform(X)

# Pilih jumlah klaster lewat silhouette (sampel 8.000 baris)
for k in range(2, 8):
    labels = KMeans(k, n_init=10, random_state=42).fit_predict(Xs)
    skor = silhouette_score(Xs, labels, sample_size=8000, random_state=1)
    print(f"k={k} silhouette={skor:.3f}")

# k=4 dipakai di dashboard: lebih terbaca daripada k=2 meski skornya lebih rendah
df["segmen"] = KMeans(4, n_init=10, random_state=42).fit_predict(Xs)

profil = df.groupby("segmen").agg(
    nasabah=("GCIF", "size"),
    usia=("Usia", "mean"),
    pendapatan=("Pendapatan_Tahunan", "mean"),
    saldo=("Total_Relationship_Balance", "mean"),
    produk=("Total_Kepemilikan_Produk", "mean"),
)
print(profil.round(1))
print(df.groupby("segmen")[produk].mean().round(2))
