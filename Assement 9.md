# Klasifikasi Zona COVID-19 Menggunakan KNN

## 1. Pemodelan

Untuk membangun model klasifikasi zona COVID-19 menggunakan KNN, diperlukan beberapa tahap pemodelan dan preprocessing data.

### 1.1 Deskripsi Data

Dataset terdiri dari 100 daerah (data training) dengan 14 parameter indikator (C1-C14) dan target klasifikasi zona (KUNING, ORANGE, MERAH). Terdapat 5 daerah baru yang perlu diklasifikasikan.

Struktur parameter:
- **C1-C6, C11-C14**: Parameter kategorikal dengan nilai Y/T
- **C7, C9**: Nilai persentase (misal: 50%)
- **C8, C10**: Nilai desimal dengan format X,Y

### 1.2 Preprocessing Data

Untuk menggunakan algoritma KNN, semua fitur perlu dikonversi ke format numerik:

#### 1.2.1 Konversi Data Kategorikal
- Nilai Y dikonversi menjadi 1
- Nilai T dikonversi menjadi 0

#### 1.2.2 Konversi Persentase
- Nilai persentase (C7, C9) dikonversi ke bentuk desimal
- Contoh: 50% → 0.50, 3% → 0.03

#### 1.2.3 Standardisasi Format Numerik
- Nilai desimal (C8, C10) distandarisasi dengan mengganti koma (,) dengan titik (.)
- Contoh: 0,25 → 0.25

Berikut contoh data training sebelum dan sesudah preprocessing:

**Tabel 1: Contoh Data Training Sebelum Preprocessing (5 baris pertama)**

| No | Daerah | C1 | C2 | C3 | C4 | C5 | C6 | C7    | C8    | C9   | C10    | C11 | C12 | C13 | C14 | Zona   |
|----|--------|----|----|----|----|----|----|-------|-------|------|--------|-----|-----|-----|-----|--------|
| 1  | A001   | Y  | Y  | Y  | Y  | Y  | Y  | 76%   | 0,11  | 4%   | -0,13  | Y   | Y   | Y   | Y   | KUNING |
| 2  | A002   | Y  | T  | Y  | Y  | Y  | T  | 12%   | 0,13  | 1%   | 0,1    | Y   | T   | Y   | Y   | MERAH  |
| 3  | A003   | Y  | Y  | Y  | Y  | Y  | Y  | 23%   | 0     | 2%   | 0,19   | Y   | Y   | Y   | Y   | ORANGE |
| 4  | A004   | Y  | Y  | Y  | Y  | Y  | Y  | 79%   | 0,12  | 2%   | -0,04  | T   | Y   | T   | Y   | KUNING |
| 5  | A005   | Y  | Y  | Y  | T  | Y  | Y  | 90%   | 0,06  | 5%   | -0,11  | Y   | Y   | Y   | T   | KUNING |

**Tabel 2: Contoh Data Training Setelah Preprocessing (5 baris pertama)**

| No | Daerah | C1 | C2 | C3 | C4 | C5 | C6 | C7   | C8   | C9   | C10    | C11 | C12 | C13 | C14 | Zona   |
|----|--------|----|----|----|----|----|----|------|------|------|--------|-----|-----|-----|-----|--------|
| 1  | A001   | 1  | 1  | 1  | 1  | 1  | 1  | 0.76 | 0.11 | 0.04 | -0.13  | 1   | 1   | 1   | 1   | KUNING |
| 2  | A002   | 1  | 0  | 1  | 1  | 1  | 0  | 0.12 | 0.13 | 0.01 | 0.10   | 1   | 0   | 1   | 1   | MERAH  |
| 3  | A003   | 1  | 1  | 1  | 1  | 1  | 1  | 0.23 | 0.00 | 0.02 | 0.19   | 1   | 1   | 1   | 1   | ORANGE |
| 4  | A004   | 1  | 1  | 1  | 1  | 1  | 1  | 0.79 | 0.12 | 0.02 | -0.04  | 0   | 1   | 0   | 1   | KUNING |
| 5  | A005   | 1  | 1  | 1  | 0  | 1  | 1  | 0.90 | 0.06 | 0.05 | -0.11  | 1   | 1   | 1   | 0   | KUNING |

Sama halnya untuk data baru yang akan diklasifikasikan:

**Tabel 3: Data Baru Sebelum Preprocessing**

| No | Daerah | C1 | C2 | C3 | C4 | C5 | C6 | C7   | C8    | C9   | C10    | C11 | C12 | C13 | C14 |
|----|--------|----|----|----|----|----|----|------|-------|------|--------|-----|-----|-----|-----|
| 1  | A101   | Y  | Y  | Y  | T  | Y  | Y  | 50%  | 0,25  | 3%   | -0,1   | Y   | Y   | Y   | Y   |
| 2  | A102   | Y  | T  | Y  | Y  | T  | T  | 10%  | 0,1   | 2%   | 0,1    | Y   | T   | T   | Y   |
| 3  | A103   | Y  | Y  | Y  | Y  | Y  | Y  | 25%  | 0,1   | 4%   | 0,02   | Y   | Y   | Y   | Y   |
| 4  | A104   | Y  | Y  | Y  | T  | Y  | Y  | 30%  | 0,15  | 2%   | -0,03  | T   | Y   | Y   | Y   |
| 5  | A105   | Y  | T  | Y  | T  | Y  | Y  | 90%  | 0,01  | 5%   | -0,15  | Y   | Y   | Y   | T   |

**Tabel 4: Data Baru Setelah Preprocessing**

| No | Daerah | C1 | C2 | C3 | C4 | C5 | C6 | C7   | C8   | C9   | C10    | C11 | C12 | C13 | C14 |
|----|--------|----|----|----|----|----|----|------|------|------|--------|-----|-----|-----|-----|
| 1  | A101   | 1  | 1  | 1  | 0  | 1  | 1  | 0.50 | 0.25 | 0.03 | -0.10  | 1   | 1   | 1   | 1   |
| 2  | A102   | 1  | 0  | 1  | 1  | 0  | 0  | 0.10 | 0.10 | 0.02 | 0.10   | 1   | 0   | 0   | 1   |
| 3  | A103   | 1  | 1  | 1  | 1  | 1  | 1  | 0.25 | 0.10 | 0.04 | 0.02   | 1   | 1   | 1   | 1   |
| 4  | A104   | 1  | 1  | 1  | 0  | 1  | 1  | 0.30 | 0.15 | 0.02 | -0.03  | 0   | 1   | 1   | 1   |
| 5  | A105   | 1  | 0  | 1  | 0  | 1  | 1  | 0.90 | 0.01 | 0.05 | -0.15  | 1   | 1   | 1   | 0   |

### 1.3 Normalisasi dengan Min-Max Scaling

Setelah preprocessing, semua fitur dinormalisasi menggunakan Min-Max Scaling untuk menyamakan skala data dengan rumus:

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

Hasil normalisasi membuat semua fitur berada dalam rentang [0, 1].

**Tabel 5: Data Training Setelah Normalisasi (5 baris pertama)**

| No | Daerah | C1    | C2    | C3    | C4    | C5    | C6    | C7    | C8    | C9    | C10   | C11   | C12   | C13   | C14   | Zona   |
|----|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| 1  | A001   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.760 | 0.550 | 0.400 | 0.179 | 1.000 | 1.000 | 1.000 | 1.000 | KUNING |
| 2  | A002   | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.120 | 0.650 | 0.100 | 0.683 | 1.000 | 0.000 | 1.000 | 1.000 | MERAH  |
| 3  | A003   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.230 | 0.000 | 0.200 | 0.963 | 1.000 | 1.000 | 1.000 | 1.000 | ORANGE |
| 4  | A004   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.790 | 0.600 | 0.200 | 0.359 | 0.000 | 1.000 | 0.000 | 1.000 | KUNING |
| 5  | A005   | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.900 | 0.300 | 0.500 | 0.201 | 1.000 | 1.000 | 1.000 | 0.000 | KUNING |

**Tabel 6: Data Baru Setelah Normalisasi**

| No | Daerah | C1    | C2    | C3    | C4    | C5    | C6    | C7    | C8    | C9    | C10   | C11   | C12   | C13   | C14   |
|----|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1  | A101   | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.300 | 0.214 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2  | A102   | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.100 | 0.500 | 0.200 | 0.683 | 1.000 | 0.000 | 0.000 | 1.000 |
| 3  | A103   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 0.500 | 0.400 | 0.514 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4  | A104   | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.300 | 0.750 | 0.200 | 0.386 | 0.000 | 1.000 | 1.000 | 1.000 |
| 5  | A105   | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.900 | 0.050 | 0.500 | 0.143 | 1.000 | 1.000 | 1.000 | 0.000 |

### 1.4 Algoritma KNN

K-Nearest Neighbors (KNN) digunakan untuk klasifikasi dengan parameter:

1. **Nilai K = 33**: Menggunakan 33 tetangga terdekat (sekitar 1/3 dari total data training)
2. **Metrik Jarak**: Euclidean Distance dengan rumus:
   $$d(p,q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$
3. **Voting**: Kelas mayoritas dari K tetangga terdekat ditetapkan sebagai hasil klasifikasi
4. **Probabilitas**: Dihitung berdasarkan proporsi kelas di antara tetangga terdekat:
   $$P(class_i) = \frac{n_i}{k}$$
   
   Dimana:
   - $P(class_i)$ adalah probabilitas kelas i
   - $n_i$ adalah jumlah tetangga dengan kelas i
   - $k$ adalah jumlah total tetangga (33)

5. **Tie-breaking**: Jika terjadi seri (jumlah kelas sama), kelas dengan jarak rata-rata terkecil dipilih

Distribusi kelas pada data training:
- **KUNING**: 36 data (36.00%)
- **ORANGE**: 34 data (34.00%)
- **MERAH**: 30 data (30.00%)

## 2. Hasil Klasifikasi

| No | Daerah | Proses Klasifikasi |
|----|--------|-------------------|
| 1  | A101   | **KUNING** - Dari 33 tetangga terdekat: 16 KUNING (48.48%), 5 MERAH (15.15%), 12 ORANGE (36.36%). Tetangga terdekat adalah A040 (ORANGE, jarak 0.9437), A089 (KUNING, jarak 0.9610), dan A046 (ORANGE, jarak 1.1825). Klasifikasi ditentukan berdasarkan mayoritas kelas dari 33 tetangga terdekat dengan perhitungan jarak Euclidean. |
| 2  | A102   | **KUNING** - Dari 33 tetangga terdekat: 12 KUNING (36.36%), 9 MERAH (27.27%), 12 ORANGE (36.36%). Terjadi tie antara KUNING dan ORANGE. Tetangga terdekat adalah A002 (MERAH, jarak 1.4258), A032 (ORANGE, jarak 1.6180), dan A034 (KUNING, jarak 1.6357). Tie-breaking dilakukan dengan menghitung rata-rata jarak untuk setiap kelas, dimana KUNING memiliki rata-rata jarak terkecil (2.1058) dibandingkan ORANGE (2.1693) dan MERAH (2.3201). |
| 3  | A103   | **KUNING** - Dari 33 tetangga terdekat: 14 KUNING (42.42%), 5 MERAH (15.15%), 14 ORANGE (42.42%). Terjadi tie antara KUNING dan ORANGE. Tetangga terdekat adalah A051 (ORANGE, jarak 0.2744), A077 (ORANGE, jarak 0.5201), dan A018 (MERAH, jarak 0.5647). Meskipun tetangga terdekat adalah ORANGE, tie-breaking menghasilkan KUNING karena rata-rata jarak ke tetangga KUNING sedikit lebih kecil dibandingkan ke tetangga ORANGE. |
| 4  | A104   | **KUNING** - Dari 33 tetangga terdekat: 14 KUNING (42.42%), 6 MERAH (18.18%), 13 ORANGE (39.39%). Tetangga terdekat adalah A039 (MERAH, jarak 1.1924), A010 (ORANGE, jarak 1.2460), dan A040 (ORANGE, jarak 1.2869). Meskipun tetangga terdekat adalah dari kelas MERAH dan ORANGE, mayoritas dari 33 tetangga adalah KUNING, sehingga diklasifikasikan sebagai KUNING. |
| 5  | A105   | **KUNING** - Dari 33 tetangga terdekat: 14 KUNING (42.42%), 9 MERAH (27.27%), 10 ORANGE (30.30%). Tetangga terdekat adalah A005 (KUNING, jarak 1.0371), A022 (MERAH, jarak 1.1057), dan A023 (ORANGE, jarak 1.3207). Meskipun distribusi kelas cukup merata, kelas KUNING memiliki jumlah terbanyak dan tetangga terdekat juga KUNING. |
