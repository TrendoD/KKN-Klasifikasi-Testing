# Klasifikasi Zona COVID Menggunakan Algoritma KNN

## Pendahuluan

Proyek ini bertujuan untuk mengklasifikasikan daerah ke dalam zona COVID (MERAH, ORANGE, atau KUNING) berdasarkan 14 parameter (C1-C14) menggunakan algoritma K-Nearest Neighbors (KNN). Klasifikasi dilakukan dengan menggunakan dataset training berisi 100 daerah dengan zona yang telah diketahui untuk memprediksi zona dari 5 daerah baru.
Proyek ini hanya berupa Tugas Kampus Asesmen 09_K - Klasifikasi.docx

## Dataset

Proyek ini menggunakan dua dataset:
1. **Dataset Training (dataset Covid.csv)**: Berisi 100 daerah dengan 14 parameter (C1-C14) dan label zona (KUNING, ORANGE, MERAH).
2. **Dataset Baru (dataset baru.csv)**: Berisi 5 daerah baru dengan 14 parameter yang perlu diklasifikasikan.

### Struktur Data

Setiap daerah memiliki 14 parameter:
- **C1-C6, C11-C14**: Atribut kategorikal (Y/T)
- **C7, C9**: Nilai persentase (contoh: 50%)
- **C8, C10**: Nilai desimal

### Dataset Training (Sebelum Preprocessing)

Berikut adalah dataset training yang digunakan dalam proyek ini:

| No | Daerah | C1 | C2 | C3 | C4 | C5 | C6 | C7    | C8    | C9   | C10    | C11 | C12 | C13 | C14 | Zona   |
|----|--------|----|----|----|----|----|----|-------|-------|------|--------|-----|-----|-----|-----|--------|
| 1  | A001   | Y  | Y  | Y  | Y  | Y  | Y  | 76%   | 0,11  | 4%   | -0,13  | Y   | Y   | Y   | Y   | KUNING |
| 2  | A002   | Y  | T  | Y  | Y  | Y  | T  | 12%   | 0,13  | 1%   | 0,1    | Y   | T   | Y   | Y   | MERAH  |
| 3  | A003   | Y  | Y  | Y  | Y  | Y  | Y  | 23%   | 0     | 2%   | 0,19   | Y   | Y   | Y   | Y   | ORANGE |
| 4  | A004   | Y  | Y  | Y  | Y  | Y  | Y  | 79%   | 0,12  | 2%   | -0,04  | T   | Y   | T   | Y   | KUNING |
| 5  | A005   | Y  | Y  | Y  | T  | Y  | Y  | 90%   | 0,06  | 5%   | -0,11  | Y   | Y   | Y   | T   | KUNING |
| 6  | A006   | Y  | T  | T  | Y  | Y  | Y  | 10%   | 0     | 8%   | -0,13  | Y   | Y   | Y   | Y   | ORANGE |
| 7  | A007   | T  | T  | Y  | Y  | Y  | T  | 64%   | 0,03  | 5%   | 0,11   | Y   | Y   | Y   | Y   | ORANGE |
| 8  | A008   | Y  | T  | Y  | Y  | Y  | Y  | 47%   | 0,13  | 1%   | 0,03   | Y   | Y   | Y   | Y   | ORANGE |
| 9  | A009   | T  | Y  | Y  | Y  | Y  | Y  | 91%   | 0,13  | 5%   | 0,2    | Y   | T   | T   | Y   | KUNING |
| 10 | A010   | Y  | Y  | Y  | Y  | Y  | Y  | 30%   | 0,18  | 9%   | -0,1   | T   | Y   | Y   | Y   | ORANGE |
| 11 | A011   | T  | Y  | T  | Y  | T  | Y  | 22%   | 0,17  | 4%   | 0,18   | Y   | Y   | Y   | Y   | MERAH  |
| 12 | A012   | T  | Y  | Y  | Y  | Y  | T  | 41%   | 0     | 8%   | 0      | Y   | Y   | Y   | Y   | ORANGE |
| 13 | A013   | T  | T  | T  | Y  | Y  | Y  | 93%   | 0,12  | 1%   | 0,15   | Y   | Y   | Y   | Y   | KUNING |
| 14 | A014   | Y  | T  | Y  | Y  | Y  | Y  | 33%   | 0,04  | 1%   | 0,03   | Y   | Y   | Y   | T   | ORANGE |
| 15 | A015   | Y  | T  | Y  | Y  | Y  | T  | 50%   | 0,18  | 6%   | 0,11   | T   | Y   | T   | Y   | MERAH  |
| 16 | A016   | Y  | Y  | Y  | Y  | Y  | T  | 29%   | 0,2   | 9%   | 0,12   | Y   | T   | Y   | Y   | MERAH  |
| 17 | A017   | T  | Y  | Y  | Y  | Y  | Y  | 51%   | 0,19  | 5%   | 0,12   | Y   | Y   | Y   | Y   | ORANGE |
| 18 | A018   | Y  | Y  | Y  | Y  | Y  | Y  | 1%    | 0,17  | 2%   | 0,13   | Y   | Y   | Y   | Y   | MERAH  |
| 19 | A019   | Y  | Y  | T  | Y  | T  | Y  | 72%   | 0,13  | 3%   | -0,14  | T   | Y   | Y   | Y   | KUNING |
| 20 | A020   | T  | T  | Y  | Y  | Y  | T  | 71%   | 0,15  | 10%  | 0,12   | Y   | Y   | Y   | Y   | ORANGE |
| 21 | A021   | Y  | T  | T  | Y  | Y  | Y  | 41%   | 0,01  | 3%   | 0,04   | Y   | Y   | T   | Y   | ORANGE |
| 22 | A022   | Y  | T  | Y  | T  | Y  | Y  | 0%    | 0     | 0%   | -0,01  | Y   | Y   | Y   | T   | MERAH  |
| 23 | A023   | Y  | T  | Y  | T  | Y  | Y  | 22%   | 0     | 10%  | -0,09  | Y   | Y   | Y   | Y   | ORANGE |
| 24 | A024   | T  | Y  | Y  | Y  | Y  | Y  | 27%   | 0,08  | 4%   | 0,17   | Y   | T   | Y   | Y   | MERAH  |
| 25 | A025   | T  | Y  | Y  | Y  | T  | Y  | 43%   | 0,18  | 10%  | 0,05   | T   | Y   | Y   | Y   | ORANGE |
| 26 | A026   | T  | Y  | Y  | Y  | T  | Y  | 39%   | 0,02  | 3%   | 0,1    | T   | Y   | T   | Y   | MERAH  |
| 27 | A027   | T  | Y  | Y  | Y  | Y  | T  | 88%   | 0,08  | 3%   | 0,03   | Y   | Y   | Y   | Y   | KUNING |
| 28 | A028   | Y  | T  | T  | Y  | T  | Y  | 66%   | 0,1   | 8%   | -0,09  | Y   | T   | Y   | Y   | KUNING |
| 29 | A029   | Y  | Y  | T  | Y  | Y  | Y  | 6%    | 0,11  | 8%   | -0,15  | Y   | Y   | T   | T   | MERAH  |
| 30 | A030   | Y  | Y  | Y  | T  | Y  | Y  | 18%   | 0,15  | 5%   | -0,06  | Y   | Y   | Y   | T   | MERAH  |
| 31 | A031   | Y  | Y  | Y  | Y  | Y  | Y  | 78%   | 0,2   | 10%  | 0,07   | T   | Y   | Y   | Y   | KUNING |
| 32 | A032   | Y  | T  | Y  | Y  | Y  | T  | 36%   | 0,06  | 2%   | -0,15  | Y   | T   | Y   | Y   | ORANGE |
| 33 | A033   | Y  | T  | Y  | Y  | T  | T  | 71%   | 0,17  | 8%   | 0,08   | Y   | Y   | Y   | Y   | ORANGE |
| 34 | A034   | Y  | Y  | Y  | Y  | T  | T  | 85%   | 0,11  | 5%   | 0,05   | Y   | Y   | T   | Y   | KUNING |
| 35 | A035   | Y  | Y  | Y  | Y  | T  | Y  | 67%   | 0,14  | 4%   | 0,15   | Y   | Y   | Y   | Y   | KUNING |
| 36 | A036   | Y  | Y  | T  | Y  | T  | Y  | 54%   | 0,2   | 2%   | 0,11   | Y   | Y   | T   | Y   | MERAH  |
| 37 | A037   | Y  | T  | T  | Y  | Y  | Y  | 33%   | 0,04  | 0%   | 0,19   | Y   | Y   | Y   | T   | MERAH  |
| 38 | A038   | Y  | T  | T  | Y  | Y  | Y  | 12%   | 0,16  | 4%   | -0,02  | T   | T   | Y   | Y   | MERAH  |
| 39 | A039   | Y  | Y  | Y  | T  | Y  | Y  | 0%    | 0,04  | 2%   | 0,03   | Y   | Y   | Y   | Y   | MERAH  |
| 40 | A040   | Y  | Y  | Y  | T  | Y  | Y  | 46%   | 0,16  | 9%   | 0,1    | Y   | Y   | Y   | Y   | ORANGE |
| 41 | A041   | T  | Y  | Y  | Y  | T  | T  | 63%   | 0,09  | 3%   | 0,14   | Y   | Y   | Y   | T   | ORANGE |
| 42 | A042   | T  | Y  | Y  | Y  | Y  | T  | 57%   | 0,03  | 9%   | -0,03  | Y   | Y   | T   | Y   | KUNING |
| 43 | A043   | T  | T  | Y  | Y  | Y  | Y  | 67%   | 0,15  | 2%   | -0,01  | Y   | T   | Y   | Y   | ORANGE |
| 44 | A044   | T  | Y  | T  | Y  | T  | Y  | 1%    | 0,04  | 6%   | -0,03  | Y   | Y   | Y   | Y   | MERAH  |
| 45 | A045   | Y  | Y  | T  | Y  | Y  | Y  | 61%   | 0,06  | 9%   | -0,04  | Y   | Y   | Y   | T   | KUNING |
| 46 | A046   | Y  | Y  | Y  | T  | Y  | Y  | 32%   | 0,09  | 10%  | 0,07   | Y   | Y   | Y   | Y   | ORANGE |
| 47 | A047   | Y  | Y  | Y  | T  | T  | Y  | 79%   | 0,08  | 6%   | 0,05   | Y   | Y   | T   | Y   | KUNING |
| 48 | A048   | T  | Y  | Y  | Y  | T  | Y  | 73%   | 0,17  | 3%   | 0,11   | T   | T   | Y   | Y   | ORANGE |
| 49 | A049   | Y  | T  | T  | Y  | Y  | T  | 25%   | 0,01  | 4%   | -0,04  | Y   | Y   | Y   | T   | MERAH  |
| 50 | A050   | Y  | Y  | T  | Y  | Y  | Y  | 64%   | 0,06  | 2%   | 0      | Y   | Y   | Y   | Y   | KUNING |
| 51 | A051   | Y  | Y  | Y  | Y  | Y  | Y  | 18%   | 0,06  | 3%   | -0,03  | Y   | Y   | Y   | Y   | ORANGE |
| 52 | A052   | T  | Y  | Y  | Y  | T  | Y  | 81%   | 0,11  | 7%   | 0,08   | Y   | Y   | Y   | Y   | KUNING |
| 53 | A053   | Y  | Y  | T  | Y  | T  | Y  | 26%   | 0,19  | 1%   | 0,16   | Y   | Y   | T   | T   | MERAH  |
| 54 | A054   | Y  | Y  | Y  | Y  | T  | Y  | 63%   | 0,15  | 4%   | 0,05   | Y   | T   | Y   | Y   | ORANGE |
| 55 | A055   | Y  | Y  | Y  | Y  | Y  | Y  | 58%   | 0,07  | 9%   | 0,17   | Y   | Y   | Y   | Y   | KUNING |
| 56 | A056   | T  | Y  | T  | Y  | Y  | T  | 100%  | 0,11  | 1%   | 0,03   | Y   | Y   | Y   | Y   | KUNING |
| 57 | A057   | Y  | T  | Y  | T  | Y  | T  | 26%   | 0,13  | 3%   | -0,04  | Y   | Y   | Y   | Y   | MERAH  |
| 58 | A058   | Y  | Y  | Y  | T  | T  | Y  | 18%   | 0,11  | 3%   | 0,04   | Y   | Y   | T   | T   | MERAH  |
| 59 | A059   | Y  | Y  | Y  | Y  | Y  | Y  | 88%   | 0,13  | 8%   | 0,04   | Y   | Y   | T   | Y   | KUNING |
| 60 | A060   | T  | Y  | Y  | Y  | Y  | Y  | 6%    | 0,13  | 6%   | 0,05   | Y   | T   | Y   | Y   | MERAH  |
| 61 | A061   | T  | Y  | Y  | Y  | T  | Y  | 21%   | 0,2   | 1%   | -0,02  | T   | T   | Y   | Y   | MERAH  |
| 62 | A062   | T  | Y  | T  | Y  | T  | Y  | 69%   | 0,2   | 7%   | 0,16   | Y   | T   | Y   | T   | MERAH  |
| 63 | A063   | Y  | Y  | T  | Y  | Y  | T  | 20%   | 0,03  | 8%   | -0,01  | Y   | Y   | Y   | Y   | ORANGE |
| 64 | A064   | Y  | Y  | Y  | Y  | Y  | T  | 75%   | 0,16  | 8%   | -0,02  | Y   | Y   | Y   | Y   | KUNING |
| 65 | A065   | Y  | T  | Y  | Y  | Y  | Y  | 77%   | 0,03  | 2%   | 0,03   | Y   | Y   | T   | Y   | KUNING |
| 66 | A066   | Y  | T  | Y  | T  | T  | Y  | 2%    | 0,15  | 9%   | -0,13  | Y   | Y   | Y   | T   | MERAH  |
| 67 | A067   | Y  | Y  | Y  | Y  | T  | Y  | 54%   | 0,02  | 3%   | 0,17   | Y   | T   | Y   | Y   | ORANGE |
| 68 | A068   | T  | Y  | Y  | Y  | Y  | Y  | 89%   | 0,09  | 6%   | 0,16   | Y   | Y   | Y   | Y   | KUNING |
| 69 | A069   | Y  | Y  | Y  | Y  | Y  | Y  | 63%   | 0,07  | 5%   | 0,17   | Y   | Y   | Y   | Y   | KUNING |
| 70 | A070   | Y  | Y  | Y  | Y  | Y  | Y  | 52%   | 0,05  | 7%   | 0,14   | T   | Y   | Y   | Y   | ORANGE |
| 71 | A071   | Y  | Y  | Y  | Y  | T  | Y  | 45%   | 0,11  | 1%   | -0,04  | Y   | Y   | Y   | Y   | ORANGE |
| 72 | A072   | Y  | Y  | Y  | Y  | T  | T  | 28%   | 0,04  | 8%   | -0,12  | Y   | Y   | Y   | T   | ORANGE |
| 73 | A073   | T  | T  | Y  | T  | T  | T  | 61%   | 0,05  | 7%   | 0,02   | Y   | Y   | Y   | Y   | ORANGE |
| 74 | A074   | Y  | Y  | Y  | T  | Y  | Y  | 76%   | 0,09  | 2%   | 0,03   | Y   | T   | Y   | Y   | KUNING |
| 75 | A075   | Y  | Y  | Y  | Y  | Y  | Y  | 4%    | 0,2   | 2%   | -0,04  | T   | Y   | T   | T   | MERAH  |
| 76 | A076   | T  | Y  | Y  | Y  | Y  | Y  | 88%   | 0,12  | 7%   | 0,18   | Y   | Y   | Y   | Y   | KUNING |
| 77 | A077   | Y  | Y  | Y  | Y  | Y  | Y  | 5%    | 0,09  | 1%   | -0,11  | Y   | Y   | Y   | Y   | ORANGE |
| 78 | A078   | Y  | T  | T  | Y  | T  | Y  | 78%   | 0,03  | 5%   | -0,12  | Y   | Y   | Y   | Y   | KUNING |
| 79 | A079   | Y  | Y  | Y  | Y  | Y  | Y  | 20%   | 0,02  | 9%   | 0,16   | Y   | T   | T   | T   | MERAH  |
| 80 | A080   | T  | Y  | Y  | Y  | Y  | T  | 0%    | 0,15  | 9%   | -0,14  | T   | Y   | Y   | Y   | MERAH  |
| 81 | A081   | T  | Y  | Y  | T  | Y  | Y  | 49%   | 0,14  | 3%   | 0,04   | Y   | Y   | Y   | Y   | ORANGE |
| 82 | A082   | Y  | Y  | Y  | T  | Y  | Y  | 95%   | 0,17  | 7%   | 0,04   | Y   | Y   | T   | Y   | KUNING |
| 83 | A083   | Y  | T  | Y  | Y  | T  | Y  | 52%   | 0,11  | 5%   | -0,15  | Y   | Y   | Y   | Y   | KUNING |
| 84 | A084   | T  | Y  | Y  | Y  | Y  | Y  | 46%   | 0,15  | 5%   | -0,02  | Y   | T   | Y   | T   | ORANGE |
| 85 | A085   | T  | Y  | Y  | Y  | Y  | Y  | 25%   | 0,13  | 1%   | -0,12  | Y   | T   | Y   | Y   | ORANGE |
| 86 | A086   | Y  | Y  | Y  | Y  | T  | Y  | 78%   | 0,07  | 8%   | 0,2    | T   | Y   | Y   | Y   | KUNING |
| 87 | A087   | Y  | Y  | Y  | Y  | Y  | Y  | 84%   | 0,17  | 8%   | -0,15  | Y   | Y   | Y   | Y   | KUNING |
| 88 | A088   | Y  | Y  | Y  | Y  | Y  | T  | 4%    | 0,14  | 8%   | -0,02  | Y   | Y   | T   | T   | MERAH  |
| 89 | A089   | Y  | Y  | Y  | T  | Y  | Y  | 96%   | 0,17  | 3%   | 0,16   | Y   | Y   | Y   | Y   | KUNING |
| 90 | A090   | Y  | Y  | T  | Y  | T  | Y  | 10%   | 0,06  | 6%   | -0,12  | Y   | Y   | Y   | Y   | ORANGE |
| 91 | A091   | T  | Y  | Y  | Y  | Y  | Y  | 53%   | 0,17  | 9%   | -0,07  | Y   | T   | Y   | Y   | KUNING |
| 92 | A092   | T  | T  | Y  | Y  | Y  | Y  | 93%   | 0,07  | 3%   | -0,03  | T   | Y   | Y   | Y   | KUNING |
| 93 | A093   | Y  | Y  | Y  | Y  | Y  | T  | 41%   | 0,08  | 10%  | -0,11  | Y   | Y   | T   | Y   | KUNING |
| 94 | A094   | Y  | Y  | Y  | T  | T  | Y  | 20%   | 0,12  | 9%   | 0,08   | Y   | Y   | Y   | T   | MERAH  |
| 95 | A095   | Y  | Y  | Y  | Y  | Y  | Y  | 92%   | 0,19  | 9%   | -0,1   | Y   | Y   | Y   | Y   | KUNING |
| 96 | A096   | T  | Y  | Y  | Y  | Y  | Y  | 97%   | 0,09  | 10%  | -0,02  | Y   | Y   | Y   | Y   | KUNING |
| 97 | A097   | Y  | Y  | Y  | Y  | Y  | T  | 23%   | 0,17  | 0%   | 0,11   | T   | Y   | Y   | Y   | MERAH  |
| 98 | A098   | T  | Y  | Y  | Y  | T  | Y  | 15%   | 0,12  | 3%   | 0,04   | Y   | Y   | T   | Y   | MERAH  |
| 99 | A099   | Y  | T  | Y  | T  | Y  | Y  | 21%   | 0,14  | 7%   | -0,14  | Y   | Y   | Y   | Y   | ORANGE |
| 100| A100   | T  | Y  | T  | Y  | Y  | Y  | 47%   | 0,17  | 8%   | 0      | Y   | Y   | T   | Y   | ORANGE |

### Dataset Baru (Sebelum Preprocessing)

Berikut adalah dataset baru yang perlu diklasifikasikan:

| No | Daerah | C1 | C2 | C3 | C4 | C5 | C6 | C7   | C8    | C9   | C10    | C11 | C12 | C13 | C14 |
|----|--------|----|----|----|----|----|----|------|-------|------|--------|-----|-----|-----|-----|
| 1  | A101   | Y  | Y  | Y  | T  | Y  | Y  | 50%  | 0,25  | 3%   | -0,1   | Y   | Y   | Y   | Y   |
| 2  | A102   | Y  | T  | Y  | Y  | T  | T  | 10%  | 0,1   | 2%   | 0,1    | Y   | T   | T   | Y   |
| 3  | A103   | Y  | Y  | Y  | Y  | Y  | Y  | 25%  | 0,1   | 4%   | 0,02   | Y   | Y   | Y   | Y   |
| 4  | A104   | Y  | Y  | Y  | T  | Y  | Y  | 30%  | 0,15  | 2%   | -0,03  | T   | Y   | Y   | Y   |
| 5  | A105   | Y  | T  | Y  | T  | Y  | Y  | 90%  | 0,01  | 5%   | -0,15  | Y   | Y   | Y   | T   |

## Metodologi

### 1. Praproses Data

Data diproses dengan langkah-langkah berikut:

#### 1.1 Konversi Data Kategorikal ke Numerik
- Atribut C1-C6, C11-C14 yang bernilai Y diubah menjadi angka 1, dan T menjadi angka 0.
- Transformasi: Y → 1, T → 0

#### 1.2 Normalisasi Persentase ke Desimal
- Atribut C7, C9 yang berupa persentase dikonversi ke bentuk desimal.
- Transformasi: X% → X/100
  - Contoh: 50% → 0.50, 3% → 0.03

#### 1.3 Standarisasi Format Data Numerik
- Atribut C8 dan C10 yang sudah dalam bentuk desimal distandarisasi dengan mengganti notasi koma (,) dengan titik (.).
- Transformasi: X,Y → X.Y
  - Contoh: 0,25 → 0.25

### 2. Normalisasi Fitur dengan Min-Max Scaling

Min-Max Scaling diaplikasikan pada semua fitur untuk menyamakan skala data dan memperbaiki performa algoritma KNN. Formula yang digunakan adalah:

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

Dimana:
- $X_{norm}$ adalah nilai yang dinormalisasi
- $X$ adalah nilai asli
- $X_{min}$ adalah nilai minimum fitur tersebut
- $X_{max}$ adalah nilai maksimum fitur tersebut

Hasil normalisasi membuat semua fitur berada dalam rentang [0, 1], membuat perhitungan jarak lebih konsisten.

### Dataset Training Setelah Normalisasi

Berikut adalah data training setelah normalisasi dengan Min-Max Scaling:

| No | Daerah | C1    | C2    | C3    | C4    | C5    | C6    | C7    | C8    | C9    | C10   | C11   | C12   | C13   | C14   | Zona   |
|----|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| 1  | A001   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.760 | 0.550 | 0.400 | 0.179 | 1.000 | 1.000 | 1.000 | 1.000 | KUNING |
| 2  | A002   | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.120 | 0.650 | 0.100 | 0.683 | 1.000 | 0.000 | 1.000 | 1.000 | MERAH  |
| 3  | A003   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.230 | 0.000 | 0.200 | 0.963 | 1.000 | 1.000 | 1.000 | 1.000 | ORANGE |
| 4  | A004   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.790 | 0.600 | 0.200 | 0.359 | 0.000 | 1.000 | 0.000 | 1.000 | KUNING |
| 5  | A005   | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.900 | 0.300 | 0.500 | 0.201 | 1.000 | 1.000 | 1.000 | 0.000 | KUNING |
| 6  | A006   | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.100 | 0.000 | 0.800 | 0.179 | 1.000 | 1.000 | 1.000 | 1.000 | ORANGE |
| 7  | A007   | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.640 | 0.150 | 0.500 | 0.693 | 1.000 | 1.000 | 1.000 | 1.000 | ORANGE |
| 8  | A008   | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.470 | 0.650 | 0.100 | 0.559 | 1.000 | 1.000 | 1.000 | 1.000 | ORANGE |
| 9  | A009   | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.910 | 0.650 | 0.500 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | KUNING |
| 10 | A010   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.300 | 0.900 | 0.900 | 0.214 | 0.000 | 1.000 | 1.000 | 1.000 | ORANGE |
| ... | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...    |
| 91 | A091   | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.530 | 0.850 | 0.900 | 0.257 | 1.000 | 0.000 | 1.000 | 1.000 | KUNING |
| 92 | A092   | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.930 | 0.350 | 0.300 | 0.373 | 0.000 | 1.000 | 1.000 | 1.000 | KUNING |
| 93 | A093   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.410 | 0.400 | 1.000 | 0.201 | 1.000 | 1.000 | 0.000 | 1.000 | KUNING |
| 94 | A094   | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.200 | 0.600 | 0.900 | 0.622 | 1.000 | 1.000 | 1.000 | 0.000 | MERAH  |
| 95 | A095   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.920 | 0.950 | 0.900 | 0.214 | 1.000 | 1.000 | 1.000 | 1.000 | KUNING |
| 96 | A096   | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.970 | 0.450 | 1.000 | 0.386 | 1.000 | 1.000 | 1.000 | 1.000 | KUNING |
| 97 | A097   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.230 | 0.850 | 0.000 | 0.693 | 0.000 | 1.000 | 1.000 | 1.000 | MERAH  |
| 98 | A098   | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.150 | 0.600 | 0.300 | 0.559 | 1.000 | 1.000 | 0.000 | 1.000 | MERAH  |
| 99 | A099   | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.210 | 0.700 | 0.700 | 0.165 | 1.000 | 1.000 | 1.000 | 1.000 | ORANGE |
| 100| A100   | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.470 | 0.850 | 0.800 | 0.400 | 1.000 | 1.000 | 0.000 | 1.000 | ORANGE |

### Dataset Baru Setelah Normalisasi

Berikut adalah dataset baru setelah normalisasi dengan Min-Max Scaling:

| No | Daerah | C1    | C2    | C3    | C4    | C5    | C6    | C7    | C8    | C9    | C10   | C11   | C12   | C13   | C14   |
|----|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1  | A101   | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.300 | 0.214 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2  | A102   | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.100 | 0.500 | 0.200 | 0.683 | 1.000 | 0.000 | 0.000 | 1.000 |
| 3  | A103   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 0.500 | 0.400 | 0.514 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4  | A104   | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.300 | 0.750 | 0.200 | 0.386 | 0.000 | 1.000 | 1.000 | 1.000 |
| 5  | A105   | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.900 | 0.050 | 0.500 | 0.143 | 1.000 | 1.000 | 1.000 | 0.000 |

### 3. Pemodelan dengan KNN

#### 3.1 Algoritma KNN

K-Nearest Neighbors (KNN) adalah algoritma klasifikasi yang mengklasifikasikan objek baru berdasarkan mayoritas kelas dari k tetangga terdekatnya. Langkah-langkah algoritma KNN:

1. Tentukan nilai k (jumlah tetangga terdekat yang akan diperhatikan)
2. Hitung jarak antar titik data baru dengan semua titik data training
3. Urutkan jarak dan tentukan k tetangga terdekat berdasarkan jarak terkecil
4. Tentukan kelas mayoritas dari k tetangga terdekat
5. Berikan kelas mayoritas sebagai prediksi untuk titik data baru

#### 3.2 Perhitungan Jarak Euclidean

Jarak antara dua titik data dihitung menggunakan jarak Euclidean:

$$d(p,q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

Dimana:
- $d(p,q)$ adalah jarak antara titik p dan q
- $p_i$ adalah nilai fitur ke-i dari titik p
- $q_i$ adalah nilai fitur ke-i dari titik q
- $n$ adalah jumlah fitur

#### 3.3 Voting Kelas Mayoritas

Setelah menemukan k tetangga terdekat, kelas yang paling banyak muncul di antara tetangga tersebut dipilih sebagai hasil klasifikasi. Jika terjadi seri, kelas dengan jarak total terkecil dipilih.

#### 3.4 Perhitungan Probabilitas

Probabilitas suatu kelas dihitung berdasarkan proporsi kelas tersebut di antara k tetangga terdekat:

$$P(class_i) = \frac{n_i}{k}$$

Dimana:
- $P(class_i)$ adalah probabilitas kelas i
- $n_i$ adalah jumlah tetangga dengan kelas i
- $k$ adalah jumlah total tetangga yang diperhitungkan

### 4. Parameter KNN

Pemodelan dilakukan dengan algoritma K-Nearest Neighbors (KNN) menggunakan parameter:

- **K = 33**: Menggunakan 33 tetangga terdekat untuk klasifikasi
- **Distance Metric**: Euclidean distance
- **Normalisasi**: Min-Max scaling untuk semua fitur
- **Weight**: Uniform (semua tetangga memiliki bobot yang sama)

Nilai K=33 dipilih karena:
1. Berjumlah sekitar 1/3 dari total data training
2. Memperhitungkan distribusi kelas yang cukup seimbang
3. Mengurangi dampak outlier
4. Mengurangi overfitting terhadap noise dalam data

## Hasil Klasifikasi

Hasil klasifikasi dengan K=33 untuk 5 daerah baru adalah sebagai berikut:

| No | Daerah | Zona Prediksi | Probabilitas KUNING | Probabilitas MERAH | Probabilitas ORANGE |
|----|--------|---------------|---------------------|--------------------|--------------------|
| 1  | A101   | KUNING        | 48.48%              | 15.15%             | 36.36%             |
| 2  | A102   | KUNING        | 36.36%              | 27.27%             | 36.36%             |
| 3  | A103   | KUNING        | 42.42%              | 15.15%             | 42.42%             |
| 4  | A104   | KUNING        | 42.42%              | 18.18%             | 39.39%             |
| 5  | A105   | KUNING        | 42.42%              | 27.27%             | 30.30%             |

### Tetangga Terdekat untuk Setiap Daerah

#### Daerah A101

10 tetangga terdekat:
1. Daerah A040 (Zona ORANGE), jarak: 0.9437
2. Daerah A089 (Zona KUNING), jarak: 0.9610
3. Daerah A046 (Zona ORANGE), jarak: 1.1825
4. Daerah A030 (Zona MERAH), jarak: 1.1855
5. Daerah A081 (Zona ORANGE), jarak: 1.2094
6. Daerah A039 (Zona MERAH), jarak: 1.2249
7. Daerah A087 (Zona KUNING), jarak: 1.2434
8. Daerah A099 (Zona ORANGE), jarak: 1.2489
9. Daerah A001 (Zona KUNING), jarak: 1.2550
10. Daerah A095 (Zona KUNING), jarak: 1.2753

Dari 33 tetangga terdekat: 16 KUNING, 5 MERAH, 12 ORANGE.

#### Daerah A102

10 tetangga terdekat:
1. Daerah A002 (Zona MERAH), jarak: 1.4258
2. Daerah A032 (Zona ORANGE), jarak: 1.6180
3. Daerah A034 (Zona KUNING), jarak: 1.6357
4. Daerah A033 (Zona ORANGE), jarak: 1.6905
5. Daerah A067 (Zona ORANGE), jarak: 1.8449
6. Daerah A054 (Zona ORANGE), jarak: 1.8449
7. Daerah A015 (Zona MERAH), jarak: 1.8657
8. Daerah A065 (Zona KUNING), jarak: 1.9004
9. Daerah A083 (Zona KUNING), jarak: 1.9440
10. Daerah A016 (Zona MERAH), jarak: 1.9441

Dari 33 tetangga terdekat: 12 KUNING, 9 MERAH, 12 ORANGE.

#### Daerah A103

10 tetangga terdekat:
1. Daerah A051 (Zona ORANGE), jarak: 0.2744
2. Daerah A077 (Zona ORANGE), jarak: 0.5201
3. Daerah A018 (Zona MERAH), jarak: 0.5647
4. Daerah A069 (Zona KUNING), jarak: 0.6005
5. Daerah A001 (Zona KUNING), jarak: 0.6680
6. Daerah A003 (Zona ORANGE), jarak: 0.7255
7. Daerah A055 (Zona KUNING), jarak: 0.7517
8. Daerah A087 (Zona KUNING), jarak: 0.9309
9. Daerah A095 (Zona KUNING), jarak: 1.0094
10. Daerah A008 (Zona ORANGE), jarak: 1.0778

Dari 33 tetangga terdekat: 14 KUNING, 5 MERAH, 14 ORANGE.

#### Daerah A104

10 tetangga terdekat:
1. Daerah A039 (Zona MERAH), jarak: 1.1924
2. Daerah A010 (Zona ORANGE), jarak: 1.2460
3. Daerah A040 (Zona ORANGE), jarak: 1.2869
4. Daerah A089 (Zona KUNING), jarak: 1.3230
5. Daerah A070 (Zona ORANGE), jarak: 1.3358
6. Daerah A046 (Zona ORANGE), jarak: 1.3461
7. Daerah A031 (Zona KUNING), jarak: 1.4193
8. Daerah A081 (Zona ORANGE), jarak: 1.4452
9. Daerah A030 (Zona MERAH), jarak: 1.4532
10. Daerah A077 (Zona ORANGE), jarak: 1.4882

Dari 33 tetangga terdekat: 14 KUNING, 6 MERAH, 13 ORANGE.

#### Daerah A105

10 tetangga terdekat:
1. Daerah A005 (Zona KUNING), jarak: 1.0371
2. Daerah A022 (Zona MERAH), jarak: 1.1057
3. Daerah A023 (Zona ORANGE), jarak: 1.3207
4. Daerah A014 (Zona ORANGE), jarak: 1.3311
5. Daerah A099 (Zona ORANGE), jarak: 1.3926
6. Daerah A030 (Zona MERAH), jarak: 1.4403
7. Daerah A066 (Zona MERAH), jarak: 1.5581
8. Daerah A057 (Zona MERAH), jarak: 1.7054
9. Daerah A008 (Zona ORANGE), jarak: 1.7232
10. Daerah A046 (Zona ORANGE), jarak: 1.7724

Dari 33 tetangga terdekat: 14 KUNING, 9 MERAH, 10 ORANGE.

### Distribusi Kelas pada Data Training

- **KUNING**: 36 data (36.00%)
- **ORANGE**: 34 data (34.00%)
- **MERAH**: 30 data (30.00%)

### Analisis Hasil

1. **Daerah A101**: Diklasifikasikan sebagai KUNING dengan probabilitas 48.48%. Tetangga terdekat adalah A040 (ORANGE) dan A089 (KUNING). Probabilitas ORANGE cukup tinggi (36.36%), menunjukkan adanya ambiguitas antara kedua zona tersebut.

2. **Daerah A102**: Diklasifikasikan sebagai KUNING dengan probabilitas 36.36%. Menariknya, probabilitas untuk ORANGE juga 36.36%, menunjukkan ambiguitas sempurna dalam klasifikasi. Perhitungan tie-breaking menggunakan rata-rata jarak memberikan sedikit keunggulan pada zona KUNING.

3. **Daerah A103**: Diklasifikasikan sebagai KUNING dengan probabilitas 42.42%, dan probabilitas ORANGE juga 42.42%. Kesamaan probabilitas ini menunjukkan daerah ini berada tepat di perbatasan klasifikasi. Tie-breaking berdasarkan rata-rata jarak memberikan hasil KUNING dengan selisih minimal.

4. **Daerah A104**: Diklasifikasikan sebagai KUNING dengan probabilitas 42.42%. Probabilitas ORANGE juga cukup tinggi (39.39%), menunjukkan kedekatan dengan zona tersebut.

5. **Daerah A105**: Diklasifikasikan sebagai KUNING dengan probabilitas 42.42%. Probabilitas MERAH (27.27%) dan ORANGE (30.30%) menunjukkan distribusi yang cukup merata di antara tiga kelas, meskipun cenderung ke KUNING.

### Dampak Nilai K=33

Penggunaan K=33 untuk 100 data training berarti model mempertimbangkan 33% dari seluruh data training untuk setiap prediksi. Nilai K yang besar ini menghasilkan:

1. **Mayoritas prediksi KUNING**: Cenderung memprediksi zona KUNING karena distribusi kelas training menunjukkan KUNING sebagai kelas mayoritas (36% dari data).

2. **Smoothing effect**: Nilai K yang besar mengurangi dampak outlier dan noise, tetapi juga mengurangi sensitifitas model terhadap pola lokal yang mungkin penting.

3. **Probabilitas yang berdekatan**: Distribusi probabilitas antar kelas cenderung lebih merata, menunjukkan tingkat ambiguitas yang lebih tinggi dalam klasifikasi.

4. **Dominasi kelas mayoritas**: Dengan K yang besar, kelas mayoritas dalam dataset training (KUNING) mendapatkan keuntungan dalam voting, terutama untuk kasus-kasus yang berada di perbatasan.

### Perhitungan Sederhana untuk Kasus A102

Untuk kasus A102 yang memiliki tie antara KUNING dan ORANGE (masing-masing 36.36% atau 12/33 dari tetangga):
- 12 tetangga adalah KUNING
- 9 tetangga adalah MERAH
- 12 tetangga adalah ORANGE

Karena ada tie, rata-rata jarak ke setiap kelas dihitung:
- Rata-rata jarak ke tetangga KUNING: 2.1058
- Rata-rata jarak ke tetangga ORANGE: 2.1693
- Rata-rata jarak ke tetangga MERAH: 2.3201

Kelas dengan rata-rata jarak terkecil (KUNING) dipilih sebagai hasil klasifikasi.

## Struktur Repositori

Repositori ini terdiri dari file-file berikut:

1. `dataset Covid.csv` - Dataset training dengan 100 daerah yang sudah terlabel
2. `dataset baru.csv` - Dataset baru dengan 5 daerah yang perlu diklasifikasi
3. `knn_classifier.py` - Skrip Python untuk pemrosesan data dan klasifikasi KNN
4. `training_data_processed.csv` - Data training yang sudah diproses
5. `new_data_processed.csv` - Data baru yang sudah diproses
6. `hasil_klasifikasi_detail_k33.csv` - Hasil klasifikasi lengkap dengan probabilitas untuk K=33
7. `README.md` - Dokumentasi proyek

## Cara Menjalankan

1. Pastikan Python dan library berikut telah terinstal:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn

2. Jalankan script knn_classifier.py:
   ```
   python knn_classifier.py
   ```

3. Hasil klasifikasi akan disimpan dalam file `hasil_klasifikasi_detail_k33.csv`

## Pseudocode

Berikut adalah pseudocode dari algoritma KNN yang diimplementasikan:

```
function KNN_Classifier(training_data, new_data, k):
    # Preprocessing
    processed_training_data = preprocess(training_data)
    processed_new_data = preprocess(new_data)
    
    # Extract features and target
    X_train = processed_training_data[features]
    y_train = processed_training_data[target]
    X_new = processed_new_data[features]
    
    # Normalize features
    X_train_normalized = normalize(X_train)
    X_new_normalized = normalize(X_new, using_same_parameters)
    
    # Initialize results
    predictions = []
    probabilities = []
    
    # For each new data point
    for i in range(len(X_new_normalized)):
        # Calculate distances to all training points
        distances = []
        for j in range(len(X_train_normalized)):
            distance = euclidean_distance(X_new_normalized[i], X_train_normalized[j])
            distances.append((distance, y_train[j]))
        
        # Sort by distance and get k nearest neighbors
        distances.sort()
        k_nearest = distances[:k]
        
        # Count occurrences of each class
        class_counts = count_classes(k_nearest)
        
        # Get prediction (class with maximum count)
        prediction = get_max_class(class_counts)
        
        # Calculate probabilities
        probabilities_i = calculate_probabilities(class_counts, k)
        
        predictions.append(prediction)
        probabilities.append(probabilities_i)
    
    return predictions, probabilities
```

## Kesimpulan

Klasifikasi dengan KNN (K=33) menghasilkan semua daerah diklasifikasikan sebagai zona KUNING. Namun, probabilitas yang berdekatan antar kelas menunjukkan adanya ambiguitas dalam klasifikasi. Nilai K yang besar ini memperkuat bias model terhadap kelas mayoritas (KUNING), yang perlu dipertimbangkan dalam interpretasi hasil.

Beberapa hal penting untuk dipertimbangkan:

1. Pada beberapa kasus (A102 dan A103), terjadi tie dalam voting kelas, yang diselesaikan dengan mempertimbangkan rata-rata jarak.

2. Nilai K=33 mungkin terlalu besar dan menyebabkan over-smoothing, mengurangi sensitifitas model terhadap variasi lokal yang penting.

3. Distribusi kelas yang tidak seimbang dalam data training (KUNING 36%, ORANGE 34%, MERAH 30%) mempengaruhi hasil klasifikasi terutama untuk kasus-kasus yang berada di perbatasan.

4. Untuk mendapatkan hasil yang lebih robust, perlu dipertimbangkan untuk menggunakan beberapa nilai K yang berbeda dan metode validasi silang untuk memilih K optimal. 
