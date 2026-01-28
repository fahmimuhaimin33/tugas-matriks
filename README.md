# Tugas Matriks – Aplikasi Web

Aplikasi Flask sederhana untuk membantu menyelesaikan tugas Aljabar Linear dan Matriks:

- Operasi matriks (soal no. 1): menghitung \(CA\), \(A^T + B\), dan \((CB)^T\) lengkap dengan langkah baris–kolom.
- Invers matriks dengan metode Operasi Baris Elementer (Gauss–Jordan), beserta setiap langkah transformasi \([A \mid I]\).
- Penyelesaian SPL 3 variabel menggunakan Aturan Cramer, menampilkan \(\det(A)\) dan determinan tiap matriks pengganti kolom.

## Menjalankan secara lokal

```bash
python -m venv .venv
.venv\Scripts\activate  # di Windows
pip install -r requirements.txt
python app.py
```

Aplikasi akan berjalan di `http://127.0.0.1:5000`.

## Deploy ke Railway

Railway akan membaca file `Procfile`:

```text
web: python app.py
```

Langkah umum:

1. Push kode ke GitHub repo ini.
2. Di Railway, buat **New Project → Deploy from GitHub** lalu pilih repositori `tugas_matriks`.
3. Railway akan:
   - Membaca `requirements.txt` untuk meng-install dependency (`Flask`, `sympy`).
   - Menjalankan perintah pada `Procfile` (`web: python app.py`).
4. Setelah deploy sukses, ambil URL public yang diberikan Railway dan gunakan sebagai link aplikasi.

