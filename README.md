# Studi komparatif model generatif

Eksperimen Fashion-MNIST untuk Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM. MLP classifier dan decoder fitur beku dipakai untuk membandingkan rekonstruksi dan sampling dari ruang fitur klasifikasi.

## Menjalankan kode

Gunakan Python 3.10 atau lebih baru. Untuk GPU, pasang PyTorch dan torchvision dengan build CUDA yang cocok untuk mesin. Dari direktori repositori, jalankan:

```bash
python -m pip install -r requirements.txt
python main.py --profile smoke --device cpu
python main.py --profile full --device cuda
```

Profil smoke memakai subset dan satu epoch untuk memeriksa alur. Profil full melatih semua model pada 56.000/7.000/7.000 gambar dan memakai FID secara default. Run pertama mengunduh Fashion-MNIST dan bobot Inception.

## Menjalankan per tahap

```bash
python main.py --profile full --device cuda --stage train
python main.py --profile full --device cuda --stage evaluate
python main.py --profile full --stage report --identity identity.json
```

Isi `identity.json` dengan kunci `name`, `nim`, `class`, dan `github`. Nama PDF mengikuti format tugas setelah identitas diisi. Jalankan evaluasi dan laporan di folder output yang sama dengan hasil training. Gunakan `--output` untuk menunjuk folder lain. Tahap training dapat diteruskan dari checkpoint epoch terakhir. Pakai folder output baru saat mengubah konfigurasi.

## Hasil eksperimen Tesla T4

Folder `hasil-tesla-t4/` berisi keluaran asli profil `full` berbasis seed 42. Split terdiri dari 56.000 citra latihan, 7.000 citra validasi, dan 7.000 citra test. DCGAN memperoleh FID 39,71, Vanilla AE memperoleh MSE test 0,007689, dan MLP mencapai akurasi test 89,37%.

Buka `metrics.csv` untuk perbandingan enam model dan `results.json` untuk evaluasi MLP serta metrik tambahan. Lima gambar dan riwayat tiap epoch ada di folder yang sama. Buat PDF dari hasil yang sudah tersimpan dengan perintah berikut.

```bash
python main.py --stage report --output hasil-tesla-t4 --identity identity.json
```

## Isi folder hasil

| Berkas | Isi |
|---|---|
| `run_config.json`, `split_indices.npz` | Parameter run, perangkat, versi PyTorch, dan indeks data |
| `history_*.csv`, `checkpoints/` | Riwayat tiap epoch dan bobot model |
| `metrics.csv`, `results.json` | Metrik model, eksperimen MLP, dan benchmark |
| `class_coverage.csv` | Frekuensi kelas real dan kelas prediksi sampel |
| `figur_*.png` | Lima figur utama |
| PDF laporan | Pembahasan dari data dan figur run |

## Evaluasi

- Semua model memakai split yang sama. VAE merekonstruksi dengan mean posterior. PSNR dan SSIM dirata-ratakan per gambar.
- FID membandingkan 3.000 gambar test dan 3.000 sampel per model pada profil full. Citra grayscale direplikasi ke tiga kanal dan dinilai dengan Inception.
- Latensi merupakan median setelah lima warm-up. DDPM diukur untuk seluruh reverse sampling. AE deterministik diukur dari encode sampai decode.
- Kelas prediksi MLP menjadi proxy cakupan mode. AUROC anomali merupakan proxy menggunakan citra test dengan modifikasi piksel terkontrol.
- PDF dibangun dari hasil aktual. Baca kembali figur dan hasil secara visual sebelum menyelesaikan pembahasan.

## Struktur

`models.py` memuat arsitektur. `experiment.py` mengelola data, training, dan checkpoint. `evaluate.py` menghitung metrik dan membuat figur. `report.py` membaca angka dan figur untuk membuat PDF. `main.py` menjalankan tahap yang dipilih.
