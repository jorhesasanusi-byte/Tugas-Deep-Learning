import json
import re
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from matplotlib.font_manager import FontProperties, findfont
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from experiment import MODEL_NAMES


INK = colors.HexColor("#18332A")
MUTED = colors.HexColor("#4B6056")
RULE = colors.HexColor("#D5DFD9")
PALE = colors.HexColor("#EEF4F0")
PAGE_WIDTH = 210 * mm
PAGE_HEIGHT = 297 * mm
BODY_WIDTH = 172 * mm


def f(value, decimals=4):
    return "—" if value is None else f"{value:.{decimals}f}"


def slug(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")


def register_fonts():
    pdfmetrics.registerFont(TTFont("Body", findfont(FontProperties(family="DejaVu Sans"))))
    pdfmetrics.registerFont(TTFont("Bold", findfont(FontProperties(family="DejaVu Sans", weight="bold"))))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold")


def architecture_diagrams():
    rows = [
        ("Vanilla AE", "Citra", "Encoder MLP", "z 64", "Decoder MLP"),
        ("Sparse AE", "Citra", "Encoder MLP", "z 64 + L1", "Decoder MLP"),
        ("VAE", "Citra", "μ, log σ²", "Sampling z", "Decoder MLP"),
        ("Vanilla GAN", "Noise z", "G MLP", "Citra", "D MLP"),
        ("DCGAN", "Noise z", "G ConvT", "Citra", "D Conv"),
        ("DDPM", "Citra/noise, t", "Blok down", "Blok up + skip", "Prediksi ε"),
        ("MLP", "Citra", "MLP + ReLU", "Fitur 64", "Logit 10"),
    ]
    canvas = Drawing(BODY_WIDTH, 135)
    for i, (label, *stages) in enumerate(rows):
        y = 112 - i * 17
        canvas.add(String(2, y + 4, label, fontName="Bold", fontSize=7.5, fillColor=INK))
        for j, stage in enumerate(stages):
            x = 86 + j * 98
            canvas.add(Rect(x, y, 79, 13, fillColor=PALE, strokeColor=RULE, strokeWidth=0.5))
            canvas.add(String(x + 39.5, y + 3.4, stage, textAnchor="middle", fontName="Body",
                              fontSize=7.5, fillColor=INK))
            if j < len(stages) - 1:
                canvas.add(Line(x + 80, y + 6.5, x + 94, y + 6.5, strokeColor=MUTED, strokeWidth=0.7))
                canvas.add(Polygon([x + 94, y + 6.5, x + 90, y + 8.8, x + 90, y + 4.2],
                                   strokeColor=None, fillColor=MUTED))
    return canvas


def footer(canvas, doc):
    canvas.setStrokeColor(RULE)
    canvas.line(19 * mm, 14 * mm, 191 * mm, 14 * mm)
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(19 * mm, 9 * mm, "Studi komparatif model generatif · Deep Learning")
    canvas.drawRightString(191 * mm, 9 * mm, str(doc.page))


def build_report(folder, identity):
    register_fonts()
    meta = json.loads((folder / "run_config.json").read_text())
    cfg = meta["settings"]
    result = json.loads((folder / "results.json").read_text())
    rows = {row["model"]: row for row in result["metric_table"]}
    mode_rows = result["generation"]
    anomaly = result["anomaly_auc"]
    accuracy = result["mlp_accuracy"]
    smoke = cfg["profile"] == "smoke"

    styles = {
        "cover": ParagraphStyle("cover", fontName="Bold", fontSize=25, leading=31, textColor=INK, spaceAfter=14),
        "h1": ParagraphStyle("h1", fontName="Bold", fontSize=19, leading=24, textColor=INK, spaceAfter=12),
        "h2": ParagraphStyle("h2", fontName="Bold", fontSize=11, leading=15, textColor=INK, spaceBefore=8, spaceAfter=4),
        "p": ParagraphStyle("p", fontName="Body", fontSize=9.6, leading=15, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("small", fontName="Body", fontSize=8, leading=12, textColor=MUTED, spaceAfter=4),
        "table": ParagraphStyle("table", fontName="Body", fontSize=7.8, leading=11, textColor=INK),
        "header": ParagraphStyle("header", fontName="Bold", fontSize=7.8, leading=11, textColor=colors.white),
        "caption": ParagraphStyle("caption", fontName="Body", fontSize=8.4, leading=12, textColor=MUTED,
                                  alignment=TA_CENTER, spaceBefore=7, spaceAfter=8),
    }
    story = []

    def paragraph(text, style="p"):
        story.append(Paragraph(escape(str(text)), styles[style]))

    def new_page(title):
        story.append(PageBreak())
        paragraph(title, "h1")

    def table(header, values, widths):
        data = [
            [Paragraph(escape(str(value)), styles["header" if i == 0 else "table"]) for value in row]
            for i, row in enumerate([header] + list(values))
        ]
        block = Table(data, colWidths=[BODY_WIDTH * ratio for ratio in widths])
        block.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
        ]))
        story.append(block)
        story.append(Spacer(1, 9))

    def picture(filename, caption, height=142):
        story.append(Image(str(folder / filename), width=BODY_WIDTH, height=height * mm, kind="proportional"))
        paragraph(caption, "caption")

    reconstructed = MODEL_NAMES[:3]
    best_mse = min(reconstructed, key=lambda name: rows[name]["test_mse"])
    fid_models = [name for name in MODEL_NAMES if rows[name]["fid"] is not None]
    fid_best = min(fid_models, key=lambda name: rows[name]["fid"]) if fid_models else None
    slow = rows["DDPM"]["latency_batch64_ms"]
    fastest = min([name for name in MODEL_NAMES[2:5]], key=lambda name: rows[name]["latency_batch64_ms"])

    story.append(Spacer(1, 22 * mm))
    paragraph("Deep Learning · Sesi 07", "small")
    paragraph("Studi Komparatif Empiris Model Generatif", "cover")
    paragraph("Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM", "h2")
    paragraph("Dengan eksperimen proyeksi dan rekonstruksi fitur MLP Classifier")
    table(["Identitas", "Keterangan"], [
        ("Nama lengkap", identity["name"] or "[isi nama lengkap]"),
        ("NIM", identity["nim"] or "[isi NIM]"),
        ("Kelas", identity["class"] or "[isi kelas]"),
    ], [0.25, 0.75])
    url = identity["github"].strip()
    if url.startswith("https://github.com/"):
        story.append(Paragraph(f"Repositori publik · <link href={quoteattr(url)} color='#196045'>"
                               f"{escape(url)}</link>", styles["p"]))
    else:
        paragraph("Repositori publik · [isi URL GitHub setelah repositori tersedia]")
    if smoke:
        paragraph("PROFIL UJI TEKNIS · Hasil di PDF ini berasal dari data dan epoch yang diperkecil.", "h2")
    paragraph("Abstrak", "h2")
    fid_text = (f"FID terendah diperoleh {fid_best} ({f(rows[fid_best]['fid'], 2)})."
                if fid_best else "FID belum tersedia pada run ini.")
    paragraph(
        f"Enam model dilatih dengan pembagian Fashion-MNIST yang sama. "
        f"Evaluasi memakai {result['test_count']:,} citra untuk rekonstruksi, sampel prior untuk generasi, "
        f"dan inferensi batch 64 untuk latensi. {best_mse} memperoleh MSE terendah "
        f"({f(rows[best_mse]['test_mse'], 6)}). {fid_text} "
        f"MLP memperoleh akurasi {accuracy:.2%}. "
        "Uji proyeksi dan dekoder beku menilai informasi spasial yang dapat diambil dari fitur diskriminatif."
    )

    new_page("1. Data dan arsitektur")
    split = [cfg.get("train_limit") or 56_000, cfg.get("val_limit") or 7000, cfg.get("test_limit") or 7000]
    paragraph(
        f"Fashion-MNIST memuat 70.000 citra grayscale 28 × 28. Indeks pembagian 80/10/10 diacak dengan "
        f"seed {cfg['seed']} dan disimpan. Run ini memakai {split[0]:,} train, {split[1]:,} validation, "
        f"dan {split[2]:,} test. Batch {cfg['batch_size']}. Pengacakan batch dilakukan pada train."
    )
    table(["Model", "Alur arsitektur", "Adam dan epoch"], [
        ("Vanilla AE", "784 → 256 ReLU → 64 → 256 ReLU → 784 sigmoid", f"lr 0.001 · {cfg['ae_epochs']}"),
        ("Sparse AE", "Vanilla AE + penalti mean |z| pada bottleneck 64", f"lr 0.001 · {cfg['ae_epochs']}"),
        ("VAE", "784 → 256 ReLU → (μ, log σ²) → 64 → 256 ReLU → 784 sigmoid", f"lr 0.001 · {cfg['ae_epochs']}"),
        ("Vanilla GAN", "G 100 → 256 → 512 → 784 tanh. D 784 → 512 → 256 → 1. LeakyReLU.", f"lr 0.0002 · {cfg['gan_epochs']}"),
        ("DCGAN", "G ConvTranspose 7² → 14² → 28² tanh. D Conv 28² → 14² → 7² → 1. BatchNorm pada lapisan tengah.", f"lr 0.0002 · {cfg['gan_epochs']}"),
        ("DDPM", f"U-Net residual {cfg['diffusion_width']} kanal dasar, 28² → 14² → 28², embedding t, GroupNorm + SiLU.", f"lr 0.0005 · {cfg['ddpm_epochs']}"),
        ("MLP", "784 → 256 ReLU → 64 ReLU → 10 logit", f"lr 0.001 · {cfg['mlp_epochs']}"),
        ("Decoder fitur", "64 → 256 ReLU → 784 sigmoid", f"lr 0.001 · {cfg['decoder_epochs']}"),
    ], [0.19, 0.58, 0.23])
    paragraph("Skema aliran tiap model", "h2")
    story.append(architecture_diagrams())
    paragraph("Pada GAN, discriminator juga menerima citra asli pada langkah update D.", "small")
    paragraph("Pemilihan checkpoint", "h2")
    paragraph("AE, VAE, decoder, dan DDPM dipilih dari validation loss. MLP dipilih dari akurasi validation. "
              "GAN dan DCGAN memakai epoch terakhir agar pemilihan tidak menggunakan FID test.")
    paragraph("Skala citra", "h2")
    paragraph("Citra asli berada pada [0, 1]. Discriminator GAN menerima citra asli dan sintetis pada [−1, 1]. "
              "DDPM memakai [−1, 1] saat noising, lalu memetakan sampel kembali ke [0, 1] untuk evaluasi.")

    new_page("2. Fungsi objektif model")
    table(["Model", "Loss per batch"], [
        ("Vanilla AE", "L = mean[(x − x̂)²]."),
        ("Sparse AE", f"L = MSE(x, x̂) + {cfg['sparse_lambda']} · mean(|z|)."),
        ("VAE", "L = mean per citra [Σ BCE(x, x̂) + ½ Σ(μ² + exp(log σ²) − 1 − log σ²)]. "
         "Sampel z = μ + exp(½ log σ²) · ε dengan ε ~ N(0, I)."),
        ("Vanilla GAN dan DCGAN", "L_D = ½[BCEWithLogits(D(x), 1) + BCEWithLogits(D(G(z)), 0)]. "
         "L_G = BCEWithLogits(D(G(z)), 1). z ~ N(0, I)."),
        ("DDPM", "x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε. L = mean[(ε − εθ(x_t, t))²]. "
         "t diambil merata dari langkah 1 sampai T."),
        ("MLP", "L = −mean[log softmax(logit) pada label kelas aktual]."),
        ("Decoder fitur", "L = MSE(x, decoder(encoder_MLP(x))). Gradien hanya memperbarui decoder."),
    ], [0.29, 0.71])
    paragraph("Interpretasi rekonstruksi", "h2")
    paragraph("Prediksi yang meminimalkan MSE untuk z tetap adalah E[x | z]. Bila satu kode mencakup "
              "beberapa posisi kontur yang masuk akal, rata-rata piksel dapat melembutkan batas. BCE "
              "per piksel juga mengestimasi nilai rata-rata probabilitas piksel. Bottleneck dan penekanan "
              "informasi melalui KL ikut memengaruhi tekstur yang bertahan.")
    paragraph("Interpretasi adversarial dan diffusion", "h2")
    paragraph("Jensen–Shannon muncul pada game minimax GAN dengan discriminator optimal. Optimisasi aktual "
              "menggunakan loss generator non-saturating untuk mempertahankan sinyal gradien pada awal training. "
              "DDPM melatih prediksi noise langsung dengan target yang diketahui. Saat sampling ia memakai "
              f"mean posterior untuk setiap langkah dari {cfg['diffusion_steps']} langkah reverse diffusion.")

    new_page("3. Eksperimen MLP Classifier")
    passed = "mencapai" if accuracy >= 0.88 else "belum mencapai"
    paragraph("Q5. Apa yang bisa dipulihkan dari fitur MLP?", "h2")
    paragraph(f"Akurasi MLP sebesar {accuracy:.2%} pada {result['test_count']:,} gambar test. "
              f"Run ini {passed} target 88%.")
    picture("figur_3_mlp.png", "Figur 3. Proyeksi bobot, rekonstruksi fitur beku, dan sampel Gaussian fitur.", 112)
    table(["Metode", "MSE", "PSNR (dB)", "SSIM"], [
        [name, f(metrics["mse"], 6), f(metrics["psnr"], 2), f(metrics["ssim"], 4)]
        for name, metrics in result["mlp_metrics"].items()
    ], [0.38, 0.20, 0.22, 0.20])
    paragraph(
        "Cross-entropy mengoptimasi P(y | x). Detail intra-kelas dapat tersimpan bila berguna bagi classifier, "
        "tetapi objective ini tidak memberi target untuk memulihkan piksel. Proyeksi 784 → 256 mempunyai rank "
        "paling tinggi 256 sehingga WᵀW tidak dapat menjadi identitas pada ruang 784 dimensi. ReLU "
        "memetakan semua nilai negatif ke nol. Kedua sifat ini membatasi rekonstruksi lewat transpose."
    )
    mlp_gap = result["mlp_metrics"]["Frozen decoder"]["mse"] - result["mlp_metrics"]["Vanilla AE"]["mse"]
    paragraph(f"MSE decoder beku berbeda {mlp_gap:+.6f} dari Vanilla AE pada citra test yang sama. "
              "Gunakan selisih ini bersama PSNR, SSIM, dan Figur 3 untuk menilai seberapa banyak struktur yang dipulihkan.")
    paragraph("Decoder beku menguji informasi yang bisa dipulihkan dengan mempelajari pemetaan baru. "
              "Prior Gaussian diagonal pada fitur test menyamakan rata-rata dan varians marginal, "
              "sehingga korelasi antardimensi dan geometri fitur berkelompok belum terwakili.")

    new_page("4. Perbandingan kuantitatif")
    table(["Model", "Laten", "s/epoch", "ms/64", "MSE", "PSNR", "SSIM", "FID"], [
        [name, row["latent_dim"], f(row["train_seconds"], 2), f(row["latency_batch64_ms"], 2),
         f(row["test_mse"], 5), f(row["test_psnr_db"], 2), f(row["test_ssim"], 3), f(row["fid"], 2)]
        for name, row in rows.items()
    ], [0.18, 0.14, 0.11, 0.12, 0.12, 0.11, 0.11, 0.11])
    paragraph(
        f"MSE, PSNR, dan SSIM dihitung per gambar dan dirata-ratakan atas {result['test_count']:,} data test. "
        "PSNR memakai rentang piksel 1. VAE merekonstruksi dengan mean posterior. "
        "Latensi AE mencakup encoder dan decoder, sedangkan VAE memakai jalur sampling prior. "
        f"DDPM mencakup {cfg['diffusion_steps']} langkah."
    )
    paragraph("Stabilitas selama training", "h2")
    table(["Model", "Indikator dari run"], [
        (name, (f"Simpangan baku loss G antar-epoch {f(row['g_loss_std'])}"
                if row["g_loss_std"] is not None else
                f"Train loss {f(row['train_loss_initial'])} → {f(row['train_loss_final'])}"))
        for name, row in rows.items()
    ], [0.28, 0.72])
    paragraph("Besaran loss antarkeluarga memakai reduksi dan objective berbeda. Interpretasi stabilitas "
              "membandingkan bentuk kurva, perubahan intra-model, dan hasil sampel.")
    paragraph("Keragaman kelas hasil generasi", "h2")
    table(["Model", "Kelas prediksi ≥1%", "Jarak distribusi (TV)"], [
        [name, score["classes_at_least_1pct"], f(score["class_tv"], 3)]
        for name, score in mode_rows.items()
    ], [0.34, 0.33, 0.33])
    paragraph(f"Hitungan memakai {result['generated_samples_per_model']:,} citra per model dan label prediksi MLP. "
              "Prediksi kelas citra buram atau rusak dapat keliru. Metrik ini memberi petunjuk kasar tentang coverage.",
              "small")

    new_page("5. Rekonstruksi dan sparsity")
    picture("figur_1_rekonstruksi.png", "Figur 1. Delapan gambar test yang sama direkonstruksi oleh tiga model.", 129)
    mse_gap = rows["Sparse AE"]["test_mse"] - rows["Vanilla AE"]["test_mse"]
    paragraph("Q1. Fidelity rekonstruksi", "h2")
    paragraph(
        f"MSE terendah diperoleh {best_mse}. Selisih MSE Sparse AE terhadap Vanilla AE sebesar {mse_gap:+.6f}. "
        f"PSNR ketiganya masing-masing {f(rows['Vanilla AE']['test_psnr_db'], 2)}, "
        f"{f(rows['Sparse AE']['test_psnr_db'], 2)}, dan {f(rows['VAE']['test_psnr_db'], 2)} dB sesuai urutan figur. "
        "SSIM melengkapi MSE dengan menilai kemiripan struktur lokal."
    )
    paragraph("Q2. Representasi fitur", "h2")
    paragraph(
        f"Proporsi koordinat dengan |z| ≤ 0,05 sebesar {result['sparsity_fraction']['Vanilla AE']:.2%} "
        f"pada Vanilla AE dan {result['sparsity_fraction']['Sparse AE']:.2%} pada Sparse AE. "
        "Penalti L1 menekan magnitudo aktivasi. Perbedaan ukuran ini dapat dibaca bersama MSE dan SSIM "
        "untuk melihat ongkos representasi yang lebih jarang pada fidelity."
    )
    paragraph("Dampak bottleneck terhadap kontur bergantung pada detail yang perlu dipertahankan. Saat dua "
              "variasi spasial terpetakan ke z yang serupa, MSE mendorong prediksi di antara keduanya. Rekonstruksi "
              "VAE juga mengikuti prior melalui KL, sehingga kapasitas untuk detail spesifik perlu dibagi dengan "
              "keteraturan laten.")

    new_page("6. Kualitas generasi")
    picture("figur_2_sampel.png", "Figur 2. Enam belas sampel prior acak per model.", 142)
    paragraph("Q1. FID pada sampel baru", "h2")
    paragraph(fid_text + f" Perbandingan memakai {result['generated_samples_per_model']:,} sampel per model. "
              "Perhitungan FID memakai jumlah gambar asli dan sintetis yang sama. Kanal grayscale disalin "
              "ke tiga kanal untuk ekstraktor Inception.")
    paragraph("Q2. Ketajaman dan keragaman", "h2")
    edges = ", ".join(f"{name} {f(item['mean_edge_strength'], 3)}" for name, item in mode_rows.items())
    paragraph(f"Rata-rata gradien absolut piksel citra test sebesar {f(result['real_edge_strength'], 3)}. "
              f"Pada sampel prior, nilainya {edges}. Noise juga dapat meningkatkan gradien, sehingga "
              "angka ini dibaca bersama kontur objek pada Figur 2.")
    paragraph("Discriminator memberi sinyal berdasarkan bentuk citra. Sinyal tersebut dapat mendorong kontur "
              "yang tajam, tetapi ketajaman tidak mengukur berapa banyak jenis pakaian yang dicakup. "
              "Konvolusi DCGAN memanfaatkan kedekatan piksel. Perbedaan coverage dibaca dari Figur 2 serta "
              "tabel kelas prediksi dan TV.")

    new_page("7. Interpolasi ruang laten")
    picture("figur_4_interpolasi.png", "Figur 4. Dekode enam titik dari α = 0 sampai α = 1.", 141)
    paragraph("Q4. Pemilihan titik dan perubahan citra", "h2")
    for item in result["interpolation_endpoints"]:
        paragraph(item, "small")
    paragraph("Interpolasi memakai z(α) = (1 − α)z_A + αz_B. AE memakai hasil encoder, VAE memakai mean "
              "posterior, dan GAN memakai noise masukan. Titik GAN dipilih dari prediksi kelas pada 256 "
              "sampel kandidat. Bila hanya tersedia satu kelas prediksi, perbandingan memakai dua noise "
              "yang paling jauh dari 64 kandidat.")
    jumps = result["interpolation_max_adjacent_mse"]
    transitions = ", ".join(f"{name} {f(value, 4)}" for name, value in jumps.items())
    paragraph(f"MSE tertinggi antara dua langkah bersebelahan adalah {transitions}. "
              "Ukuran ini mengukur lompatan piksel terbesar pada tiap lintasan. Interpretasinya juga bergantung pada titik awal dan akhir.")
    paragraph("Pada AE deterministik, objective rekonstruksi tidak membatasi kepadatan antara dua kode citra. "
              "Interpolasi dapat melintasi kode yang jarang muncul saat training. KL VAE mendorong "
              "posterior lebih dekat ke prior, sehingga titik di antaranya lebih berpeluang berada di daerah "
              "yang dikenal decoder. Satu garis interpolasi memberi bukti lokal pada pasangan terpilih.")

    new_page("8. Stabilitas dan Generative Trilemma")
    picture("figur_5_loss_dan_denoise.png", "Figur 5. Kurva loss dan lintasan DDPM dari x_T ke x_0.", 140)
    paragraph("Q3. Kualitas, kecepatan, dan cakupan", "h2")
    paragraph(f"Sampler satu langkah tercepat pada batch 64 adalah {fastest} "
              f"({f(rows[fastest]['latency_batch64_ms'], 2)} ms). DDPM membutuhkan {f(slow, 2)} ms "
              f"untuk {cfg['diffusion_steps']} langkah. Kualitas dibandingkan lewat FID, sedangkan "
              "cakupan dinilai dari kemunculan kelas dan inspeksi Figur 2.")
    paragraph("GAN menghadapi objective yang berubah karena generator dan discriminator dioptimasi bersamaan. "
              "Discriminator terlalu kuat dapat melemahkan sinyal generator, dan generator dapat terpusat "
              "pada sedikit mode yang efektif. Arsitektur lokal dan BatchNorm DCGAN membantu aliran fitur. "
              "DDPM memakai target noise yang tersedia, sehingga loss tidak bergantung pada lawan yang ikut belajar.")

    new_page("9. Rekomendasi industri")
    best_anomaly = max(anomaly, key=anomaly.get)
    one = result["latency_batch1_ms"]
    mobile = min(one, key=one.get)
    visual_fid = [name for name in ("DCGAN", "DDPM") if rows[name]["fid"] is not None]
    paragraph("Q6. Inspeksi manufaktur", "h2")
    paragraph(f"Pada uji anomali sintetis yang membalik piksel kotak tengah 8 × 8, AUROC Vanilla AE "
              f"{f(anomaly['Vanilla AE'], 3)} dan Sparse AE {f(anomaly['Sparse AE'], 3)}. "
              f"{best_anomaly} adalah kandidat awal berdasarkan proxy ini. Uji lanjutan perlu citra cacat produksi "
              "karena ukuran, tekstur, dan posisi kecacatan nyata dapat berbeda.")
    paragraph("Filter avatar interaktif", "h2")
    table(["Kandidat", "Latensi batch 1", "Latensi batch 64"], [
        [name, f(value, 2) + " ms", f(rows[name]["latency_batch64_ms"], 2) + " ms"]
        for name, value in one.items()
    ], [0.35, 0.32, 0.33])
    paragraph(f"{mobile} punya latensi batch 1 terendah pada perangkat pengujian. Batch 1 lebih dekat ke "
              "satu permintaan interaktif. Syarat 30 ms perlu diuji langsung di ponsel, termasuk "
              "preprocessing, pemindahan data, dan variasi waktu antarpanggilan.")
    paragraph("Pembuatan aset visual", "h2")
    if visual_fid:
        visual = min(visual_fid, key=lambda name: rows[name]["fid"])
        paragraph(f"Di antara DCGAN dan DDPM, {visual} mencatat FID terendah pada Fashion-MNIST. "
                  f"DDPM memakai {f(slow, 2)} ms per batch 64, sementara DCGAN memakai "
                  f"{f(rows['DCGAN']['latency_batch64_ms'], 2)} ms. Pilih kandidat kualitas untuk uji resolusi "
                  "produksi dan kandidat cepat untuk membuat banyak variasi.")
    else:
        paragraph("Perbandingan kualitas DCGAN dan DDPM memerlukan FID dan inspeksi hasil sampel. "
                  "Latensi mengukur biaya tiap pilihan, sedangkan kualitas aset resolusi tinggi "
                  "perlu diuji pada dataset dan resolusi target.")
    paragraph("Fashion-MNIST memakai pakaian grayscale 28 × 28. Ekstrapolasi ke aset RGB atau kamera inspeksi "
              "memerlukan evaluasi pada domain target.")

    new_page("10. Reproduksi dan rujukan")
    table(["Komponen", "Catatan run"], [
        ["Perangkat training/evaluasi", f"{meta['device']} · {meta['hardware']}"],
        ["Versi PyTorch", meta["torch"]],
        ["Seed dan split", f"{cfg['seed']} · indeks di split_indices.npz"],
        ["FID", f"Inception {cfg['fid_feature']} · {result['generated_samples_per_model']:,} sampel per model"
         if result["fid_enabled"] else "Belum dihitung pada run ini"],
        ["Latensi", f"Median {cfg['latency_repeats']} pengukuran setelah 5 warm-up, tanpa transfer data awal"],
        ["Artefak", "metrics.csv, class_coverage.csv, results.json, riwayat epoch, dan lima figur PNG"],
    ], [0.31, 0.69])
    paragraph("Batas analisis", "h2")
    paragraph("Hasil satu seed mengukur satu realisasi training. Pemeriksaan stabilitas yang lebih kuat "
              "dapat memakai beberapa seed dan melaporkan sebaran metrik. FID menggunakan fitur Inception "
              "yang dilatih pada citra alami, sehingga nilainya perlu dibaca bersama contoh grayscale dan "
              "indikator coverage.")
    paragraph("Rujukan", "h2")
    references = [
        "Xiao, Rasul, & Vollgraf (2017). Fashion-MNIST. arXiv:1708.07747.",
        "Goodfellow et al. (2014). Generative Adversarial Nets. NeurIPS 27.",
        "Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.",
        "Radford, Metz, & Chintala (2016). Unsupervised Representation Learning with Deep Convolutional GANs. ICLR.",
        "Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. NeurIPS 30.",
        "Ho, Jain, & Abbeel (2020). Denoising Diffusion Probabilistic Models. NeurIPS 33.",
        "Nichol & Dhariwal (2021). Improved Denoising Diffusion Probabilistic Models. ICML.",
        "Wang, Bovik, Sheikh, & Simoncelli (2004). Image Quality Assessment. IEEE Transactions on Image Processing.",
    ]
    for reference in references:
        paragraph(reference, "small")

    name = slug(identity["name"])
    nim = slug(identity["nim"])
    filename = f"TUGAS_DL_SESI07_{nim}_{name}.pdf" if nim and name else "laporan_draf.pdf"
    if smoke:
        filename = "laporan_uji_smoke.pdf"
    output = Path(folder) / filename
    doc = SimpleDocTemplate(str(output), pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
                            rightMargin=19 * mm, leftMargin=19 * mm,
                            topMargin=18 * mm, bottomMargin=19 * mm,
                            title="Studi Komparatif Empiris Model Generatif", author=identity["name"] or None)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Laporan disimpan di {output}", flush=True)
    return output
