from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata
from skimage.metrics import structural_similarity
from torch.nn import functional as F

from experiment import MODEL_NAMES, loader, reload_models, save_json, write_csv


def metric_values(originals, predictions):
    originals = originals[:, 0].cpu().numpy()
    predictions = predictions[:, 0].float().cpu().numpy()
    mse = np.mean((originals - predictions) ** 2, axis=(1, 2), dtype=np.float64)
    psnr = -10 * np.log10(np.maximum(mse, 1e-12))
    ssim = [
        structural_similarity(original, prediction, data_range=1.0)
        for original, prediction in zip(originals, predictions)
    ]
    return np.stack((mse, psnr, ssim), axis=1)


@torch.inference_mode()
def scores(reconstruct, batches, device):
    parts = []
    for images, _ in batches:
        parts.append(metric_values(images, reconstruct(images.to(device))))
    return dict(zip(("mse", "psnr", "ssim"), np.concatenate(parts).mean(axis=0).tolist()))


@torch.inference_mode()
def latency_ms(run, device, repeats, warmup=5):
    for _ in range(warmup):
        run()
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((perf_counter() - start) * 1000)
    return float(np.median(times))


def generators(models, cfg, device):
    vae, gan, dcgan, diffusion = (models[name] for name in ("VAE", "Vanilla GAN", "DCGAN", "DDPM"))

    def noise(size, rng):
        return torch.randn(size, device=device, generator=rng)

    def sample_vae(count, rng):
        return vae.decoder(noise((count, cfg.latent_dim), rng))

    def sample_gan(count, rng):
        return (gan(noise((count, cfg.gan_latent), rng)) + 1) / 2

    def sample_dcgan(count, rng):
        return (dcgan(noise((count, cfg.gan_latent), rng)) + 1) / 2

    def sample_ddpm(count, rng):
        return diffusion.sample(noise((count, 1, 28, 28), rng), rng)[0]

    return {"VAE": sample_vae, "Vanilla GAN": sample_gan,
            "DCGAN": sample_dcgan, "DDPM": sample_ddpm}


def eval_rng(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


@torch.inference_mode()
def anomaly_auc(reconstruct, batches, device):
    normal_errors, patch_errors = [], []
    for images, _ in batches:
        images = images.to(device)
        patched = images.clone()
        # Balik nilai piksel di kotak tengah agar tesnya deterministik.
        patched[:, :, 10:18, 10:18] = 1 - patched[:, :, 10:18, 10:18]
        normal_errors.extend((images - reconstruct(images)).square().flatten(1).mean(1).cpu().tolist())
        patch_errors.extend((patched - reconstruct(patched)).square().flatten(1).mean(1).cpu().tolist())
    count = len(patch_errors)
    ranks = rankdata(normal_errors + patch_errors)
    return float((ranks[count:].sum() - count * (count + 1) / 2) / count**2)


def mean_edge_strength(images):
    horizontal = (images[..., 1:] - images[..., :-1]).abs().flatten(1).mean(1)
    vertical = (images[..., 1:, :] - images[..., :-1, :]).abs().flatten(1).mean(1)
    return (horizontal + vertical) / 2


def rgb(images):
    return images.float().clamp(0, 1).repeat(1, 3, 1, 1)


@torch.inference_mode()
def sample_comparison(samplers, classifier, test_set, cfg, device, use_fid):
    from torch.utils.data import DataLoader, Subset

    requested = cfg.fid_samples if use_fid else min(cfg.fid_samples, 256)
    count = min(requested, len(test_set))
    fid = None
    if use_fid:
        from torchmetrics.image.fid import FrechetInceptionDistance

        fid = FrechetInceptionDistance(
            feature=cfg.fid_feature, normalize=True, reset_real_features=False,
        ).to(device)

    real_counts = torch.zeros(10, dtype=torch.long)
    real_edge_sum = 0.0
    real_batches = DataLoader(Subset(test_set, range(count)), batch_size=32)
    for images, labels in real_batches:
        real_counts += labels.bincount(minlength=10)
        real_edge_sum += mean_edge_strength(images).sum().item()
        if fid is not None:
            fid.update(rgb(images.to(device)), real=True)

    previews, stats = {}, {}
    for i, (name, sample) in enumerate(samplers.items()):
        rng = eval_rng(device, cfg.seed + 50 + i)
        histogram = torch.zeros(10, dtype=torch.long)
        preview_chunks = []
        fake_edge_sum = 0.0
        remaining_preview = 16
        for start in range(0, count, 32):
            images = sample(min(32, count - start), rng)
            histogram += classifier(images).argmax(1).cpu().bincount(minlength=10)
            fake_edge_sum += mean_edge_strength(images).sum().item()
            if remaining_preview:
                preview = images[:remaining_preview].cpu()
                preview_chunks.append(preview)
                remaining_preview -= len(preview)
            if fid is not None:
                fid.update(rgb(images), real=False)

        fid_score = None if fid is None else fid.compute().item()
        if fid is not None:
            fid.reset()

        prob = histogram.numpy() / count
        real_prob = real_counts.numpy() / count
        stats[name] = {
            "fid": fid_score,
            "classes_at_least_1pct": int((prob >= 0.01).sum()),
            "class_tv": float(np.abs(prob - real_prob).sum() / 2),
            "predicted_classes": histogram.tolist(),
            "mean_edge_strength": fake_edge_sum / count,
        }
        previews[name] = torch.cat(preview_chunks)
        print(f"{name} | {count} sampel | FID {fid_score}", flush=True)

    return previews, stats, real_counts.tolist(), count, real_edge_sum / count


def sample_examples(test_set, device):
    examples = {}
    for image, label in test_set:
        examples.setdefault(label, image)
        if len(examples) == 10:
            break
    labels = sorted(examples)
    return torch.stack([examples[label] for label in labels]).to(device), labels


def interpolation(a, b):
    alpha = torch.linspace(0, 1, 6, device=a.device)[:, None]
    return torch.lerp(a[None], b[None], alpha)


@torch.inference_mode()
def gan_endpoints(name, generator, classifier, cfg, device, class_pair, class_names, seed):
    rng = eval_rng(device, seed)
    points = torch.randn(256, cfg.gan_latent, device=device, generator=rng)
    images = (generator(points) + 1) / 2
    confidence, classes = classifier(images).softmax(1).max(1)
    available = classes.unique().tolist()
    if all(label in available for label in class_pair):
        selected = class_pair
    else:
        selected = classes.bincount(minlength=10).argsort(descending=True)[:min(2, len(available))].tolist()

    if len(selected) == 2:
        endpoints = [
            points[confidence.masked_fill(classes != label, -1).argmax()] for label in selected
        ]
        note = f"{name}: {class_names[selected[0]]} → {class_names[selected[1]]}"
    else:
        distance = torch.cdist(points[:64], points[:64])
        pair = int(distance.argmax())
        endpoints = [points[pair // 64], points[pair % 64]]
        note = f"{name}: satu kelas prediksi ({class_names[selected[0]]}), dua noise terjauh dari 64 kandidat"
    return (generator(interpolation(*endpoints)) + 1) / 2, note


def draw_image(ax, image, limits=(0, 1)):
    ax.imshow(image.detach().cpu().squeeze(0), cmap="gray", vmin=limits[0], vmax=limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig, file):
    fig.savefig(file, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def row_figure(rows, file, headings=None):
    columns = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), columns, figsize=(columns * 1.1 + 1, len(rows) * 1.15),
                             squeeze=False, layout="constrained")
    for row_index, (name, images) in enumerate(rows):
        for col, image in enumerate(images):
            draw_image(axes[row_index, col], image)
            if row_index == 0 and headings:
                axes[row_index, col].set_title(headings[col], fontsize=8)
        axes[row_index, 0].set_ylabel(name, rotation=0, labelpad=8, ha="right", va="center",
                                      fontsize=9)
    save_figure(fig, file)


def sampling_figure(previews, file):
    fig = plt.figure(figsize=(8, 8), layout="constrained")
    panels = fig.subfigures(2, 2).ravel()
    for panel, (name, images) in zip(panels, previews.items()):
        panel.suptitle(name, fontsize=14)
        for ax, image in zip(panel.subplots(4, 4).ravel(), images):
            draw_image(ax, image)
    save_figure(fig, file)


def progress_figure(histories, trajectory, file):
    fig = plt.figure(figsize=(10, 8.1), layout="constrained")
    layout = fig.add_gridspec(4, 2, height_ratios=(1.0, 1.0, 1.0, 0.66))
    for i, (task, name) in enumerate(zip(
        ("vanilla_ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"), MODEL_NAMES,
    )):
        ax = fig.add_subplot(layout[i // 2, i % 2])
        records = histories[task]
        epochs = [row["epoch"] for row in records]
        series = {"train_loss": "Train", "val_loss": "Validation"}
        if task in {"gan", "dcgan"}:
            series = {"train_g_loss": "Generator", "train_d_loss": "Discriminator"}
        for key, label in series.items():
            ax.plot(epochs, [row[key] for row in records], marker=".", label=label)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    strip = layout[3, :].subgridspec(1, len(trajectory))
    for i, (step, images) in enumerate(trajectory):
        ax = fig.add_subplot(strip[0, i])
        draw_image(ax, images[0], (-1, 1))
        ax.set_title(f"x_{step}", fontsize=9)
    save_figure(fig, file)


@torch.inference_mode()
def run_evaluation(cfg, splits, histories, folder, device, use_fid):
    _, _, test_set = splits
    test_batches = loader(test_set, cfg, device)
    class_names = test_set.dataset.datasets[0].classes
    models = {
        name: reload_models(task, cfg, folder, device)[role]
        for name, task, role in [
            ("Vanilla AE", "vanilla_ae", "model"), ("Sparse AE", "sparse_ae", "model"),
            ("VAE", "vae", "model"), ("Vanilla GAN", "gan", "G"),
            ("DCGAN", "dcgan", "G"), ("DDPM", "ddpm", "model"),
            ("classifier", "mlp", "model"), ("decoder", "frozen_decoder", "model"),
        ]
    }
    classifier = models["classifier"]
    vanilla, sparse, vae = (models[name] for name in MODEL_NAMES[:3])

    def reconstruct_vae(images):
        return vae.decoder(vae.encode(images)[0])

    def reconstruct_mlp(images):
        return models["decoder"](classifier.encode(images))

    reconstructions = {
        "Vanilla AE": lambda images: vanilla(images)[0],
        "Sparse AE": lambda images: sparse(images)[0],
        "VAE": reconstruct_vae,
        "Frozen decoder": reconstruct_mlp,
        "Transpose fitur": lambda images: classifier.project_features(classifier.encode(images)),
    }
    recon_scores = {
        name: scores(reconstruct, test_batches, device)
        for name, reconstruct in reconstructions.items()
    }
    print("Evaluasi rekonstruksi selesai.", flush=True)

    correct, feature_batches, sparse_zero, vanilla_zero, seen = 0, [], 0, 0, 0
    for images, labels in test_batches:
        images, labels = images.to(device), labels.to(device)
        features = classifier.encode(images)
        feature_batches.append(features.cpu())
        correct += (classifier.fc3(features).argmax(1) == labels).sum().item()
        sparse_zero += (sparse.encoder(images).abs() <= 0.05).sum().item()
        vanilla_zero += (vanilla.encoder(images).abs() <= 0.05).sum().item()
        seen += features.numel()

    all_features = torch.cat(feature_batches).to(device)
    rng = eval_rng(device, cfg.seed + 90)
    sampled_features = (
        all_features.mean(0)
        + torch.randn(10, cfg.latent_dim, device=device, generator=rng)
        * all_features.std(0, unbiased=False)
    )
    test_images, _ = next(iter(test_batches))
    latency_images = test_images[:64].to(device)
    preview = test_images[:8].to(device)
    row_figure(
        [("Asli", preview)] + [(name, reconstructions[name](preview)) for name in MODEL_NAMES[:3]],
        folder / "figur_1_rekonstruksi.png",
    )

    chosen, labels = sample_examples(test_set, device)
    one_hot = F.one_hot(torch.tensor(labels, device=device), 10).float()
    features = classifier.encode(chosen)
    mlp_rows = [
        ("Asli", chosen),
        ("Transpose kelas", classifier.project_classes(one_hot)),
        ("Transpose fitur", classifier.project_features(features)),
        ("Frozen decoder", models["decoder"](features)),
        ("Vanilla AE", reconstructions["Vanilla AE"](chosen)),
        ("Gaussian fitur", models["decoder"](sampled_features)[:len(chosen)]),
    ]
    row_figure(mlp_rows, folder / "figur_3_mlp.png", [class_names[i] for i in labels])

    class_pair = (0, 9) if 0 in labels and 9 in labels else (labels[0], labels[-1])
    endpoints = chosen[[labels.index(label) for label in class_pair]]
    line_rows, endpoint_notes = [], [
        f"AE dan VAE: {class_names[class_pair[0]]} → {class_names[class_pair[1]]}"
    ]
    for name, model in zip(MODEL_NAMES[:3], (vanilla, sparse, vae)):
        zs = model.encode(endpoints)[0] if name == "VAE" else model.encoder(endpoints)
        line_rows.append((name, model.decoder(interpolation(*zs))))
    for i, name in enumerate(("Vanilla GAN", "DCGAN")):
        images, note = gan_endpoints(name, models[name], classifier, cfg, device,
                                     class_pair, class_names, cfg.seed + i + 120)
        line_rows.append((name, images))
        endpoint_notes.append(note)
    row_figure(line_rows, folder / "figur_4_interpolasi.png", [f"α={a/5:.1f}" for a in range(6)])

    rng = eval_rng(device, cfg.seed + 92)
    noise = torch.randn(1, 1, 28, 28, device=device, generator=rng)
    _, trajectory = models["DDPM"].sample(noise, rng, keep_trajectory=True)
    progress_figure(histories, trajectory, folder / "figur_5_loss_dan_denoise.png")

    samplers = generators(models, cfg, device)
    generated, gen_stats, real_classes, sampled_count, real_edge_strength = sample_comparison(
        samplers, classifier, test_set, cfg, device, use_fid,
    )
    sampling_figure(generated, folder / "figur_2_sampel.png")

    timing = {
        name: (lambda x=latency_images, fn=reconstructions[name]: fn(x))
        for name in MODEL_NAMES[:2]
    }
    rng = eval_rng(device, cfg.seed + 95)
    vae_noise = torch.randn(64, cfg.latent_dim, device=device, generator=rng)
    gan_noise = torch.randn(64, cfg.gan_latent, device=device, generator=rng)
    initial_noise = torch.randn(64, 1, 28, 28, device=device, generator=rng)
    timing.update({
        "VAE": lambda: vae.decoder(vae_noise),
        "Vanilla GAN": lambda: models["Vanilla GAN"](gan_noise),
        "DCGAN": lambda: models["DCGAN"](gan_noise),
        "DDPM": lambda: models["DDPM"].sample(initial_noise, rng)[0],
    })
    small_batch = {
        "VAE": lambda: vae.decoder(vae_noise[:1]),
        "Vanilla GAN": lambda: models["Vanilla GAN"](gan_noise[:1]),
        "DCGAN": lambda: models["DCGAN"](gan_noise[:1]),
    }
    table = []
    for i, (name, task) in enumerate(zip(MODEL_NAMES, ("vanilla_ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"))):
        reconstruction = recon_scores.get(name, {})
        score = gen_stats.get(name, {})
        latent_dim = cfg.latent_dim if i < 3 else (cfg.gan_latent if i < 5 else "1×28×28")
        table.append({
            "model": name, "latent_dim": latent_dim,
            "train_seconds": float(np.mean([row["seconds"] for row in histories[task]])),
            "latency_batch64_ms": latency_ms(timing[name], device, cfg.latency_repeats),
            "test_mse": reconstruction.get("mse"), "test_psnr_db": reconstruction.get("psnr"),
            "test_ssim": reconstruction.get("ssim"), "fid": score.get("fid"),
            "predicted_modes_ge_1pct": score.get("classes_at_least_1pct"),
            "g_loss_std": float(np.std([row["train_g_loss"] for row in histories[task]])) if i in (3, 4) else None,
            "train_loss_initial": histories[task][0].get("train_loss"),
            "train_loss_final": histories[task][-1].get("train_loss"),
        })

    results = {
        "profile": cfg.profile, "test_count": len(test_set), "fid_enabled": use_fid,
        "generated_samples_per_model": sampled_count, "metric_table": table,
        "mlp_accuracy": correct / len(test_set), "mlp_metrics": {
            name: recon_scores[name] for name in ["Vanilla AE", "Frozen decoder", "Transpose fitur"]
        }, "sparsity_fraction": {"Vanilla AE": vanilla_zero / seen, "Sparse AE": sparse_zero / seen},
        "class_counts_real": real_classes, "generation": gen_stats,
        "real_edge_strength": real_edge_strength,
        "interpolation_endpoints": endpoint_notes,
        "interpolation_max_adjacent_mse": {
            name: (images[1:] - images[:-1]).square().flatten(1).mean(1).max().item()
            for name, images in line_rows
        },
        "anomaly_auc": {name: anomaly_auc(reconstructions[name], test_batches, device)
                        for name in MODEL_NAMES[:2]},
        "latency_batch1_ms": {name: latency_ms(fn, device, cfg.latency_repeats)
                              for name, fn in small_batch.items()},
    }
    save_json(folder / "results.json", results)
    write_csv(folder / "metrics.csv", table)
    coverage_rows = [
        {"kelas": label, "real": real_classes[i],
         **{name: row["predicted_classes"][i] for name, row in gen_stats.items()}}
        for i, label in enumerate(class_names)
    ]
    write_csv(folder / "class_coverage.csv", coverage_rows)
    print(f"Evaluasi selesai. Akurasi MLP {results['mlp_accuracy']:.2%}.", flush=True)
    return results
