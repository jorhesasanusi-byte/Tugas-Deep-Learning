import csv
import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms

from models import (
    Classifier, DCDiscriminator, DCGenerator, Diffusion, FrozenDecoder,
    GANDiscriminator, GANGenerator, SparseAE, VAE, VanillaAE,
)


MODEL_NAMES = ["Vanilla AE", "Sparse AE", "VAE", "Vanilla GAN", "DCGAN", "DDPM"]
TASKS = ["vanilla_ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm", "mlp", "frozen_decoder"]


@dataclass(frozen=True)
class Config:
    profile: str = "full"
    seed: int = 42
    batch_size: int = 128
    latent_dim: int = 64
    gan_latent: int = 100
    sparse_lambda: float = 0.02
    ae_epochs: int = 15
    gan_epochs: int = 25
    ddpm_epochs: int = 20
    mlp_epochs: int = 32
    decoder_epochs: int = 15
    diffusion_steps: int = 100
    diffusion_width: int = 24
    fid_samples: int = 3000
    fid_feature: int = 2048
    latency_repeats: int = 20
    workers: int = 0
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None

    def epochs(self, task):
        return {
            "gan": self.gan_epochs, "dcgan": self.gan_epochs,
            "ddpm": self.ddpm_epochs, "mlp": self.mlp_epochs,
            "frozen_decoder": self.decoder_epochs,
        }.get(task, self.ae_epochs)


def config_for(profile, workers=0):
    if profile == "full":
        return Config(workers=workers)
    if profile == "smoke":
        return Config(
            profile="smoke", ae_epochs=1, gan_epochs=1, ddpm_epochs=1,
            mlp_epochs=1, decoder_epochs=1, diffusion_steps=8, diffusion_width=8,
            fid_samples=16, latent_dim=64, latency_repeats=3,
            train_limit=384, val_limit=160, test_limit=160,
        )
    raise ValueError(f"Profil tidak dikenal: {profile}")


class FashionMNIST(datasets.FashionMNIST):
    mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        *datasets.FashionMNIST.mirrors,
    ]


def load_splits(root, cfg, folder):
    root = str(root)
    transform = transforms.ToTensor()
    train = FashionMNIST(root, train=True, download=True, transform=transform)
    test = FashionMNIST(root, train=False, download=True, transform=transform)
    dataset = ConcatDataset([train, test])
    parts = random_split(
        dataset, [56_000, 7000, 7000],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    limits = [cfg.train_limit, cfg.val_limit, cfg.test_limit]
    indices = [part.indices[:limit] for part, limit in zip(parts, limits)]
    np.savez_compressed(folder / "split_indices.npz",
                        train=indices[0], val=indices[1], test=indices[2])
    return tuple(Subset(dataset, idx) for idx in indices)


def loader(dataset, cfg, device, seed=None):
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=seed is not None,
        generator=None if seed is None else torch.Generator().manual_seed(seed),
        num_workers=cfg.workers, pin_memory=device.type == "cuda",
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def prepare_run(folder, cfg, device):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "checkpoints").mkdir(exist_ok=True)
    settings = asdict(cfg)
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    path = folder / "run_config.json"
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored["digest"] != digest:
            raise ValueError("Konfigurasi folder hasil berubah. Gunakan folder output lain.")
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    if path.is_file() and stored["device"] != str(device):
        raise ValueError("Device berbeda dari run sebelumnya. Gunakan folder output lain.")
    save_json(path, {
        "digest": digest, "settings": settings, "device": str(device),
        "hardware": hardware or platform.machine(), "torch": str(torch.__version__),
    })


def build_models(task, cfg):
    if task == "gan":
        return {"G": GANGenerator(cfg.gan_latent), "D": GANDiscriminator()}
    if task == "dcgan":
        return {"G": DCGenerator(cfg.gan_latent), "D": DCDiscriminator()}
    if task == "ddpm":
        return {"model": Diffusion(cfg.diffusion_steps, cfg.diffusion_width)}
    classes = {
        "vanilla_ae": VanillaAE, "sparse_ae": SparseAE, "vae": VAE,
        "mlp": Classifier, "frozen_decoder": FrozenDecoder,
    }
    return {"model": classes[task](cfg.latent_dim)}


def reload_models(task, cfg, folder, device):
    models = {name: model.to(device).eval() for name, model in build_models(task, cfg).items()}
    weights = torch.load(folder / "checkpoints" / f"{task}_best.pt",
                         map_location=device, weights_only=True)
    for name, model in models.items():
        model.load_state_dict(weights[name])
    return models


def named_states(models):
    return {name: model.state_dict() for name, model in models.items()}


def build_optimizers(task, models):
    if task in {"gan", "dcgan"}:
        return {
            name: torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
            for name, model in models.items()
        }
    lr = 0.0005 if task == "ddpm" else 0.001
    return {"model": torch.optim.Adam(models["model"].parameters(), lr=lr)}


def vae_losses(images, reconstructed, mean, logvar):
    bce = F.binary_cross_entropy(reconstructed, images, reduction="none").flatten(1).sum(1)
    kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(1)
    return {"loss": (bce + kl).mean(), "bce": bce.mean(), "kl": kl.mean()}


def batch_losses(task, model, images, labels, cfg, classifier=None, validation_rng=None):
    if task == "mlp":
        logits = model(images)
        return {"loss": F.cross_entropy(logits, labels),
                "accuracy": (logits.argmax(1) == labels).float().mean()}
    if task == "frozen_decoder":
        with torch.no_grad():
            features = classifier.encode(images)
        return {"loss": F.mse_loss(model(features), images)}
    if task == "ddpm":
        return {"loss": model.training_loss(images, validation_rng)}
    if task == "vae":
        if validation_rng is None:
            reconstructed, mean, logvar = model(images)
        else:
            mean, logvar = model.encode(images)
            noise = torch.randn(mean.shape, device=images.device, generator=validation_rng)
            reconstructed = model.decoder(mean + (0.5 * logvar).exp() * noise)
        return vae_losses(images, reconstructed, mean, logvar)

    reconstructed, latent = model(images)
    mse = F.mse_loss(reconstructed, images)
    result = {"loss": mse, "mse": mse, "near_zero": (latent.abs() <= 0.05).float().mean()}
    if task == "sparse_ae":
        penalty = latent.abs().mean()
        result.update(loss=mse + cfg.sparse_lambda * penalty, latent_l1=penalty)
    return result


def gan_step(models, optimizers, images, latent_dim):
    g, d = models["G"], models["D"]
    optimizers["D"].zero_grad(set_to_none=True)
    fake = g(torch.randn(len(images), latent_dim, device=images.device))
    real_logits, fake_logits = d(images * 2 - 1), d(fake.detach())
    d_loss = (
        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    ) / 2
    d_loss.backward()
    optimizers["D"].step()

    d.requires_grad_(False)
    optimizers["G"].zero_grad(set_to_none=True)
    logits = d(fake)
    g_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    g_loss.backward()
    optimizers["G"].step()
    d.requires_grad_(True)
    return {"g_loss": g_loss, "d_loss": d_loss}


def run_epoch(task, models, batches, cfg, device, optimizers=None, classifier=None, scaler=None):
    training = optimizers is not None
    for model in models.values():
        model.train(training)
    sums = {}
    seen = 0
    val_rng = None if training else torch.Generator(device=device).manual_seed(cfg.seed + 991)

    with torch.set_grad_enabled(training):
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            if task in {"gan", "dcgan"}:
                losses = gan_step(models, optimizers, images, cfg.gan_latent)
            else:
                model = models["model"]
                with torch.autocast(device_type=device.type, enabled=scaler is not None):
                    losses = batch_losses(task, model, images, labels, cfg, classifier, val_rng)
                if training:
                    optimizer = optimizers["model"]
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is None:
                        losses["loss"].backward()
                        optimizer.step()
                    else:
                        scaler.scale(losses["loss"]).backward()
                        scaler.step(optimizer)
                        scaler.update()

            seen += len(images)
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + value.item() * len(images)

    if not all(np.isfinite(value) for value in sums.values()):
        raise FloatingPointError(f"Loss non-finite pada {task}.")
    return {name: value / seen for name, value in sums.items()}


def train_all(cfg, splits, folder, device):
    train_set, val_set, _ = splits
    val_batches = loader(val_set, cfg, device)
    classifier = None
    histories = {}

    for i, task in enumerate(TASKS):
        models = {name: model.to(device) for name, model in build_seeded(task, cfg, i).items()}
        optimizers = build_optimizers(task, models)
        scaler = torch.amp.GradScaler("cuda") if task == "ddpm" and device.type == "cuda" else None
        last_path = folder / "checkpoints" / f"{task}_last.pt"
        best_path = folder / "checkpoints" / f"{task}_best.pt"
        history = []
        best = float("inf")
        if last_path.is_file():
            state = torch.load(last_path, map_location=device, weights_only=True)
            for name, model in models.items():
                model.load_state_dict(state["models"][name])
                optimizers[name].load_state_dict(state["optimizers"][name])
            history, best = state["history"], state["best"]
            if scaler is not None and state.get("scaler") is not None:
                scaler.load_state_dict(state["scaler"])

        num_epochs = cfg.epochs(task)
        for epoch in range(len(history) + 1, num_epochs + 1):
            epoch_seed = cfg.seed + i * 1000 + epoch
            seed_all(epoch_seed)
            batches = loader(train_set, cfg, device, seed=epoch_seed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            train_stats = run_epoch(task, models, batches, cfg, device, optimizers, classifier, scaler)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record = {"epoch": epoch, "seconds": perf_counter() - start}
            record.update({f"train_{name}": value for name, value in train_stats.items()})
            if task in {"gan", "dcgan"}:
                save_checkpoint(best_path, named_states(models))
            else:
                val_stats = run_epoch(task, models, val_batches, cfg, device, classifier=classifier)
                record.update({f"val_{name}": value for name, value in val_stats.items()})
                score = -val_stats["accuracy"] if task == "mlp" else val_stats["loss"]
                if score < best:
                    best = score
                    save_checkpoint(best_path, named_states(models))

            history.append(record)
            save_checkpoint(last_path, {
                "models": named_states(models), "optimizers": named_states(optimizers),
                "history": history, "best": best,
                "scaler": None if scaler is None else scaler.state_dict(),
            })
            summary = " | ".join(f"{name} {value:.4f}" for name, value in record.items() if name != "epoch")
            print(f"{task} {epoch}/{num_epochs} | {summary}", flush=True)

        write_csv(folder / f"history_{task}.csv", history)
        histories[task] = history
        if task == "mlp":
            classifier = reload_models(task, cfg, folder, device)["model"]
            classifier.requires_grad_(False)

    return histories


def build_seeded(task, cfg, i):
    seed_all(cfg.seed + i * 1000)
    return build_models(task, cfg)
