import math

import torch
from torch import nn
from torch.nn import functional as F


class VanillaAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 784),
            nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, images):
        latent = self.encoder(images)
        return self.decoder(latent), latent


class SparseAE(VanillaAE):
    # Penalti L1 ditambahkan ke loss, jadi bentuk arsitekturnya sama.
    pass


class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mean = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 784),
            nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28)),
        )

    def encode(self, images):
        hidden = self.encoder(images)
        return self.mean(hidden), self.logvar(hidden)

    def forward(self, images):
        mean, logvar = self.encode(images)
        std = (0.5 * logvar).exp()
        z = mean + std * torch.randn_like(std)
        return self.decoder(z), mean, logvar


class GANGenerator(nn.Sequential):
    def __init__(self, latent_dim=100):
        super().__init__(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 784), nn.Tanh(), nn.Unflatten(1, (1, 28, 28)),
        )


class GANDiscriminator(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1),
        )


class DCGenerator(nn.Sequential):
    def __init__(self, latent_dim=100):
        super().__init__(
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, 128, 7, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),
        )


class DCDiscriminator(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 7), nn.Flatten(),
        )


class Classifier(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 10)

    def encode(self, images):
        hidden = F.relu(self.fc1(images.flatten(1)))
        return F.relu(self.fc2(hidden))

    def forward(self, images):
        return self.fc3(self.encode(images))

    def project_features(self, features):
        hidden = F.relu(features @ self.fc2.weight)
        pixels = (hidden @ self.fc1.weight).sigmoid()
        return pixels.reshape(-1, 1, 28, 28)

    def project_classes(self, one_hot):
        return self.project_features(F.relu(one_hot @ self.fc3.weight))


class FrozenDecoder(nn.Sequential):
    def __init__(self, latent_dim=64):
        super().__init__(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 784),
            nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28)),
        )


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        frequency = torch.exp(-math.log(10_000) * torch.arange(width // 2) / (width // 2))
        self.register_buffer("frequency", frequency)
        self.projection = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                        nn.Linear(width * 2, width * 2))

    def forward(self, step):
        phase = step[:, None] * self.frequency[None, :]
        return self.projection(torch.cat([phase.sin(), phase.cos()], dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, source, target, time_width):
        super().__init__()
        self.conv1 = nn.Conv2d(source, target, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, target)
        self.time = nn.Linear(time_width, target)
        self.conv2 = nn.Conv2d(target, target, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, target)
        self.skip = nn.Identity() if source == target else nn.Conv2d(source, target, 1)

    def forward(self, x, embedding):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(embedding)[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class Denoiser(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.time = TimeEmbedding(width)
        self.down = ResidualBlock(1, width, 2 * width)
        self.middle = ResidualBlock(width, 2 * width, 2 * width)
        self.up = ResidualBlock(3 * width, width, 2 * width)
        self.output = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, images, step):
        embedding = self.time(step)
        skip = self.down(images, embedding)
        compressed = self.middle(F.avg_pool2d(skip, 2), embedding)
        up = F.interpolate(compressed, size=skip.shape[-2:], mode="nearest")
        return self.output(self.up(torch.cat([up, skip], dim=1), embedding))


class Diffusion(nn.Module):
    def __init__(self, steps=100, width=24):
        super().__init__()
        self.denoiser = Denoiser(width)
        self.steps = steps

        t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        beta = (1 - curve[1:] / curve[:-1]).clamp(0.0001, 0.999)
        alpha = 1 - beta
        alpha_bar = alpha.cumprod(0)
        previous = torch.cat((torch.ones(1, dtype=alpha.dtype), alpha_bar[:-1]))

        schedules = {
            "alpha_bar": alpha_bar,
            "coef_clean": beta * previous.sqrt() / (1 - alpha_bar),
            "coef_noisy": alpha.sqrt() * (1 - previous) / (1 - alpha_bar),
            "posterior_std": (beta * (1 - previous) / (1 - alpha_bar)).sqrt(),
        }
        for name, values in schedules.items():
            self.register_buffer(name, values.float())

    def training_loss(self, images, rng=None):
        steps = torch.randint(self.steps, (len(images),), device=images.device, generator=rng)
        noise = torch.randn(images.shape, device=images.device, generator=rng)
        alpha_bar = self.alpha_bar[steps, None, None, None]
        clean = images * 2 - 1
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        predicted = self.denoiser(noisy, steps)
        return F.mse_loss(predicted, noise)

    @torch.inference_mode()
    def sample(self, initial_noise, rng=None, keep_trajectory=False):
        images = initial_noise
        snapshots = []
        targets = set(torch.linspace(self.steps, 0, 6).round().int().tolist())
        if keep_trajectory:
            snapshots.append((self.steps, images.cpu()))

        for i in reversed(range(self.steps)):
            timestep = torch.full((len(images),), i, device=images.device, dtype=torch.long)
            with torch.autocast(device_type=images.device.type, enabled=images.is_cuda):
                predicted = self.denoiser(images, timestep)
            alpha_bar = self.alpha_bar[i]
            clean = (images - (1 - alpha_bar).sqrt() * predicted) / alpha_bar.sqrt()
            clean = clean.clamp(-1, 1)
            images = self.coef_clean[i] * clean + self.coef_noisy[i] * images
            if i > 0:
                noise = torch.randn(images.shape, device=images.device, generator=rng)
                images = images + self.posterior_std[i] * noise
            if keep_trajectory and i in targets:
                snapshots.append((i, images.cpu()))

        return ((images + 1) / 2).clamp(0, 1), snapshots
