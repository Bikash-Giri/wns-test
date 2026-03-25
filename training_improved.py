import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class AudioDatasetWithStats(Dataset):
    def __init__(
        self,
        folder: str,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        target_time_frames: int = 128,
    ):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_time_frames = target_time_frames

        self.files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".npy")]
        )
        if not self.files:
            raise ValueError(f"No .npy files found in {folder}")

    def __len__(self):
        return len(self.files)

    def _wav_to_mel_db(self, wav: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=wav.astype(np.float32),
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=1.0)
        return mel_db

    def _pad_or_crop_time(self, mel_db: np.ndarray) -> np.ndarray:
        t = mel_db.shape[1]
        if t < self.target_time_frames:
            pad = self.target_time_frames - t
            mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
        elif t > self.target_time_frames:
            mel_db = mel_db[:, : self.target_time_frames]
        return mel_db

    def _normalize(self, mel_db: np.ndarray):
        mean = float(mel_db.mean())
        std = float(mel_db.std() + 1e-6)
        mel_norm = (mel_db - mean) / std
        return mel_norm, mean, std

    def __getitem__(self, idx):
        wav = np.load(self.files[idx]).squeeze()
        if wav.ndim != 1:
            raise ValueError(f"Expected 1D waveform in {self.files[idx]}, got shape {wav.shape}")

        mel_db = self._wav_to_mel_db(wav)
        mel_db = self._pad_or_crop_time(mel_db)
        mel_norm, _, _ = self._normalize(mel_db)
        mel_norm = np.expand_dims(mel_norm, axis=0)  # [1, 128, 128]
        return torch.tensor(mel_norm, dtype=torch.float32)

    def get_item_with_stats(self, idx):
        wav = np.load(self.files[idx]).squeeze()
        mel_db = self._wav_to_mel_db(wav)
        mel_db = self._pad_or_crop_time(mel_db)
        mel_norm, mean, std = self._normalize(mel_db)
        mel_norm = np.expand_dims(mel_norm, axis=0)
        return torch.tensor(mel_norm, dtype=torch.float32), mean, std


class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 16 * 16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            # No Sigmoid: output can be negative/positive for z-scored mel.
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 16, 16)
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=0.5):
    # L1 tends to keep edges/detail in spectrograms better than plain MSE.
    recon_loss = F.mse_loss(recon_x, x)
    # l1_loss    = F.l1_loss(recon_x, x, reduction="mean")
    # l2_loss    = F.mse_loss(recon_x, x, reduction="mean")
    # recon_loss = l1_loss + 0.5 * l2_loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def mel_db_to_audio(mel_db: np.ndarray, sr=16000, n_fft=1024, hop_length=256):
    mel_power = librosa.db_to_power(mel_db, ref=1.0)
    stft = librosa.feature.inverse.mel_to_stft(mel_power, sr=sr, n_fft=n_fft)
    audio = librosa.griffinlim(
        stft,
        n_iter=128,
        hop_length=hop_length,
        win_length=n_fft,
        momentum=0.99,
        init="random",
    )
    # Keep peaks safe for writing.
    audio = np.clip(audio, -0.99, 0.99)
    return audio


def main():
    dataset = AudioDatasetWithStats("segments")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = VAE(latent_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10000
    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_l, kl_l = loss_function(recon, batch, mu, logvar, beta=0.1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()

        n = len(loader)
        print(
            f"Epoch {epoch + 1:03d} | total: {total_loss / n:.4f} | "
            f"recon: {total_recon / n:.4f} | kl: {total_kl / n:.4f}"
        )

    # Reconstruct one sample with correct de-normalization.
    model.eval()
    with torch.no_grad():
        sample, mean, std = dataset.get_item_with_stats(0)
        sample = sample.unsqueeze(0).to(device)
        recon, _, _ = model(sample)
        recon_norm = recon.squeeze(0).squeeze(0).cpu().numpy()  # [128, 128]

    # Proper inverse: undo normalization before dB->power.
    recon_db = recon_norm * std + mean
    audio = mel_db_to_audio(recon_db, sr=dataset.sr, n_fft=dataset.n_fft, hop_length=dataset.hop_length)
    sf.write("reconstructed_improved.wav", audio, dataset.sr)
    print("Saved: reconstructed_improved.wav")


if __name__ == "__main__":
    main()

