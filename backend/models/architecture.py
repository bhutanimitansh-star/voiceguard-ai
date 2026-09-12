"""
VoiceGuard AI - Deep Learning Architecture
============================================
CNN + BiLSTM + Attention network for Human vs AI-generated voice
classification, operating on log-Mel spectrograms.

Design rationale
-----------------
- Convolutional front-end: learns local time-frequency textures (formant
  structure, vocoder artifacts, spectral checkerboard patterns typical of
  GAN/diffusion vocoders) with residual blocks to keep gradients healthy
  and allow deeper feature extraction without degradation.
- BiLSTM: models long-range temporal dependencies across the spectrogram's
  time axis in both directions - useful because unnatural prosody/pitch
  contours in synthetic speech often only become apparent over time.
- Attention layer: learns to weight the most discriminative time steps
  (e.g. a glitchy phoneme boundary), which also gives us a ready-made
  explainability signal (attention weights -> highlighted timeline
  regions) without needing a separate Grad-CAM pass.
- Grad-CAM (see explainability.py) is additionally computed over the last
  convolutional feature map for a frequency x time saliency heatmap.

Input:  (batch, 1, n_mels, T)   log-Mel spectrogram
Output: (batch, num_classes)    softmax logits [Human, AI Voice]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Convolutional residual block
# ----------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    """Two 3x3 convs with BatchNorm + ReLU and a residual (skip) connection.

    A 1x1 projection is used on the skip path whenever the channel count
    changes, so the block is usable for both same-size and up-channeling
    transitions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out = F.relu(out + identity)
        return out


# ----------------------------------------------------------------------
# Temporal (Bahdanau-style) attention over BiLSTM outputs
# ----------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """Additive attention that produces a single context vector plus
    per-timestep attention weights (used for explainability)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (batch, T, hidden_dim)
        scores = self.attn_fc(lstm_out).squeeze(-1)        # (batch, T)
        weights = F.softmax(scores, dim=1)                  # (batch, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out)  # (batch, 1, hidden_dim)
        context = context.squeeze(1)                         # (batch, hidden_dim)
        return context, weights


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class VoiceGuardNet(nn.Module):
    """CNN -> BiLSTM -> Attention -> Dense -> Softmax voice authenticity
    classifier.

    Args:
        n_mels: number of Mel frequency bins (frequency axis of input)
        num_classes: number of output classes (2: Human / AI Voice)
        lstm_hidden: hidden size per LSTM direction
        lstm_layers: number of stacked BiLSTM layers
        cnn_channels: channel progression for the residual CNN stack
        dropout: dropout probability used throughout
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        cnn_channels=(32, 64, 128),
        dropout: float = 0.3,
    ):
        super().__init__()

        c1, c2, c3 = cnn_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # Each block halves both frequency and time dims via stride=2
        self.block1 = ResidualConvBlock(c1, c1, stride=1, dropout=dropout)
        self.block2 = ResidualConvBlock(c1, c2, stride=2, dropout=dropout)
        self.block3 = ResidualConvBlock(c2, c3, stride=2, dropout=dropout)

# Reduce only the frequency dimension before the LSTM.
# CNN output: [B, 128, 32, T]
# After pooling: [B, 128, 8, T]
        self.freq_pool = nn.AdaptiveAvgPool2d((8, None))

        self.freq_after_cnn = 8
        self.cnn_out_channels = c3
        self.lstm_input_size = self.cnn_out_channels * self.freq_after_cnn

        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attention = TemporalAttention(hidden_dim=lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Auxiliary head that fuses hand-crafted acoustic summary stats
        # (MFCC/Chroma/Centroid/Rolloff/ZCR/Pitch/HNR) with the deep
        # embedding, improving robustness on edge cases / short clips.
        self.aux_feature_dim = 9  # matches AcousticFeatures.summary_stats()
        self.aux_fc = nn.Sequential(
            nn.Linear(self.aux_feature_dim, 32),
            nn.ReLU(inplace=True),
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual CNN stack. Kept separate so Grad-CAM can hook
        into the final conv activation map easily."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)  # (batch, C, F', T')
        return x

    def forward(self, mel_spectrogram: torch.Tensor, aux_features: torch.Tensor = None):
        """
        Args:
            mel_spectrogram: (batch, 1, n_mels, T)
            aux_features: optional (batch, 9) hand-crafted feature vector

        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, T') - for explainability / timeline highlighting
        """

        # CNN feature extraction
        cnn_out = self.extract_cnn_features(mel_spectrogram)
        # Shape: (B, C, F, T)

        # Reduce frequency dimension
        # (B, 128, 32, T) -> (B, 128, 8, T)
        cnn_out = self.freq_pool(cnn_out)

        batch, channels, freq, time = cnn_out.shape

        # Move time dimension forward
        # (B, C, F, T) -> (B, T, C, F)
        seq = cnn_out.permute(0, 3, 1, 2)

        # Collapse C and F into the LSTM feature dimension
        # (B, T, C, F) -> (B, T, C*F)
        seq = seq.reshape(
            batch,
            time,
            channels * freq
        )

        # BiLSTM
        lstm_out, _ = self.bilstm(seq)

        # Temporal attention
        context, attn_weights = self.attention(lstm_out)

        # Fuse auxiliary acoustic features
        if aux_features is not None:
            aux_embed = self.aux_fc(aux_features)
            fused = torch.cat(
                [context, aux_embed],
                dim=1
            )
            logits = self.fusion_classifier(fused)

        else:
            logits = self.classifier(context)

        return logits, attn_weights


def build_model(config: dict = None) -> VoiceGuardNet:
    """Factory function used by both training and inference code paths."""
    config = config or {}
    return VoiceGuardNet(
        n_mels=config.get("n_mels", 128),
        num_classes=config.get("num_classes", 2),
        lstm_hidden=config.get("lstm_hidden", 64),
        lstm_layers=config.get("lstm_layers", 1),
        dropout=config.get("dropout", 0.3),
    )


if __name__ == "__main__":
    # Quick shape sanity check
    model = build_model()
    dummy_mel = torch.randn(4, 1, 128, 251)   # ~4s window at hop=256, sr=16000
    dummy_aux = torch.randn(4, 9)
    logits, attn = model(dummy_mel, dummy_aux)
    print("logits:", logits.shape)   # (4, 2)
    print("attn:", attn.shape)       # (4, T')
