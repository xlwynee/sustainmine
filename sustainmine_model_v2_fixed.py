"""
SustainMine multimodal model.
- Sentinel-2: 6-channel image input
- Sentinel-5P: 3 numerical gas features (NO2, SO2, CO)
- Ground sensors: 9 numerical features
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SatelliteImageEncoder(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 6,
                 embed_dim: int = 384, depth: int = 6, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, 4.0, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x


class NumericalFeatureEncoder(nn.Module):
    def __init__(self, s5_dim: int = 3, sensor_dim: int = 9, embed_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.s5p_encoder = nn.Sequential(
            nn.Linear(s5_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
        )

    def forward(self, sentinel5p_data: torch.Tensor, sensor_data: torch.Tensor) -> torch.Tensor:
        s5_feat = self.s5p_encoder(sentinel5p_data)
        sensor_feat = self.sensor_encoder(sensor_data)
        return self.fusion(torch.cat([s5_feat, sensor_feat], dim=1))


class MultimodalFusion(nn.Module):
    def __init__(self, embed_dim: int = 384, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, numerical_feat: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        query = numerical_feat.unsqueeze(1)
        attn_out, _ = self.cross_attn(self.norm1(query), image_tokens, image_tokens)
        x = query + attn_out
        x = x + self.ffn(self.norm2(x))
        return x.squeeze(1)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int = 384, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForecastingHead(nn.Module):
    def __init__(self, embed_dim: int = 384, num_forecast_steps: int = 3, num_pollutants: int = 6):
        super().__init__()
        self.forecast_queries = nn.Parameter(torch.randn(1, num_forecast_steps, embed_dim))
        self.temporal_attn = nn.MultiheadAttention(embed_dim, 6, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_pollutants),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        queries = self.forecast_queries.expand(batch_size, -1, -1)
        key_value = features.unsqueeze(1)
        out, _ = self.temporal_attn(queries, key_value, key_value)
        out = self.norm(out + queries)
        return self.output_proj(out)


class SustainMineModelV2(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 6,
                 s5_dim: int = 3, sensor_dim: int = 9, embed_dim: int = 384, depth: int = 6,
                 num_heads: int = 6, num_classes: int = 3, num_forecast_steps: int = 3,
                 num_pollutants: int = 6, dropout: float = 0.1, freeze_image_encoder: bool = False):
        super().__init__()
        self.image_encoder = SatelliteImageEncoder(img_size, patch_size, in_channels, embed_dim, depth, num_heads, dropout)
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        self.numerical_encoder = NumericalFeatureEncoder(s5_dim, sensor_dim, embed_dim, dropout)
        self.fusion = MultimodalFusion(embed_dim, num_heads, dropout)
        self.classifier = ClassificationHead(embed_dim, num_classes)
        self.forecaster = ForecastingHead(embed_dim, num_forecast_steps, num_pollutants)

    def forward(self, sentinel2_image: torch.Tensor, sentinel5p_data: torch.Tensor, sensor_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, image_tokens = self.image_encoder(sentinel2_image)
        numerical_feat = self.numerical_encoder(sentinel5p_data, sensor_data)
        fused = self.fusion(numerical_feat, image_tokens)
        return {
            "classification": self.classifier(fused),
            "forecast": self.forecaster(fused),
            "features": fused,
        }


class SustainMineModel(SustainMineModelV2):
    """Backward-compatible alias for training scripts."""
    pass


class SustainMineDataset(Dataset):
    def __init__(self, sentinel2_images, sentinel5_features, sensor_data, labels, forecasts):
        self.images = torch.as_tensor(sentinel2_images, dtype=torch.float32)
        self.s5 = torch.as_tensor(sentinel5_features, dtype=torch.float32)
        self.sensors = torch.as_tensor(sensor_data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.forecasts = torch.as_tensor(forecasts, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],
            "s5": self.s5[idx],
            "sensor": self.sensors[idx],
            "label": self.labels[idx],
            "forecast": self.forecasts[idx],
        }


def compute_multi_task_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor,
                            forecasts: torch.Tensor, alpha: float = 0.5,
                            beta: float = 0.5) -> Dict[str, torch.Tensor]:
    class_loss = F.cross_entropy(outputs["classification"], labels)
    forecast_loss = F.mse_loss(outputs["forecast"], forecasts)
    total_loss = alpha * class_loss + beta * forecast_loss
    return {
        "total": total_loss,
        "classification": class_loss,
        "forecast": forecast_loss,
    }


if __name__ == "__main__":
    model = SustainMineModelV2()
    s2 = torch.randn(2, 6, 224, 224)
    s5 = torch.randn(2, 3)
    sensors = torch.randn(2, 9)
    out = model(s2, s5, sensors)
    print("classification:", out["classification"].shape)
    print("forecast:", out["forecast"].shape)
