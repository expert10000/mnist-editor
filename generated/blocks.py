import torch
from torch import nn


class ConvBnGelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * mask.floor()


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class MultiBranchResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, branch_count=4, use_se=False, se_reduction=8, drop_path=0.0):
        super().__init__()
        if out_channels % branch_count != 0:
            raise ValueError("out_channels must be divisible by branch_count")
        branch_channels = out_channels // branch_count
        self.branches = nn.ModuleList([
            ConvBnGelu(in_channels, branch_channels, 3, stride, 1),
            ConvBnGelu(in_channels, branch_channels, 5, stride, 2),
            ConvBnGelu(in_channels, branch_channels, 3, stride, 2, dilation=2),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.GELU(),
            ),
        ])
        self.fusion = nn.Sequential(
            ConvBnGelu(out_channels, out_channels, 1, 1, 0),
            ConvBnGelu(out_channels, out_channels, 3, 1, 1),
        )
        self.se = SqueezeExcite(out_channels, se_reduction) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, x):
        merged = torch.cat([branch(x) for branch in self.branches], dim=1)
        merged = self.fusion(merged)
        merged = self.se(merged)
        merged = self.drop_path(merged)
        return self.activation(merged + self.shortcut(x))


class PoolingFusion(nn.Module):
    def __init__(self, mode="gap_gmp"):
        super().__init__()
        self.mode = mode
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg = self.avg(x).flatten(1)
        if self.mode == "gap":
            return avg
        return torch.cat([avg, self.max(x).flatten(1)], dim=1)


class FeatureHead(nn.Module):
    def __init__(self, in_features, hidden_features=192, embedding_features=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, embedding_features),
        )

    def forward(self, x):
        return self.net(x)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, classes),
        )

    def forward(self, x):
        return self.net(x)
