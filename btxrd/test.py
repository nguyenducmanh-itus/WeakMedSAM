import math
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch
from IPython.display import Markdown
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor, nn
from torchvision import models
from torchvision.io import decode_image
from torchvision.transforms import v2 as T


# ImageNet normalization weights per channel
IMAGENET1K_MEAN = [0.485, 0.456, 0.406]
IMAGENET1K_STD = [0.229, 0.224, 0.225]

transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET1K_MEAN, IMAGENET1K_STD),
    ]
)


def load_image(path: str | Path) -> Tensor:
    # Transform images into tensors
    img: Tensor = transform(decode_image(str(path)))

    # Add dimension to imitate batch size equal to 1: (C,H,W) -> (B,C,H,W)
    img = img.unsqueeze(0)
    return img

def inverse_normalize(
    x_norm: Tensor,
    mean: list[float] = IMAGENET1K_MEAN,
    std: list[float] = IMAGENET1K_STD,
) -> Tensor:
    # Ensure mean and std have the correct shape
    _mean = torch.as_tensor(mean).to(x_norm.device).view(1, -1, 1, 1)
    _std = torch.as_tensor(std).to(x_norm.device).view(1, -1, 1, 1)
    # Inverse normalization: x = x_normalized * std + mean
    return x_norm.mul(_std).add(_mean)


reverse_transform = T.Compose(
    [
        T.Lambda(inverse_normalize),
        T.Lambda(lambda x: torch.clamp(x, min=0.0, max=1.0)),
    ]
)

def get_activation(name: str, activations: dict[str, Tensor]) -> Callable:
    def hook(model: nn.Module, tensor: Tensor, output: Tensor) -> None:
        # map layer's `name` to layer's output value
        activations[name] = output.detach()

    return hook


def set_hooks(model: nn.Module, layer_ids: list[str], out: dict[str, Tensor]) -> None:
    layer_ids = [str(i) for i in layer_ids]
    for name, module in model.named_modules():
        if name in layer_ids:
            module.register_forward_hook(get_activation(name, out))
            
            
def visualize_feature_maps(
    feature_map: Tensor | np.ndarray,
    max_maps: int | None = None,
    max_cols: int = 8,
    figsize_per_plot: float = 1.0,
    norm: Literal["linear", "log", "symlog", "logit", None] = None,
    cmap: str = "viridis",
):
    if isinstance(feature_map, Tensor):
        feature_map = feature_map.cpu().numpy()

    if feature_map.ndim == 4:
        feature_map = feature_map.squeeze(0)  # remove batch dimension if present
    assert feature_map.ndim == 3, "Expected tensor shape (C, H, W)"

    C, H, W = feature_map.shape

    if max_maps:
        C = min(C, max_maps)

    n_cols = min(C, max_cols)
    n_rows = math.ceil(C / n_cols)

    figsize = (figsize_per_plot * n_cols, figsize_per_plot * n_rows)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, frameon=False, squeeze=False)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)

    for ax in axes.flat:
        ax.axis("off")

    for i in range(C):
        t = feature_map[i]
        axes.flat[i].imshow(t, cmap=cmap, norm=norm, aspect="equal", interpolation="none")

    return fig, axes

def minmax_scale_per_channel(arr: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-channel MinMax normalization. Expects (C, W, H)."""
    assert arr.ndim == 3, f"{arr.ndim=}"

    c_min = arr.min(axis=(1, 2), keepdims=True)
    c_max = arr.max(axis=(1, 2), keepdims=True)

    scaled = (arr - c_min) / (c_max - c_min + eps)  # avoid division by zero
    return scaled


def pca_rgb(
    feature_map: np.ndarray | Tensor,
    n_components: Literal[1, 3] = 3,
    normalize: bool = True,
    random_state: int | None = None,
) -> np.ndarray:
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.cpu().numpy()

    if feature_map.ndim == 4:
        feature_map = feature_map.squeeze(0)  # remove batch dimension if present
    assert feature_map.ndim == 3, "Expected array shape (C, H, W)"

    C, H, W = feature_map.shape
    pca = PCA(n_components=n_components, random_state=random_state)
    flat = feature_map.reshape(C, -1).T
    rgb = pca.fit_transform(flat).T.reshape(n_components, H, W)

    if normalize:
        rgb = minmax_scale_per_channel(rgb)

    return rgb


def visualize_feature_maps_pca(
    feature_maps: dict[str, Tensor],
    n_components: Literal[1, 3] = 3,
    max_cols: int = 4,
    figsize_per_plot: float = 2.0,
    norm: Literal["linear", "log", "symlog", "logit", None] = None,
    subtitles: bool = True,
    cmap: str = "viridis",
):
    c = len(feature_maps)
    n_cols = min(c, max_cols)
    n_rows = math.ceil(c / n_cols)
    fig_size = (figsize_per_plot * n_cols, figsize_per_plot * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, squeeze=False, frameon=False)
    fig.subplots_adjust(wspace=0.03, hspace=0.20, top=0.85)

    for ax in axes.flat:
        ax.axis("off")

    for ax, (layer, feature_map) in zip(axes.flat, feature_maps.items(), strict=False):
        rgb_features = pca_rgb(feature_map, n_components=n_components)
        rgb_features = rgb_features.transpose(1, 2, 0)
        rgb_features = rgb_features.squeeze()

        ax.imshow(rgb_features, cmap=cmap, norm=norm, aspect="equal", interpolation="none")
        if subtitles:
            ax.set_title(layer, color="0.5")

    return fig, axes

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# inspect layers within ResNet
model
sample = load_image("data/BTXRD/images/IMG000023.jpeg")
selected_layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]
resnet_feature_maps: dict[str, Tensor] = {}
set_hooks(model, selected_layers, resnet_feature_maps)

with torch.no_grad():
    model(sample)

for layer, filters in resnet_feature_maps.items():
    Markdown(f'### Layer "{layer}"')
    visualize_feature_maps(filters, max_maps=8 * 8, norm="linear")
    plt.show()

