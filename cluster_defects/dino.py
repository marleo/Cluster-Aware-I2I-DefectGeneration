from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import YoloBox


@dataclass
class DinoFeatures:
    global_embedding: torch.Tensor
    feature_map: torch.Tensor
    image_width: int
    image_height: int


def parse_forward_features(output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    global_keys = ("x_norm_clstoken", "x_norm_cls_token", "x_clstoken")
    patch_keys = ("x_norm_patchtokens", "x_norm_patch_tokens", "x_patchtokens")
    global_features = next((output[key] for key in global_keys if key in output), None)
    patch_features = next((output[key] for key in patch_keys if key in output), None)
    if global_features is None or patch_features is None:
        raise KeyError(f"Could not find expected DINO tokens. Keys: {list(output.keys())}")
    return global_features, patch_features


class DinoExtractor:
    def __init__(
        self,
        weights_path: Path,
        repo_path: Path,
        model_name: str,
        image_size: int = 512,
        device: str = "auto",
    ):
        self.weights_path = weights_path
        self.repo_path = repo_path
        self.model_name = model_name
        self.image_size = image_size
        self.device = self._resolve_device(device)
        self.model = self._load_model()
        self.patch_height, self.patch_width = self._patch_size()
        self.grid_height = image_size // self.patch_height
        self.grid_width = image_size // self.patch_width

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _load_model(self) -> torch.nn.Module:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"DINO weights not found: {self.weights_path}")
        if not self.repo_path.exists():
            raise FileNotFoundError(f"DINO repository not found: {self.repo_path}")

        repo_string = str(self.repo_path)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        from dinov3.hub import backbones

        factory = getattr(backbones, self.model_name)
        model = factory(weights=str(self.weights_path))
        return model.to(device=self.device, dtype=torch.float32).eval()

    def _patch_size(self) -> tuple[int, int]:
        size = self.model.patch_embed.patch_size
        if isinstance(size, int):
            return size, size
        return int(size[0]), int(size[1])

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return ((tensor - mean) / std).to(self.device)

    @torch.inference_mode()
    def extract(self, image_path: Path) -> DinoFeatures:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            output = self.model.forward_features(self._preprocess(image))

        global_features, patch_features = parse_forward_features(output)
        expected = self.grid_height * self.grid_width
        if patch_features.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} patch tokens, got {patch_features.shape[1]}"
            )

        global_embedding = F.normalize(global_features[0].float(), dim=0).cpu()
        feature_map = patch_features[0].reshape(
            self.grid_height,
            self.grid_width,
            patch_features.shape[-1],
        )
        feature_map = F.normalize(feature_map.float(), dim=-1).cpu()
        return DinoFeatures(global_embedding, feature_map, width, height)

    def pixel_box_to_grid_box(
        self,
        box: YoloBox,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        gx1 = int(np.floor(box.x1 / image_width * self.grid_width))
        gy1 = int(np.floor(box.y1 / image_height * self.grid_height))
        gx2 = int(np.ceil(box.x2 / image_width * self.grid_width))
        gy2 = int(np.ceil(box.y2 / image_height * self.grid_height))
        gx1 = max(0, min(gx1, self.grid_width - 1))
        gy1 = max(0, min(gy1, self.grid_height - 1))
        gx2 = max(gx1 + 1, min(gx2, self.grid_width))
        gy2 = max(gy1 + 1, min(gy2, self.grid_height))
        return gx1, gy1, gx2, gy2

    @staticmethod
    def pool_grid_region(
        feature_map: torch.Tensor,
        grid_box: tuple[int, int, int, int],
    ) -> torch.Tensor:
        gx1, gy1, gx2, gy2 = grid_box
        tokens = feature_map[gy1:gy2, gx1:gx2]
        return F.normalize(tokens.mean(dim=(0, 1)), dim=0)

    def pool_pixel_box(
        self,
        feature_map: torch.Tensor,
        box: YoloBox,
        image_width: int,
        image_height: int,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        grid_box = self.pixel_box_to_grid_box(box, image_width, image_height)
        return self.pool_grid_region(feature_map, grid_box), grid_box


def derive_window_sizes(
    local_metadata,
    fallback_sizes: list[list[int]],
    grid_height: int,
    grid_width: int,
) -> list[tuple[int, int]]:
    sizes: set[tuple[int, int]] = set()
    for quantile in (0.25, 0.50, 0.75):
        height = int(round(local_metadata["grid_height"].quantile(quantile)))
        width = int(round(local_metadata["grid_width"].quantile(quantile)))
        sizes.add((max(1, min(height, grid_height)), max(1, min(width, grid_width))))
    for height, width in fallback_sizes:
        if height <= grid_height and width <= grid_width:
            sizes.add((int(height), int(width)))
    return sorted(sizes, key=lambda size: size[0] * size[1])

