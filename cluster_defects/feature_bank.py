from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    module="joblib.externals.loky.backend.context",
)

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from .config import Config
from .dataset import list_images, matching_label_path, read_yolo_boxes
from .dino import DinoExtractor, derive_window_sizes
from .similarity import leave_one_out_scores


@dataclass
class FeatureBank:
    global_embeddings: torch.Tensor
    local_embeddings: torch.Tensor
    cluster_centers: torch.Tensor
    global_metadata: pd.DataFrame
    local_metadata: pd.DataFrame
    thresholds: dict[str, float]
    window_sizes: list[tuple[int, int]]

    @classmethod
    def load(cls, directory: Path) -> "FeatureBank":
        archive_path = directory / "real_features.npz"
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Feature bank not found: {archive_path}. Run build-bank first."
            )
        archive = np.load(archive_path, allow_pickle=False)
        summary = json.loads((directory / "bank_summary.json").read_text(encoding="utf-8"))
        return cls(
            global_embeddings=torch.from_numpy(archive["global_embeddings"]).float(),
            local_embeddings=torch.from_numpy(archive["local_embeddings"]).float(),
            cluster_centers=torch.from_numpy(archive["cluster_centers"]).float(),
            global_metadata=pd.read_csv(directory / "global_metadata.csv"),
            local_metadata=pd.read_csv(directory / "local_metadata.csv"),
            thresholds={key: float(value) for key, value in summary["thresholds"].items()},
            window_sizes=[tuple(size) for size in summary["window_sizes"]],
        )


def _load_external_clusters(
    manifest_path: Path,
    dataset_root: Path,
    image_paths: list[str],
) -> np.ndarray:
    table = pd.read_csv(manifest_path)
    required = {"image_path", "cluster_id"}
    if not required.issubset(table.columns):
        raise ValueError(f"Cluster manifest must contain columns: {sorted(required)}")

    assignments: dict[str, int] = {}
    for row in table.itertuples(index=False):
        path = Path(str(row.image_path))
        if not path.is_absolute():
            path = dataset_root / path
        assignments[str(path.resolve())] = int(row.cluster_id)

    missing = [path for path in image_paths if str(Path(path).resolve()) not in assignments]
    if missing:
        raise ValueError(
            f"Cluster manifest is missing {len(missing)} reference images; "
            f"first missing image: {missing[0]}"
        )
    return np.asarray([assignments[str(Path(path).resolve())] for path in image_paths])


def build_feature_bank(config: Config) -> FeatureBank:
    paths = config.section("paths")
    dataset = config.section("dataset")
    dino = config.section("dino")
    clustering = config.section("clustering")
    dataset_root = config.path("dataset_root")
    weights = config.path("dino_weights")
    repo = config.path("dino_repo")
    assert dataset_root is not None and weights is not None and repo is not None

    extractor = DinoExtractor(
        weights_path=weights,
        repo_path=repo,
        model_name=dino["model_name"],
        image_size=int(dino["image_size"]),
        device=dino["device"],
    )
    global_embeddings: list[torch.Tensor] = []
    local_embeddings: list[torch.Tensor] = []
    global_rows: list[dict] = []
    local_rows: list[dict] = []

    for split in dataset["reference_splits"]:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing YOLO split directories for {split}")

        image_paths = list_images(image_dir, dataset["supported_extensions"])
        for image_path in tqdm(image_paths, desc=f"DINO reference: {split}"):
            features = extractor.extract(image_path)
            label_path = matching_label_path(image_path, image_dir, label_dir)
            global_index = len(global_embeddings)
            global_embeddings.append(features.global_embedding)
            global_rows.append(
                {
                    "global_index": global_index,
                    "split": split,
                    "image_path": str(image_path.resolve()),
                    "label_path": str(label_path.resolve()),
                    "image_width": features.image_width,
                    "image_height": features.image_height,
                }
            )

            boxes = read_yolo_boxes(
                label_path,
                features.image_width,
                features.image_height,
            )
            for box_index, box in enumerate(boxes):
                padded = box.expand(
                    features.image_width,
                    features.image_height,
                    float(dataset["box_padding_fraction"]),
                )
                embedding, grid_box = extractor.pool_pixel_box(
                    features.feature_map,
                    padded,
                    features.image_width,
                    features.image_height,
                )
                local_index = len(local_embeddings)
                local_embeddings.append(embedding)
                gx1, gy1, gx2, gy2 = grid_box
                local_rows.append(
                    {
                        "local_index": local_index,
                        "global_index": global_index,
                        "split": split,
                        "image_path": str(image_path.resolve()),
                        "label_path": str(label_path.resolve()),
                        "box_index": box_index,
                        "class_id": box.class_id,
                        "x1": box.x1,
                        "y1": box.y1,
                        "x2": box.x2,
                        "y2": box.y2,
                        "grid_x1": gx1,
                        "grid_y1": gy1,
                        "grid_x2": gx2,
                        "grid_y2": gy2,
                        "grid_width": gx2 - gx1,
                        "grid_height": gy2 - gy1,
                    }
                )

    if len(global_embeddings) < 2 or len(local_embeddings) < 2:
        raise ValueError("Feature bank requires at least two images and two defect boxes")

    global_bank = torch.stack(global_embeddings)
    local_bank = torch.stack(local_embeddings)
    global_metadata = pd.DataFrame(global_rows)
    local_metadata = pd.DataFrame(local_rows)

    train_mask = global_metadata["split"] == dataset["source_split"]
    train_features = global_bank[
        torch.from_numpy(train_mask.to_numpy(copy=True))
    ].numpy()
    manifest_path = config.path("cluster_manifest", allow_empty=True)
    if manifest_path is not None:
        cluster_ids = _load_external_clusters(
            manifest_path,
            dataset_root,
            global_metadata["image_path"].tolist(),
        )
        unique_ids = sorted(set(cluster_ids.tolist()))
        centers = np.stack(
            [
                global_bank[torch.from_numpy(cluster_ids == cluster_id)].mean(dim=0).numpy()
                for cluster_id in unique_ids
            ]
        )
    else:
        n_clusters = min(int(clustering["n_clusters"]), len(train_features))
        model = KMeans(
            n_clusters=n_clusters,
            random_state=int(clustering["random_state"]),
            n_init=10,
        ).fit(train_features)
        cluster_ids = model.predict(global_bank.numpy())
        centers = model.cluster_centers_
    global_metadata["cluster_id"] = cluster_ids
    cluster_lookup = global_metadata.set_index("global_index")["cluster_id"]
    local_metadata["cluster_id"] = local_metadata["global_index"].map(cluster_lookup)

    global_top_k, global_max = leave_one_out_scores(global_bank, int(dino["top_k"]))
    local_top_k, local_max = leave_one_out_scores(local_bank, int(dino["top_k"]))
    lower_quantile = float(dino["lower_quantile"])
    duplicate_quantile = float(dino["duplicate_quantile"])
    thresholds = {
        "global_lower": float(torch.quantile(global_top_k, lower_quantile)),
        "local_lower": float(torch.quantile(local_top_k, lower_quantile)),
        "global_duplicate": float(torch.quantile(global_max, duplicate_quantile)),
        "local_duplicate": float(torch.quantile(local_max, duplicate_quantile)),
        "global_real_mean": float(global_top_k.mean()),
        "global_real_std": float(global_top_k.std()),
        "local_real_mean": float(local_top_k.mean()),
        "local_real_std": float(local_top_k.std()),
    }
    window_sizes = derive_window_sizes(
        local_metadata,
        dino["fallback_window_sizes"],
        extractor.grid_height,
        extractor.grid_width,
    )

    output_dir = config.output_root / "feature_bank"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "real_features.npz",
        global_embeddings=global_bank.numpy(),
        local_embeddings=local_bank.numpy(),
        cluster_centers=np.asarray(centers, dtype=np.float32),
    )
    global_metadata.to_csv(output_dir / "global_metadata.csv", index=False)
    local_metadata.to_csv(output_dir / "local_metadata.csv", index=False)
    summary = {
        "model_name": dino["model_name"],
        "image_size": int(dino["image_size"]),
        "device": str(extractor.device),
        "global_references": len(global_bank),
        "local_references": len(local_bank),
        "clusters": int(global_metadata["cluster_id"].nunique()),
        "cluster_counts": {
            str(key): int(value)
            for key, value in global_metadata.groupby(["split", "cluster_id"]).size().items()
        },
        "thresholds": thresholds,
        "window_sizes": window_sizes,
    }
    (output_dir / "bank_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Feature bank saved to {output_dir}")
    print(json.dumps(summary, indent=2))
    return FeatureBank(
        global_embeddings=global_bank,
        local_embeddings=local_bank,
        cluster_centers=torch.from_numpy(np.asarray(centers)).float(),
        global_metadata=global_metadata,
        local_metadata=local_metadata,
        thresholds=thresholds,
        window_sizes=window_sizes,
    )
