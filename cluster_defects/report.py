from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    module="joblib.externals.loky.backend.context",
)

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.feature import hog, local_binary_pattern

from .config import Config
from .feature_bank import FeatureBank
from .similarity import mmd_rbf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _project(groups: dict[str, torch.Tensor], random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    labels = []
    for label, tensor in groups.items():
        values = tensor.detach().cpu().numpy()
        arrays.append(values)
        labels.extend([label] * len(values))
    combined = np.vstack(arrays)
    if len(combined) >= 10:
        perplexity = min(30, max(5, (len(combined) - 1) // 3))
        projected = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_seed,
        ).fit_transform(combined)
    else:
        projected = PCA(n_components=2, random_state=random_seed).fit_transform(combined)
    return projected, np.asarray(labels)


def _plot_space(
    axis,
    title: str,
    groups: dict[str, torch.Tensor],
    footer: str,
    random_seed: int,
) -> None:
    projection, labels = _project(groups, random_seed)
    styles = {
        "Train": ("#3B82C4", "o", 16, 0.22),
        "Val": ("#C45A9D", "o", 16, 0.28),
        "Synthetic": ("#16A34A", "o", 16, 0.9),
    }
    for label, (color, marker, size, alpha) in styles.items():
        mask = labels == label
        if not mask.any():
            continue
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolors="none",
            linewidths=0,
            label=label,
        )
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    if footer:
        axis.text(0.5, -0.16, footer, transform=axis.transAxes, ha="center", va="top")


def _resolve_reference_image(config: Config, row: pd.Series) -> Path:
    recorded_path = Path(str(row["image_path"]))
    if recorded_path.exists():
        return recorded_path

    dataset_root = config.path("dataset_root")
    assert dataset_root is not None
    portable_path = dataset_root / "images" / str(row["split"]) / recorded_path.name
    if portable_path.exists():
        return portable_path
    raise FileNotFoundError(f"Reference image not found: {portable_path}")


def _resolve_synthetic_image(config: Config, row: pd.Series) -> Path:
    for column in ("destination_path", "output_path"):
        recorded_path = Path(str(row[column]))
        if recorded_path.exists():
            return recorded_path
        filename = recorded_path.name
        candidates = (
            config.output_root / "filtered" / "accepted" / filename,
            config.output_root / "candidates" / "images" / filename,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Synthetic image not found for {row['candidate_id']}")


def _load_gray_crop(
    image_path: Path,
    box: tuple[float, float, float, float],
    size: int = 128,
) -> np.ndarray:
    with Image.open(image_path) as opened:
        image = ImageOps.grayscale(opened)
        x1, y1, x2, y2 = box
        left = max(0, min(image.width - 1, int(np.floor(x1))))
        top = max(0, min(image.height - 1, int(np.floor(y1))))
        right = max(left + 1, min(image.width, int(np.ceil(x2))))
        bottom = max(top + 1, min(image.height, int(np.ceil(y2))))
        crop = image.crop((left, top, right, bottom)).resize(
            (size, size),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(crop, dtype=np.uint8)


def _extract_hog_lbp(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    lbp_histograms = []
    for points, radius in ((8, 1), (16, 2)):
        codes = local_binary_pattern(gray, points, radius, method="uniform")
        histogram, _ = np.histogram(
            codes,
            bins=np.arange(points + 3),
            range=(0, points + 2),
        )
        normalized = histogram.astype(np.float32)
        normalized /= max(float(normalized.sum()), 1.0)
        lbp_histograms.append(normalized)
    lbp_features = np.concatenate(lbp_histograms)
    return hog_features, lbp_features


def _handcrafted_groups(
    config: Config,
    bank: FeatureBank,
    results: pd.DataFrame,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    descriptors: dict[str, dict[str, list[np.ndarray]]] = {
        "HOG": {"Train": [], "Val": [], "Synthetic": []},
        "LBP": {"Train": [], "Val": [], "Synthetic": []},
    }

    for _, row in bank.local_metadata.iterrows():
        split = str(row["split"])
        if split not in ("train", "val"):
            continue
        gray = _load_gray_crop(
            _resolve_reference_image(config, row),
            (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
        )
        hog_features, lbp_features = _extract_hog_lbp(gray)
        label = split.title()
        descriptors["HOG"][label].append(hog_features)
        descriptors["LBP"][label].append(lbp_features)

    accepted = results.loc[results["decision"] == "accepted"]
    for _, row in accepted.iterrows():
        gray = _load_gray_crop(
            _resolve_synthetic_image(config, row),
            (
                float(row["best_window_x1"]),
                float(row["best_window_y1"]),
                float(row["best_window_x2"]),
                float(row["best_window_y2"]),
            ),
        )
        hog_features, lbp_features = _extract_hog_lbp(gray)
        descriptors["HOG"]["Synthetic"].append(hog_features)
        descriptors["LBP"]["Synthetic"].append(lbp_features)

    def to_tensors(feature_name: str) -> dict[str, torch.Tensor]:
        return {
            label: torch.from_numpy(np.stack(values)).float()
            for label, values in descriptors[feature_name].items()
        }

    return to_tensors("HOG"), to_tensors("LBP")


def _create_handcrafted_report(
    config: Config,
    bank: FeatureBank,
    results: pd.DataFrame,
    report_dir: Path,
    random_seed: int,
) -> Path:
    hog_groups, lbp_groups = _handcrafted_groups(config, bank, results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _plot_space(
        axes[0],
        "HOG Defect-Window Distribution (Shape and Edges)",
        hog_groups,
        "",
        random_seed,
    )
    _plot_space(
        axes[1],
        "LBP Defect-Window Distribution (Texture)",
        lbp_groups,
        "",
        random_seed,
    )
    fig.suptitle("Cluster-Aware Img2Img: Handcrafted Feature Coverage", fontsize=16)
    fig.tight_layout()
    output_path = report_dir / "handcrafted_feature_distribution.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Handcrafted feature report: {output_path}")
    return output_path


def create_distribution_report(config: Config) -> Path:
    bank = FeatureBank.load(config.output_root / "feature_bank")
    filtered_root = config.output_root / "filtered"
    results = pd.read_csv(filtered_root / "filter_results.csv")
    archive = np.load(filtered_root / "candidate_features.npz", allow_pickle=False)
    feature_ids = archive["candidate_ids"].astype(str)
    accepted_ids = set(
        results.loc[results["decision"] == "accepted", "candidate_id"].astype(str)
    )
    accepted_mask = np.asarray([candidate_id in accepted_ids for candidate_id in feature_ids])
    if not accepted_mask.any():
        raise ValueError("No accepted synthetic features are available for plotting")

    synthetic_global = torch.from_numpy(archive["global_embeddings"][accepted_mask]).float()
    synthetic_local = torch.from_numpy(archive["local_embeddings"][accepted_mask]).float()
    train_global_mask = bank.global_metadata["split"].eq("train").to_numpy(copy=True)
    val_global_mask = bank.global_metadata["split"].eq("val").to_numpy(copy=True)
    train_local_mask = bank.local_metadata["split"].eq("train").to_numpy(copy=True)
    val_local_mask = bank.local_metadata["split"].eq("val").to_numpy(copy=True)
    global_train = bank.global_embeddings[torch.from_numpy(train_global_mask)]
    global_val = bank.global_embeddings[torch.from_numpy(val_global_mask)]
    local_train = bank.local_embeddings[torch.from_numpy(train_local_mask)]
    local_val = bank.local_embeddings[torch.from_numpy(val_local_mask)]

    metrics = [
        {
            "feature_space": "global",
            "mmd_train_val": mmd_rbf(global_train, global_val),
            "mmd_train_plus_synthetic_val": mmd_rbf(
                torch.cat([global_train, synthetic_global]),
                global_val,
            ),
        },
        {
            "feature_space": "local_defect_window",
            "mmd_train_val": mmd_rbf(local_train, local_val),
            "mmd_train_plus_synthetic_val": mmd_rbf(
                torch.cat([local_train, synthetic_local]),
                local_val,
            ),
        },
    ]
    for row in metrics:
        row["delta"] = (
            row["mmd_train_plus_synthetic_val"] - row["mmd_train_val"]
        )

    random_seed = int(config.get("clustering", "random_state"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _plot_space(
        axes[0],
        "Global DINOv3 Feature Distribution",
        {"Train": global_train, "Val": global_val, "Synthetic": synthetic_global},
        f"MMD(Train, Val) = {metrics[0]['mmd_train_val']:.3f}\n"
        f"MMD(Train + Synthetic, Val) = "
        f"{metrics[0]['mmd_train_plus_synthetic_val']:.3f}",
        random_seed,
    )
    _plot_space(
        axes[1],
        "Local Defect-Window DINOv3 Feature Distribution",
        {"Train": local_train, "Val": local_val, "Synthetic": synthetic_local},
        f"MMD(Train, Val) = {metrics[1]['mmd_train_val']:.3f}\n"
        f"MMD(Train + Synthetic, Val) = "
        f"{metrics[1]['mmd_train_plus_synthetic_val']:.3f}",
        random_seed,
    )
    fig.suptitle("Cluster-Aware Img2Img: Accepted Synthetic Coverage", fontsize=16)
    fig.tight_layout()
    report_dir = config.output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "feature_distribution.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(report_dir / "mmd_summary.csv", index=False)
    _create_handcrafted_report(config, bank, results, report_dir, random_seed)
    print(f"Distribution report: {output_path}")
    return output_path
