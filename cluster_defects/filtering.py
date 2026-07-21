from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from .config import Config
from .dataset import YoloBox, read_yolo_boxes
from .dino import DinoExtractor
from .feature_bank import FeatureBank
from .similarity import maximum_similarity, top_k_mean_similarity


def _grid_box_to_pixel_box(
    grid_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
) -> tuple[int, int, int, int]:
    gx1, gy1, gx2, gy2 = grid_box
    return (
        round(gx1 / grid_width * image_width),
        round(gy1 / grid_height * image_height),
        round(gx2 / grid_width * image_width),
        round(gy2 / grid_height * image_height),
    )


def search_local_windows(
    feature_map: torch.Tensor,
    real_local_bank: torch.Tensor,
    window_sizes: list[tuple[int, int]],
    stride: int,
    best_windows_to_average: int,
    top_k: int,
    image_width: int,
    image_height: int,
) -> dict:
    grid_height, grid_width, _ = feature_map.shape
    rows: list[dict] = []
    for window_height, window_width in window_sizes:
        embeddings = []
        positions = []
        for gy1 in range(0, grid_height - window_height + 1, stride):
            for gx1 in range(0, grid_width - window_width + 1, stride):
                grid_box = (
                    gx1,
                    gy1,
                    gx1 + window_width,
                    gy1 + window_height,
                )
                embeddings.append(DinoExtractor.pool_grid_region(feature_map, grid_box))
                positions.append(grid_box)
        if not embeddings:
            continue
        stacked = torch.stack(embeddings)
        top_k_scores = top_k_mean_similarity(stacked, real_local_bank, top_k)
        max_scores = maximum_similarity(stacked, real_local_bank)
        for index, grid_box in enumerate(positions):
            rows.append(
                {
                    "grid_box": grid_box,
                    "pixel_box": _grid_box_to_pixel_box(
                        grid_box,
                        image_width,
                        image_height,
                        grid_width,
                        grid_height,
                    ),
                    "top_k_similarity": float(top_k_scores[index]),
                    "maximum_similarity": float(max_scores[index]),
                }
            )
    if not rows:
        raise ValueError("No valid local windows could be generated")
    rows.sort(key=lambda row: row["top_k_similarity"], reverse=True)
    strongest = rows[: min(best_windows_to_average, len(rows))]
    return {
        "local_score": float(np.mean([row["top_k_similarity"] for row in strongest])),
        "local_max_similarity": float(
            max(row["maximum_similarity"] for row in strongest)
        ),
        "best_window": rows[0],
        "local_method": "sliding_window_fallback",
    }


def score_mapped_source_boxes(
    extractor: DinoExtractor,
    feature_map: torch.Tensor,
    real_local_bank: torch.Tensor,
    source_boxes: list[YoloBox],
    source_size: tuple[int, int],
    candidate_size: tuple[int, int],
    padding_fraction: float,
    top_k: int,
) -> dict:
    source_width, source_height = source_size
    candidate_width, candidate_height = candidate_size
    x_scale = candidate_width / source_width
    y_scale = candidate_height / source_height
    rows = []
    for box in source_boxes:
        mapped = box.scale(x_scale, y_scale).expand(
            candidate_width,
            candidate_height,
            padding_fraction,
        )
        embedding, grid_box = extractor.pool_pixel_box(
            feature_map,
            mapped,
            candidate_width,
            candidate_height,
        )
        query = embedding.unsqueeze(0)
        rows.append(
            {
                "embedding": embedding,
                "grid_box": grid_box,
                "pixel_box": (
                    round(mapped.x1),
                    round(mapped.y1),
                    round(mapped.x2),
                    round(mapped.y2),
                ),
                "top_k_similarity": float(
                    top_k_mean_similarity(query, real_local_bank, top_k)[0]
                ),
                "maximum_similarity": float(
                    maximum_similarity(query, real_local_bank)[0]
                ),
            }
        )
    rows.sort(key=lambda row: row["top_k_similarity"], reverse=True)
    return {
        "local_score": float(np.mean([row["top_k_similarity"] for row in rows])),
        "local_max_similarity": float(
            max(row["maximum_similarity"] for row in rows)
        ),
        "best_window": rows[0],
        "local_embedding": rows[0]["embedding"],
        "local_method": "mapped_source_yolo_box",
    }


def classify_candidate(
    *,
    global_score: float,
    local_score: float,
    global_max_real: float,
    local_max_real: float,
    source_similarity: float,
    accepted_synthetic_similarity: float | None,
    bank: FeatureBank,
    config: Config,
) -> tuple[str, str]:
    settings = config.section("filter")
    dino = config.section("dino")
    if global_score < bank.thresholds["global_lower"]:
        return "rejected", "global_outlier"
    if local_score < bank.thresholds["local_lower"]:
        return "rejected", "local_defect_mismatch"
    if source_similarity >= float(settings["max_source_similarity"]):
        return "rejected", "effectively_unchanged_source"
    if (
        accepted_synthetic_similarity is not None
        and accepted_synthetic_similarity
        >= float(settings["max_accepted_synthetic_similarity"])
    ):
        return "rejected", "synthetic_near_duplicate"
    if settings["use_notebook_real_duplicate_gate"] and (
        global_max_real >= bank.thresholds["global_duplicate"]
        or local_max_real >= bank.thresholds["local_duplicate"]
    ):
        return "manual_review", "possible_real_near_duplicate"
    if settings["use_review_margin"]:
        margin = float(dino["review_margin"])
        if (
            abs(global_score - bank.thresholds["global_lower"]) <= margin
            or abs(local_score - bank.thresholds["local_lower"]) <= margin
        ):
            return "manual_review", "score_near_threshold"
    return "accepted", "passed_global_and_local"


def _prepare_destination(root: Path, decision: str, filename: str) -> Path:
    for category in ("accepted", "rejected", "manual_review"):
        old_path = root / category / filename
        if old_path.exists():
            old_path.unlink()
    directory = root / decision
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def filter_candidates(config: Config) -> Path:
    dino = config.section("dino")
    dataset = config.section("dataset")
    filter_settings = config.section("filter")
    bank = FeatureBank.load(config.output_root / "feature_bank")
    manifest_path = config.output_root / "candidates" / "candidate_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Candidate manifest not found: {manifest_path}. Run generate first."
        )
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError("Candidate manifest is empty")

    weights = config.path("dino_weights")
    repo = config.path("dino_repo")
    assert weights is not None and repo is not None
    extractor = DinoExtractor(
        weights_path=weights,
        repo_path=repo,
        model_name=dino["model_name"],
        image_size=int(dino["image_size"]),
        device=dino["device"],
    )

    source_index = {
        str(Path(row.image_path).resolve()): int(row.global_index)
        for row in bank.global_metadata.itertuples(index=False)
    }
    accepted_globals: list[torch.Tensor] = []
    rows: list[dict] = []
    feature_ids: list[str] = []
    candidate_globals: list[torch.Tensor] = []
    candidate_locals: list[torch.Tensor] = []
    filtered_root = config.output_root / "filtered"

    for candidate in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="DINO filtering",
    ):
        image_path = Path(candidate.output_path)
        try:
            features = extractor.extract(image_path)
            global_query = features.global_embedding.unsqueeze(0)
            global_score = float(
                top_k_mean_similarity(global_query, bank.global_embeddings, int(dino["top_k"]))[0]
            )
            global_max_real = float(
                maximum_similarity(global_query, bank.global_embeddings)[0]
            )

            source_path = Path(candidate.source_path)
            with Image.open(source_path) as source_image:
                source_size = source_image.size
            source_boxes = read_yolo_boxes(
                Path(candidate.source_label_path),
                source_size[0],
                source_size[1],
            )
            if source_boxes:
                local = score_mapped_source_boxes(
                    extractor=extractor,
                    feature_map=features.feature_map,
                    real_local_bank=bank.local_embeddings,
                    source_boxes=source_boxes,
                    source_size=source_size,
                    candidate_size=(features.image_width, features.image_height),
                    padding_fraction=float(dataset["box_padding_fraction"]),
                    top_k=int(dino["top_k"]),
                )
            else:
                local = search_local_windows(
                    feature_map=features.feature_map,
                    real_local_bank=bank.local_embeddings,
                    window_sizes=bank.window_sizes,
                    stride=int(dino["window_stride"]),
                    best_windows_to_average=int(dino["best_windows_to_average"]),
                    top_k=int(dino["top_k"]),
                    image_width=features.image_width,
                    image_height=features.image_height,
                )
                local["local_embedding"] = extractor.pool_grid_region(
                    features.feature_map,
                    local["best_window"]["grid_box"],
                )

            normalized_source = str(source_path.resolve())
            if normalized_source not in source_index:
                raise ValueError(f"Source image is not present in feature bank: {source_path}")
            source_embedding = bank.global_embeddings[source_index[normalized_source]]
            source_similarity = float(
                F.cosine_similarity(
                    features.global_embedding.unsqueeze(0),
                    source_embedding.unsqueeze(0),
                )[0]
            )
            accepted_similarity = None
            if accepted_globals:
                accepted_similarity = float(
                    maximum_similarity(
                        global_query,
                        torch.stack(accepted_globals),
                    )[0]
                )

            decision, reason = classify_candidate(
                global_score=global_score,
                local_score=local["local_score"],
                global_max_real=global_max_real,
                local_max_real=local["local_max_similarity"],
                source_similarity=source_similarity,
                accepted_synthetic_similarity=accepted_similarity,
                bank=bank,
                config=config,
            )
            if decision == "accepted":
                accepted_globals.append(features.global_embedding)

            destination = _prepare_destination(
                filtered_root,
                decision,
                image_path.name,
            )
            if filter_settings["copy_files"]:
                shutil.copy2(image_path, destination)
            else:
                destination = image_path

            x1, y1, x2, y2 = local["best_window"]["pixel_box"]
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "source_path": str(source_path),
                    "source_cluster": int(candidate.source_cluster),
                    "output_path": str(image_path),
                    "destination_path": str(destination.resolve()),
                    "decision": decision,
                    "reason": reason,
                    "global_top_k_similarity": global_score,
                    "global_max_real_similarity": global_max_real,
                    "local_top_k_similarity": local["local_score"],
                    "local_max_real_similarity": local["local_max_similarity"],
                    "source_global_similarity": source_similarity,
                    "accepted_synthetic_max_similarity": accepted_similarity,
                    "local_method": local["local_method"],
                    "best_window_x1": x1,
                    "best_window_y1": y1,
                    "best_window_x2": x2,
                    "best_window_y2": y2,
                    "best_window_similarity": local["best_window"]["top_k_similarity"],
                }
            )
            feature_ids.append(str(candidate.candidate_id))
            candidate_globals.append(features.global_embedding)
            candidate_locals.append(local["local_embedding"])
        except Exception as error:
            destination = _prepare_destination(
                filtered_root,
                "manual_review",
                image_path.name,
            )
            if image_path.exists() and filter_settings["copy_files"]:
                shutil.copy2(image_path, destination)
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "source_path": candidate.source_path,
                    "source_cluster": candidate.source_cluster,
                    "output_path": str(image_path),
                    "destination_path": str(destination.resolve()),
                    "decision": "manual_review",
                    "reason": f"processing_error: {error}",
                }
            )

    results = pd.DataFrame(rows)
    filtered_root.mkdir(parents=True, exist_ok=True)
    results_path = filtered_root / "filter_results.csv"
    results.to_csv(results_path, index=False)
    if candidate_globals:
        np.savez_compressed(
            filtered_root / "candidate_features.npz",
            candidate_ids=np.asarray(feature_ids),
            global_embeddings=torch.stack(candidate_globals).numpy(),
            local_embeddings=torch.stack(candidate_locals).numpy(),
        )
    summary = (
        results.groupby(["decision", "reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    (filtered_root / "filter_summary.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Filter results: {results_path}")
    return results_path
