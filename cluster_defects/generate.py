from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd

from .comfy_client import ComfyClient
from .config import Config
from .feature_bank import FeatureBank
from .workflow import build_img2img_api_workflow


MANIFEST_COLUMNS = [
    "candidate_id",
    "source_path",
    "source_label_path",
    "source_cluster",
    "seed",
    "checkpoint",
    "lora",
    "lora_model_strength",
    "lora_clip_strength",
    "positive_prompt",
    "negative_prompt",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "denoise",
    "prompt_id",
    "output_path",
]


def select_cluster_sources(
    bank: FeatureBank,
    source_split: str,
    sources_per_cluster: int,
    random_seed: int,
) -> pd.DataFrame:
    candidates = bank.global_metadata[
        bank.global_metadata["split"] == source_split
    ].copy()
    if candidates.empty:
        raise ValueError(f"No feature-bank sources found for split={source_split!r}")

    selected = []
    for cluster_id, group in candidates.groupby("cluster_id", sort=True):
        count = min(sources_per_cluster, len(group))
        selected.append(
            group.sample(
                n=count,
                random_state=random_seed + int(cluster_id),
                replace=False,
            )
        )
    return pd.concat(selected, ignore_index=True)


def _append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in MANIFEST_COLUMNS})


def generate_candidates(
    config: Config,
    sources_per_cluster: int | None = None,
    variants_per_source: int | None = None,
) -> Path:
    generation = config.section("generation")
    comfy = config.section("comfyui")
    dataset = config.section("dataset")
    bank = FeatureBank.load(config.output_root / "feature_bank")

    sources_per_cluster = (
        int(generation["sources_per_cluster"])
        if sources_per_cluster is None
        else sources_per_cluster
    )
    variants_per_source = (
        int(generation["variants_per_source"])
        if variants_per_source is None
        else variants_per_source
    )
    selected = select_cluster_sources(
        bank,
        dataset["source_split"],
        sources_per_cluster,
        int(config.get("clustering", "random_state")),
    )

    client = ComfyClient(
        comfy["server_url"],
        request_timeout=float(comfy["request_timeout_seconds"]),
        generation_timeout=float(comfy["generation_timeout_seconds"]),
        poll_interval=float(comfy["poll_interval_seconds"]),
    )
    stats = client.check_ready()
    print(
        f"ComfyUI ready: {stats['system']['comfyui_version']} "
        f"with PyTorch {stats['system']['pytorch_version']}"
    )

    output_root = config.output_root / "candidates"
    image_root = output_root / "images"
    manifest_path = output_root / "candidate_manifest.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        used_ids = set(existing["candidate_id"].astype(str))
    else:
        used_ids = set()

    seed_rng = random.Random(int(generation["seed"]))
    for source in selected.itertuples(index=False):
        source_path = Path(source.image_path)
        remote_name = (
            f"cluster_img2img/cluster_{int(source.cluster_id):02d}/"
            f"{source_path.stem}{source_path.suffix.lower()}"
        )
        uploaded_name = client.upload_image(source_path, remote_name)

        for variant in range(variants_per_source):
            seed = seed_rng.randrange(0, 2**63 - 1)
            candidate_id = (
                f"c{int(source.cluster_id):02d}_{source_path.stem}_"
                f"v{variant + 1:03d}_{seed}"
            )
            if candidate_id in used_ids:
                continue
            prefix = (
                f"cluster_img2img/cluster_{int(source.cluster_id):02d}/"
                f"{candidate_id}"
            )
            workflow = build_img2img_api_workflow(
                input_image=uploaded_name,
                checkpoint=generation["checkpoint"],
                lora=generation["lora"],
                lora_model_strength=float(generation["lora_model_strength"]),
                lora_clip_strength=float(generation["lora_clip_strength"]),
                positive_prompt=generation["positive_prompt"],
                negative_prompt=generation["negative_prompt"],
                width=int(generation["width"]),
                height=int(generation["height"]),
                seed=seed,
                steps=int(generation["steps"]),
                cfg=float(generation["cfg"]),
                sampler_name=generation["sampler_name"],
                scheduler=generation["scheduler"],
                denoise=float(generation["denoise"]),
                filename_prefix=prefix,
            )
            print(
                f"Generating {candidate_id} from cluster {int(source.cluster_id)} "
                f"source {source_path.name}"
            )
            prompt_id = client.queue_prompt(workflow)
            outputs = client.wait_for_outputs(prompt_id)
            output_descriptor = outputs[-1]
            suffix = Path(output_descriptor["filename"]).suffix or ".png"
            local_output = image_root / f"{candidate_id}{suffix}"
            client.download_image(output_descriptor, local_output)

            row = {
                "candidate_id": candidate_id,
                "source_path": str(source_path),
                "source_label_path": source.label_path,
                "source_cluster": int(source.cluster_id),
                "seed": seed,
                "checkpoint": generation["checkpoint"],
                "lora": generation["lora"],
                "lora_model_strength": generation["lora_model_strength"],
                "lora_clip_strength": generation["lora_clip_strength"],
                "positive_prompt": generation["positive_prompt"],
                "negative_prompt": generation["negative_prompt"],
                "width": generation["width"],
                "height": generation["height"],
                "steps": generation["steps"],
                "cfg": generation["cfg"],
                "sampler_name": generation["sampler_name"],
                "scheduler": generation["scheduler"],
                "denoise": generation["denoise"],
                "prompt_id": prompt_id,
                "output_path": str(local_output.resolve()),
            }
            _append_manifest(manifest_path, row)
            used_ids.add(candidate_id)

    print(f"Candidate manifest: {manifest_path}")
    return manifest_path
