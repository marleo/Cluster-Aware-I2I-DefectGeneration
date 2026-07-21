from __future__ import annotations

import argparse
from pathlib import Path

from cluster_defects.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster-aware SD 1.5 img2img generation with DINOv3 filtering."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.toml")),
        help="Path to the TOML configuration file.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("build-bank", help="Build real DINO feature banks and clusters.")

    generate = subcommands.add_parser(
        "generate",
        help="Generate cluster-balanced img2img candidates through ComfyUI.",
    )
    generate.add_argument("--sources-per-cluster", type=int)
    generate.add_argument("--variants-per-source", type=int)

    subcommands.add_parser("filter", help="Filter generated candidates with DINO.")
    subcommands.add_parser("plot", help="Create t-SNE and MMD distribution report.")

    run = subcommands.add_parser("run", help="Run build, generate, filter, and plot.")
    run.add_argument("--sources-per-cluster", type=int)
    run.add_argument("--variants-per-source", type=int)
    run.add_argument(
        "--reuse-bank",
        action="store_true",
        help="Use an existing feature bank instead of rebuilding it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Config.load(args.config)

    if args.command == "build-bank":
        from cluster_defects.feature_bank import build_feature_bank

        build_feature_bank(config)
    elif args.command == "generate":
        from cluster_defects.generate import generate_candidates

        generate_candidates(
            config,
            sources_per_cluster=args.sources_per_cluster,
            variants_per_source=args.variants_per_source,
        )
    elif args.command == "filter":
        from cluster_defects.filtering import filter_candidates

        filter_candidates(config)
    elif args.command == "plot":
        from cluster_defects.report import create_distribution_report

        create_distribution_report(config)
    elif args.command == "run":
        from cluster_defects.feature_bank import build_feature_bank
        from cluster_defects.filtering import filter_candidates
        from cluster_defects.generate import generate_candidates
        from cluster_defects.report import create_distribution_report

        bank_path = config.output_root / "feature_bank" / "real_features.npz"
        if not args.reuse_bank or not bank_path.exists():
            build_feature_bank(config)
        generate_candidates(
            config,
            sources_per_cluster=args.sources_per_cluster,
            variants_per_source=args.variants_per_source,
        )
        filter_candidates(config)
        create_distribution_report(config)


if __name__ == "__main__":
    main()

