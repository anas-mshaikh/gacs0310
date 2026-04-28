#!/usr/bin/env python3
"""
GACS Research Pipeline - CLI Entry Point
==========================================
Unified command-line interface for the entire GACS pipeline.

Usage:
    # Run the full pipeline (dataset → generate → experiment)
    python run_pipeline.py all

    # Run individual stages
    python run_pipeline.py dataset          # Build GACS dataset
    python run_pipeline.py generate         # Generate GACS + baseline videos
    python run_pipeline.py upload           # Upload to YouTube
    python run_pipeline.py metrics          # Collect experiment metrics
    python run_pipeline.py analyze          # Statistical analysis + report

    # Quick test (1 video, limited processing)
    python run_pipeline.py all --quick-test

    # Show system info
    python run_pipeline.py info
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gacs_config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    GPU_AVAILABLE,
    GPU_DEVICE,
    GPU_NAME,
    GPU_VRAM_GB,
    LABELING_BACKEND,
    LOCAL_VISION_MODEL,
    QUICK_TEST_MODE,
    SBERT_MODEL,
    SCHEMA_VERSION,
    USE_NVENC,
    USE_RAPIDS,
    setup_logging,
    validate_config,
)

logger = setup_logging("gacs_pipeline")


def show_info():
    """Display system and configuration information."""
    print("=" * 60)
    print("GACS Research Pipeline — System Info")
    print("=" * 60)

    print(f"\n  Schema Version:    {SCHEMA_VERSION}")
    print(f"  Quick-Test Mode:   {QUICK_TEST_MODE}")

    print("\n  --- GPU ---")
    print(f"  CUDA Available:    {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"  GPU Name:          {GPU_NAME}")
        print(f"  VRAM:              {GPU_VRAM_GB} GB")
    print(f"  Device:            {GPU_DEVICE}")
    print(f"  NVENC Encoding:    {USE_NVENC}")
    print(f"  RAPIDS cuML:       {USE_RAPIDS}")

    print("\n  --- Models ---")
    print(f"  Labeling Backend:  {LABELING_BACKEND}")
    if LABELING_BACKEND == "claude":
        print(f"  Claude Model:      {CLAUDE_MODEL}")
        print(f"  API Key Set:       {bool(ANTHROPIC_API_KEY)}")
    else:
        print(f"  Local Model:       {LOCAL_VISION_MODEL}")
    print(f"  SBERT Model:       {SBERT_MODEL}")

    print("\n  --- Config Validation ---")
    ok = validate_config()
    print(f"  All checks pass:   {ok}")
    print()


def run_dataset(args):
    """Run the dataset builder stage."""
    logger.info("=== Stage 1: Dataset Builder ===")
    from src.dataset_builder import main as dataset_main

    # Build args namespace for dataset_builder
    ns = argparse.Namespace(
        input=args.manifest if hasattr(args, 'manifest') else "video_manifest.csv",
        output="gacs_dataset.csv",
        quick_test=args.quick_test,
        skip_labeling=getattr(args, 'skip_labeling', False),
        visualize=not getattr(args, 'no_viz', False),
    )
    dataset_main(ns)
    logger.info("Dataset building complete.")


def run_generate(args):
    """Run the video generator stage."""
    logger.info("=== Stage 2: Video Generator ===")
    from src.video_generator import run_pipeline as generator_run

    if getattr(args, 'quick_test', False):
        import os
        os.environ['GACS_QUICK_TEST'] = '1'

    results = generator_run(
        num_clusters=getattr(args, 'clusters', 5),
        output_dir=None,
        generate_baseline=not getattr(args, 'no_baseline', False),
        visualize=not getattr(args, 'no_viz', False),
    )

    if results and results.get('gacs'):
        logger.info(f"Generated {len(results['gacs'])} GACS videos")
    if results and results.get('baseline'):
        logger.info(f"Generated {len(results['baseline'])} baseline videos")

    logger.info("Video generation complete.")


def run_upload(args):
    """Run the YouTube upload stage."""
    logger.info("=== Stage 3: YouTube Upload ===")
    from src.experiment_runner import main as experiment_main

    ns = argparse.Namespace(
        command="upload",
        quick_test=args.quick_test,
    )
    experiment_main(ns)
    logger.info("Upload complete.")


def run_metrics(args):
    """Run the metrics collection stage."""
    logger.info("=== Stage 4: Metrics Collection ===")
    from src.experiment_runner import main as experiment_main

    ns = argparse.Namespace(command="metrics", quick_test=args.quick_test)
    experiment_main(ns)
    logger.info("Metrics collection complete.")


def run_analyze(args):
    """Run the analysis and report stage."""
    logger.info("=== Stage 5: Analysis & Report ===")
    from src.experiment_runner import main as experiment_main

    ns = argparse.Namespace(command="analyze", quick_test=args.quick_test)
    experiment_main(ns)

    ns_report = argparse.Namespace(command="report", quick_test=args.quick_test)
    experiment_main(ns_report)
    logger.info("Analysis and reporting complete.")


def run_all(args):
    """Run the full pipeline end-to-end."""
    start = time.time()
    logger.info("Starting full GACS pipeline")
    show_info()

    run_dataset(args)
    run_generate(args)

    if not getattr(args, 'skip_upload', False):
        run_upload(args)
        run_metrics(args)
        run_analyze(args)
    else:
        logger.info("Skipping YouTube upload/metrics/analysis (--skip-upload)")

    elapsed = time.time() - start
    logger.info(f"Full pipeline completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="GACS Research Pipeline — Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py info                 # Show system info
  python run_pipeline.py all --quick-test     # Quick test full pipeline
  python run_pipeline.py dataset              # Build dataset only
  python run_pipeline.py generate --clusters 3 # Generate with 3 clusters
  python run_pipeline.py all --skip-upload    # Pipeline without YouTube
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # info
    subparsers.add_parser("info", help="Show system and configuration info")

    # dataset
    p_ds = subparsers.add_parser("dataset", help="Build GACS dataset from videos")
    p_ds.add_argument("--manifest", default="video_manifest.csv", help="Video manifest CSV")
    p_ds.add_argument("--skip-labeling", action="store_true", help="Use cached labels only")
    p_ds.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    p_ds.add_argument("--quick-test", action="store_true", help="Quick test mode")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate GACS + baseline videos")
    p_gen.add_argument("--clusters", type=int, default=5, help="Number of mood clusters")
    p_gen.add_argument("--no-baseline", action="store_true", help="Skip baseline videos")
    p_gen.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    p_gen.add_argument("--quick-test", action="store_true", help="Quick test mode")

    # upload
    p_up = subparsers.add_parser("upload", help="Upload videos to YouTube")
    p_up.add_argument("--quick-test", action="store_true", help="Quick test mode")

    # metrics
    p_met = subparsers.add_parser("metrics", help="Collect YouTube metrics")
    p_met.add_argument("--quick-test", action="store_true", help="Quick test mode")

    # analyze
    p_an = subparsers.add_parser("analyze", help="Run analysis and generate report")
    p_an.add_argument("--quick-test", action="store_true", help="Quick test mode")

    # all
    p_all = subparsers.add_parser("all", help="Run full pipeline end-to-end")
    p_all.add_argument("--skip-upload", action="store_true", help="Skip YouTube stages")
    p_all.add_argument("--skip-labeling", action="store_true", help="Use cached labels")
    p_all.add_argument("--clusters", type=int, default=5, help="Number of mood clusters")
    p_all.add_argument("--no-baseline", action="store_true", help="Skip baselines")
    p_all.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    p_all.add_argument("--quick-test", action="store_true", help="Quick test mode")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    handlers = {
        "info": lambda a: show_info(),
        "dataset": run_dataset,
        "generate": run_generate,
        "upload": run_upload,
        "metrics": run_metrics,
        "analyze": run_analyze,
        "all": run_all,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()
