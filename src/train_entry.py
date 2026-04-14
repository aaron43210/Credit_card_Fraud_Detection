"""Terminal entrypoint for end-to-end training outside notebook.

This avoids notebook UI crashes from large logs and allows safer GPU control.
"""

import argparse
import os

from src.data_loader import load_raw_data
from src.preprocessing import run_preprocessing_pipeline
from src.training import train_all_models
from src.utils import get_logger

logger = get_logger("TrainEntry")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full credit card fraud detection training pipeline.")
    parser.add_argument("--sample-frac", type=float, default=None, help="Optional sampling fraction for quick runs.")
    parser.add_argument("--allow-hgnn-fallback", action="store_true", help="Enable automatic MLP fallback when HGNN is too large or memory is low.")
    parser.add_argument("--strict-gpu-only", action="store_true", help="Require CUDA for HGNN training and disable all CPU fallback.")
    args = parser.parse_args()

    if args.allow_hgnn_fallback:
        os.environ["ALLOW_HGNN_FALLBACK"] = "1"
    if args.strict_gpu_only:
        os.environ["STRICT_GPU_ONLY"] = "1"

    logger.info("Loading data...")
    df = load_raw_data(sample_frac=args.sample_frac)

    logger.info("Running preprocessing...")
    prep = run_preprocessing_pipeline(df)

    logger.info("Training all models...")
    train_all_models(
        prep["X_train"],
        prep["y_train"],
        prep["X_val"],
        prep["y_val"],
        hetero_data=prep.get("hetero_data"),
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
