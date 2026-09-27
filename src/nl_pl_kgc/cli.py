from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_mock_evaluation
from .train import evaluate_model, train_adapter


def _load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for training and evaluation") from exc
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural-language vs code-prompt KGC experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    mock = sub.add_parser("smoke-test", help="Test data, prompts, parsers and metrics without a model")
    mock.add_argument("--dataset", required=True)
    mock.add_argument("--output-dir", default="outputs/smoke-test")

    train = sub.add_parser("train", help="Train one or both LoRA prompt-format conditions")
    train.add_argument("--config", default="configs/ade_mistral7b.yaml")
    train.add_argument("--format", choices=["natural", "code", "both"], default="both")

    evaluate = sub.add_parser("evaluate", help="Evaluate a trained adapter")
    evaluate.add_argument("--config", default="configs/ade_mistral7b.yaml")
    evaluate.add_argument("--format", choices=["natural", "code"], required=True)
    evaluate.add_argument("--adapter", required=True)

    args = parser.parse_args()
    if args.command == "smoke-test":
        print(json.dumps(run_mock_evaluation(args.dataset, args.output_dir), indent=2))
        return

    config = _load_config(args.config)
    if args.command == "train":
        formats = ["natural", "code"] if args.format == "both" else [args.format]
        for prompt_format in formats:
            print(f"Saved {prompt_format} adapter to {train_adapter(config, prompt_format)}")
        return

    metrics = evaluate_model(config, args.format, args.adapter)
    print(json.dumps(metrics.as_dict(), indent=2))


if __name__ == "__main__":
    main()
