from __future__ import annotations

import argparse

from coverage_pipeline.ml.registry import approve_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Approve staging model for production")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--approved-by", required=True)
    args = parser.parse_args()

    approve_model(args.model_version, args.approved_by)
    print(f"approved_model={args.model_version}")


if __name__ == "__main__":
    main()
